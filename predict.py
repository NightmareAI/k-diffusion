from distutils.command.config import config
import gc
import io
import math
from pyexpat import model
import sys
from einops import rearrange
import numpy as np
import tempfile
import threading
import typing
import queue

from omegaconf import OmegaConf
import clip
import k_diffusion as K
import lpips
from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.append('./guided-diffusion')
sys.path.append('./latent-diffusion')
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from ldm.util import instantiate_from_config

from cog import BasePredictor, Input, Path

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


class GuidedDenoiserWithGrad(nn.Module):
    def __init__(self, model, cond_fn):
        super().__init__()
        self.inner_model = model
        self.cond_fn = cond_fn
        self.orig_denoised = None

    def forward(self, x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = self.inner_model(x, sigma, **kwargs)
            self.orig_denoised = denoised.detach()
            cond_grad = self.cond_fn(x, sigma, denoised=denoised, **kwargs)
        cond_denoised = denoised + cond_grad * K.utils.append_dims(sigma ** 2, x.ndim)
        return cond_denoised

class Predictor(BasePredictor):

    def LoadCompVisModel(self):
        self.model_config = OmegaConf.load("/src/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")
#        self.model_config.update({
#            'attention_resolutions': '32, 16, 8',
#            'class_cond': False,
#            'diffusion_steps': 1000,
#            'rescale_timesteps': True,
#            'timestep_respacing': '1000',
#            'learn_sigma': True,
#            'noise_schedule': 'linear',
#            'num_channels': 256,
#            'num_head_channels': 64,
#            'num_res_blocks': 2,
#            'resblock_updown': True,
#            'use_checkpoint': True,
#            'use_fp16': True,
#            'use_scale_shift_norm': True,
#        })
        self.model_config['image_size'] = 256
        self.model_path = "/root/.cache/k-diffusion/latent_diffusion_txt2img_f8_large.ckpt"        
        pl_sd = torch.load(self.model_path, map_location='cuda')
        sd = pl_sd["state_dict"]
        self.model = instantiate_from_config(self.model_config.model)
        m, u = self.model.load_state_dict(sd, strict=False)                

    def LoadOpenAIModel(self):
        self.model_config = model_and_diffusion_defaults()        
        self.model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': '1000',
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': False,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        })
        self.model_config['image_size'] = 256
        self.model_path = "/root/.cache/k-diffusion/256x256_diffusion_uncond.pt"        
        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cuda'))

    def setup(self):        
        self.device = torch.device('cuda')
        self.LoadCompVisModel()        
        self.model.requires_grad_(False).eval().to(self.device)
        #if self.model_config['use_fp16']:
        #    self.model.convert_to_fp16()
        
        self.clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(self.device)
        self.clip_size = self.clip_model.visual.input_resolution
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)

    def predict(
        self,
        text_prompt: str = Input(description="Prompt",default="A mysterious orb by Ernst Fuchs"),
        #model: str = Input(description="Diffusion Model",default="latent_diffusion_txt2img_f8_large.ckpt", choices=['latent_diffusion_txt2img_f8_large.ckpt','256x256_diffusion_uncond.pt']),
        init_image: Path = Input(description="Initial image for the generation",default=None),
        sigma_start: int = Input(description="The starting noise level when using an init image", default=10),
        init_scale: int = Input(description="This enhances the effect of the init image, a good value is 1000.", default=1000),
        image_prompt: Path = Input(description="Image prompt",default=None),
        #batch_size: int = Input(description="The number of generations to run",ge=1,le=10,default=1),
        n_steps: int = Input(description="The number of timesteps to use", ge=50,le=1000,default=500),
        clip_guidance_scale: int = Input(description="Controls how much the image should look like the prompt.", default=1000),
        tv_scale: int = Input(description="Controls the smoothness of the final output.", default=100),
        range_scale: int = Input(description="Controls how far out of range RGB values are allowed to be.", default=50),
        cutn: int = Input(description="The number of random crops per step.", default=16),
        cut_pow: float = Input(description="Cut power", default=0.5),        
        seed: int = Input(description="Seed (leave empty to use a random seed)", default=None, le=(2**32-1), ge=0),
    ) -> typing.Iterator[Path]:
        prompts = [text_prompt]
        self.text_prompt = text_prompt
        image_prompts = []
        if (image_prompt):
            image_prompts = [str(image_prompt)]
        if (init_image):
            init_image = str(init_image)
        
        n_batches = 1
       
                
        make_cutouts = MakeCutouts(self.clip_size, cutn, cut_pow)
        side_x = side_y = self.model_config['image_size']
        
        target_embeds, weights = [], []
        
                

        # do_run
        
        make_cutouts = MakeCutouts(self.clip_size, cutn, cut_pow)
        side_x = side_y = self.model_config['image_size']

        target_embeds, weights = [], []

        for prompt in prompts:            
            txt, weight = parse_prompt(prompt)
            target_embeds.append(self.clip_model.encode_text(clip.tokenize(txt).to(self.device)).float())
            weights.append(weight)

        for prompt in image_prompts:
            path, weight = parse_prompt(prompt)
            img = Image.open(fetch(path)).convert('RGB')
            img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.clip_model.encode_image(self.normalize(batch)).float()
            target_embeds.append(embed)
            weights.extend([weight / cutn] * cutn)

        target_embeds = torch.cat(target_embeds)
        weights = torch.tensor(weights, device=self.device)
        if weights.sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        weights /= weights.sum().abs()

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert('RGB')
            init = init.resize((side_x, side_y), Image.Resampling.LANCZOS)
            init = TF.to_tensor(init).to(self.device)[None] * 2 - 1

        def cond_fn(x, sigma, denoised, cond):
            n = x.shape[0]

            # Anti-grain hack for the 256x256 ImageNet model
            #fac = sigma / (sigma ** 2 + 1) ** 0.5
            #denoised_in = x.lerp(denoised, fac)

            #clip_in = self.normalize(make_cutouts(denoised_in.add(1).div(2)))
            clip_in = self.normalize(make_cutouts(model.decode_first_stage(denoised)))
            image_embeds = self.clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(image_embeds[:, None], target_embeds[None])
            dists = dists.view([cutn, n, -1])
            losses = dists.mul(weights).sum(2).mean(0)
            tv_losses = tv_loss(denoised)
            range_losses = range_loss(denoised)
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale + range_losses.sum() * range_scale
            if init is not None and init_scale:
                init_losses = self.lpips_model(denoised, init)
                loss = loss + init_losses.sum() * init_scale
            return -torch.autograd.grad(loss, x)[0]

        #model_wrap = K.external.OpenAIDenoiser(self.model, self.diffusion, device=self.device)
        self.model_wrap = K.external.CompVisDenoiser(self.model, False, device=self.device)
        sigmas = self.model_wrap.get_sigmas(n_steps)
        if init is not None:
            sigmas = sigmas[sigmas <= sigma_start]
        self.model_guided = GuidedDenoiserWithGrad(self.model_wrap, cond_fn)

        output = queue.SimpleQueue()

        def callback(info):
            if info['i'] % 50 == 0:                
            #    denoised = self.model_wrap.decode_first_stage(self.x)
            #    nrow = math.ceil(denoised.shape[0] ** 0.5)
            #    grid = utils.make_grid(denoised, nrow, padding=0)
            #    tqdm.write(f'Step {info["i"]} of {len(sigmas) - 1}:')                                
            #    filename = f'step_{i}.png'
            #    K.utils.to_pil_image(grid).save(filename)
            #    output.put(filename)
            #    #display.display(K.utils.to_pil_image(grid))
                tqdm.write(f'')

        if seed is not None:
            torch.manual_seed(seed)

        self.success = False
        
        for i in range(n_batches):
            self.side_y = side_y
            self.side_x = side_x
            self.sigmas = sigmas
            self.init = init
            self.callback = callback            
            t = threading.Thread(target=self.worker, daemon=True)
            t.start()
            while t.is_alive():
                try:
                    image = output.get(block=True, timeout=5)
                    yield Path(image)
                except: {}
            
        tqdm.write('Done!')
        if (not self.success):
            raise RuntimeError('No output, check logs for errors')
        
        samples = self.model.decode_first_stage(self.samples)
        samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
        
        for i, out in enumerate(samples):            
            sample = 255.0 * rearrange(out.cpu().numpy(), "c h w -> h w c")            
            filename = f'out_{i}.png'
            Image.fromarray(sample.astype(np.uint8)).save(filename)
            yield Path(filename)
    
    def worker(self):
        self.x = torch.randn([1, 4, self.side_y//8, self.side_x//8], device=self.device)
        if self.init is not None:
            self.x += self.init
        n_samples = 1        
        c = self.model.get_learned_conditioning(n_samples * [self.text_prompt])
        self.samples = K.sampling.sample_heun(self.model_wrap, self.x, self.sigmas, second_order=False, s_churn=20, callback=self.callback, extra_args={"cond": c})
        self.success = True
