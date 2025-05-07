import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import math
import torch
from PIL import Image
from typing import Optional, List
import argparse

from models.diffusion import StableDiffusion
from models.scheduler import DDIMSampler, DDPMSampler
from transformers import CLIPTokenizer
from utils.model_converter import load_lora_weights
from models.lora import get_lora_model, enable_lora
from torchvision.utils import save_image


def inference(args, model, tokenizer, input_image: Optional[Image.Image] = None) -> List[Image.Image]:
    outputs = []
    iterations = math.ceil(args.n_samples / args.batch_size)
    
    if not os.path.exists("./output"):
        os.makedirs("./output")
        
    for i in range(iterations):
        if args.one_step == False:
            generated_images = model.generate(
                input_image=input_image,
                img_size=(args.img_size, args.img_size),
                prompt=args.prompt,
                uncond_prompt=args.uncond_prompt,
                do_cfg=args.do_cfg,
                cfg_scale=args.cfg_scale,
                device=args.device,
                inference_steps=args.num_inference_steps,
                strength=args.strength,
                sampler=model.sampler,
                use_cosine_schedule=args.use_cosine_schedule,
                seed=args.seed,
                tokenizer=tokenizer,
                batch_size=args.batch_size)
        else:
            generated_images = model.generate_in_one_step(
                input_image=input_image,
                img_size=(args.img_size, args.img_size),
                prompt=args.prompt,
                uncond_prompt=args.uncond_prompt,
                do_cfg=args.do_cfg,
                cfg_scale=args.cfg_scale,
                device=args.device,
                sampler=model.sampler,
                use_cosine_schedule=args.use_cosine_schedule,
                seed=args.seed,
                tokenizer=tokenizer,
                batch_size=args.batch_size)
        
        for img in generated_images:
            save_image(img, f"./output/img_{i}.jpg")
            # Image.fromarray(img).save(f"./output/img_{i}.jpg")
            outputs.append(img)
    
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Arguments')
    parser.add_argument('--model_path', help='Model path', metavar="", default='')
    parser.add_argument('--tokenizer_dir', metavar="", default='', help='Tokenizer dir')
    parser.add_argument('--device', metavar="", default='cpu', type=str, help='Choose device to train')
    parser.add_argument('--img_size', metavar="", default=512, type=int, help='Image size')
    parser.add_argument('--img_path', metavar="", default='', type=str, help="Image path")
    parser.add_argument('--prompt', metavar="", default='', type=str, help='Input prompt')
    parser.add_argument('--uncond_prompt', metavar="", default='', type=str, help='Unconditional prompt')
    parser.add_argument('--n_samples', metavar="", default=3, type=int, help='Number of generated images')
    parser.add_argument('--lora_ckpt', metavar="", default='', type=str, help='Option to use lora checkpoint')
    parser.add_argument('--do_cfg', metavar="", action=argparse.BooleanOptionalAction, help='Activate CFG')
    parser.add_argument('--cfg_scale', metavar="", default=7.5, type=float, help="Set classifer-free guidance scale (larger value tends to focus on conditional prompt, smaller value tends to focus on unconditional prompt)")
    parser.add_argument('--strength', metavar="", default=1.0, type=float, help="Set the strength to generate the image (Given image from the user, the smaller value tends to generate an image closer to the original one)")
    parser.add_argument('--num_inference_steps', help="Step to generate image", default=50, choices=range(1, 1001), metavar="Value: [1-1000]", type=int)
    parser.add_argument('--sampler', metavar="", default='ddpm', choices=['ddpm', 'ddim'], type=str, help="Sampling method: 2 options available: DDPM and DDIM")
    parser.add_argument('--use_cosine_schedule', metavar="", action=argparse.BooleanOptionalAction, help="Activate using cosine function to generate beta values used for adding and remove noise from the image.")
    parser.add_argument('--batch_size', metavar="", default=1, type=int, help="Batch size")
    parser.add_argument('--seed', default=None, type=int, help="Seed value")
    parser.add_argument('--one_step', metavar="", action=argparse.BooleanOptionalAction, help='One step generation')
    parser.add_argument('--sd_version', default="1.5", type=str, help="Stable Diffusion Model Version")
    
    args = parser.parse_args()
    
    if args.do_cfg is None:
        args.do_cfg = False
        
    if args.use_cosine_schedule is None:
        args.use_cosine_schedule = False
    
    if args.one_step is None:
        args.one_step = False
        
    input_image = None
    if os.path.exists(args.img_path):
        input_image = Image.open(args.img_path)
    
    model = StableDiffusion.from_pretrained(args.model_path, device=args.device, sd_version=args.sd_version)
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_dir)
    if args.sampler == 'ddpm':
        model.sampler = DDPMSampler.from_config(cfg_path=os.path.join(args.model_path, "scheduler"), use_cosine_schedule=args.use_cosine_schedule, device=args.device) 
    elif args.sampler == 'ddim':
        model.sampler = DDIMSampler.from_config(cfg_path=os.path.join(args.model_path, "scheduler"), use_cosine_schedule=args.use_cosine_schedule, device=args.device)
    else:
        raise ValueError("Invalid sampler type. Choose either 'ddpm' or 'ddim'.")
    
    if args.lora_ckpt.endswith(".safetensors"):
        model.unet = get_lora_model(model.unet, rank=128, alphas=128, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out', 'conv_input', 'conv_output', 'transformer_block.ffn.0.proj', 'transformer_block.ffn.1'])
        model.unet = enable_lora(model.unet, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out', 'conv_input', 'conv_output', 'transformer_block.ffn.0.proj', 'transformer_block.ffn.1'], enabled=True)
        model.cond_encoder = get_lora_model(model.cond_encoder, rank=128, alphas=128, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out', 'ffn.0', 'ffn.2'])
        model.cond_encoder = enable_lora(model.cond_encoder, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out', 'ffn.0', 'ffn.2'], enabled=True)
        state_dict = load_lora_weights(args.lora_ckpt)
        model.unet.load_state_dict(state_dict=state_dict["unet"], strict=False)
        model.cond_encoder.load_state_dict(state_dict=state_dict["cond_encoder"], strict=False)
    elif args.lora_ckpt.endswith(".ckpt"):
        model.unet = get_lora_model(model.unet, rank=32, alphas=16, lora_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
        model.unet = enable_lora(model.unet, lora_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'], enabled=True)        
        state_dict = torch.load(args.lora_ckpt, map_location="cpu")["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
            
    output_images = inference(args, model, tokenizer, input_image)
        