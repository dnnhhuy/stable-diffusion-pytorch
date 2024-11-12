import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
from typing import Optional
from PIL import Image
import argparse
from utils import load_model, create_tokenizer
from utils import load_lora_weights
from models import get_lora_model, enable_lora

def inference(args, model, tokenizer, input_image: Optional[Image.Image] = None):
    output_images = []
    for i in range(args.n_samples):
        output_image = model.generate(
            input_image=input_image,
            img_size=(args.img_size, args.img_size),
            prompt=args.prompt,
            uncond_prompt=args.uncond_prompt,
            do_cfg=args.do_cfg,
            cfg_scale=args.cfg_scale,
            device=args.device,
            strength=args.strength,
            inference_steps=args.num_inference_steps,
            sampler=args.sampler,
            use_cosine_schedule=args.use_cosine_schedule,
            seed=args.seed,
            tokenizer=tokenizer
        )
        output_images.append(output_image)
    
    if not os.path.exists("./output"):
        os.makedirs("./output")
    
    for i, img in enumerate(output_images):
        Image.fromarray(img).save(f"./output/img_{i}.jpg")
    
    return output_images

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
    
    args = parser.parse_args()
    args.seed = None
    
    if args.do_cfg is None:
        args.do_cfg = False
        
    if args.use_cosine_schedule is None:
        args.use_cosine_schedule = False
        
    input_image = None
    if os.path.exists(args.img_path):
        input_image = Image.open(args.img_path)
    
    if args.model_path and args.tokenizer_dir:
        model, tokenizer = load_model(args)
    else:
        HF_TOKEN_KEY = os.getenv("HF_TOKEN_KEY")
        from huggingface_hub import hf_hub_download
        files_to_download = ["v1-5-pruned-emaonly.ckpt",
                             "tokenizer/merges.txt",
                             "tokenizer/vocab.json"]
        for file in files_to_download:
            file = file.split('/')
            if len(file) > 1:
                hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                                subfolder=file[0],
                                filename=file[-1],
                                local_dir="./weights/model/")
            else:
                hf_hub_download(repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                                filename=file[0],
                                local_dir="./weights/model/")
        
    
    if args.lora_ckpt.endswith(".safetensors"):
        model.unet = get_lora_model(model.unet, rank=128, alphas=128, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out', 'conv_input', 'conv_output', 'transformer_block.ffn.0.proj', 'transformer_block.ffn.1'])
        model.unet = enable_lora(model.unet, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out', 'conv_input', 'conv_output', 'transformer_block.ffn.0.proj', 'transformer_block.ffn.1'], enabled=True)
        model.cond_encoder = get_lora_model(model.cond_encoder, rank=128, alphas=128, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out', 'ffn.0', 'ffn.2'])
        model.cond_encoder = enable_lora(model.cond_encoder, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out', 'ffn.0', 'ffn.2'], enabled=True)
        
        state_dict = load_lora_weights(args.lora_ckpt)
        model.unet.load_state_dict(state_dict=state_dict["unet"], strict=False)
        model.cond_encoder.load_state_dict(state_dict=state_dict["cond_encoder"], strict=False)
        
    elif args.lora_ckpt.endswith(".ckpt"):
        model.unet = get_lora_model(model.unet, rank=8, alphas=16, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out'])
        model.unet = enable_lora(model.unet, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out'], enabled=True)
        
        model.load_state_dict(torch.load(args.lora_ckpt, map_location="cpu")["model_state_dict"], strict=False)
        
            
    output_images = inference(args, model, tokenizer, input_image)
        