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
    parser.add_argument('--model_path', default='./weights/model/v1-5-pruned-emaonly.ckpt', help='Model path')
    parser.add_argument('--tokenizer_dir', default='./weights/tokenizer/', help='Tokenizer dir')
    parser.add_argument('--device', default='cpu', type=str, help='Choose device to train')
    parser.add_argument('--img_size', default=512, type=int, help='Image size')
    parser.add_argument('--img_path', default='', type=str, help="Image path")
    parser.add_argument('--prompt', default='', type=str, help='Input prompt')
    parser.add_argument('--uncond_prompt', default='', type=str, help='Unconditional prompt')
    parser.add_argument('--n_samples', default=3, type=int, help='Number of generated images')
    parser.add_argument('--lora_ckpt', default='', type=str, help='Option to use lora checkpoint')
    
    args = parser.parse_args()
    args.do_cfg = True
    args.cfg_scale = 7.0
    args.strength = 1.0
    args.num_inference_steps = 30
    args.sampler = 'ddpm'
    args.use_cosine_schedule = False
    args.seed = None
    
    input_image = None
    if os.path.exists(args.img_path):
        input_image = Image.open(args.img_path)
    
    
    model, tokenizer = load_model(args)
    
    tokenizer = create_tokenizer(args.tokenizer_dir)
    
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
        