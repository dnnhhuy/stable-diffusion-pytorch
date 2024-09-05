import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from models.diffusion import StableDiffusion
from transformers import CLIPTokenizer
from PIL import Image
from utils.model_converter import load_weights_from_ckpt
from typing import Optional
from PIL import Image
import argparse
import time
from utils.utils import load_model



def inference(args, input_image: Optional[Image.Image] = None):
    model, tokenizer = load_model(args)
    
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
    
    args = parser.parse_args()
    args.do_cfg = True
    args.cfg_scale = 7.5
    args.strength = 0.8
    args.num_inference_steps = 50
    args.sampler = 'ddpm'
    args.use_cosine_schedule = False
    args.seed = None
    
    input_image = None
    if os.path.exists(args.img_path):
        input_image = Image.open(args.img_path)
        
    output_images = inference(args, input_image)