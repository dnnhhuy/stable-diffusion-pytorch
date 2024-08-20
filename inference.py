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

def create_model():
    model = StableDiffusion(model_type='txt2img')
    loaded_state_dict = load_weights_from_ckpt('./weights/model/v1-5-pruned-emaonly.ckpt', device='cpu')
    model.vae.load_state_dict(loaded_state_dict['vae'], strict=True)
    model.unet.load_state_dict(loaded_state_dict['unet'], strict=True)
    model.cond_encoder.load_state_dict(loaded_state_dict['cond_encoder'], strict=True)
    return model

def create_tokenizer(): 
    tokenizer = CLIPTokenizer('./weights/tokenizer/tokenizer_vocab.json', merges_file='./weights/tokenizer/tokenizer_merges.txt')
    return tokenizer


def inference(args, input_image: Optional[Image.Image] = None):
    start_time = time.time()
    model = create_model()
    print(f"Loaded model in {time.time() - start_time:.2f}s")
    start_time = time.time()
    tokenizer = create_tokenizer()
    print(f"Loaded tokenizer in {time.time() - start_time:.2f}s")
    
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
    
    return output_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--device', default='cpu', type=str, help='Choose device to train')
    parser.add_argument('--img_size', default=512, type=int, help='Image size')
    parser.add_argument('--img_path', default='', type=str, help="Image path")
    parser.add_argument('--prompt', default='', type=str, help='Input prompt')
    parser.add_argument('--uncond_prompt', default='', type=str, help='Unconditional prompt')
    
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
        
    output = inference(args, input_image)
    
    output = Image.fromarray(output)
    
    if not os.path.exists('./output'):
        os.mkdir('./output/')
        
    output.save('./output/output.jpg')