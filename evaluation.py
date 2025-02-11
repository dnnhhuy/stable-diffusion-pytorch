import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
from typing import Optional, List
import argparse
from utils import load_model, load_lora_weights
from models import get_lora_model, enable_lora
from tqdm import tqdm
import json
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def initialize_writer(log_dir, model_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
    return writer

from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")
def calculate_clip_score(images, prompts):
    clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

fid_calculator = FrechetInceptionDistance(normalize=False,
                                        input_img_size=(3, 512, 512))

def generate_fake_imgs(args,
                       model,
                       tokenizer,
                       original_imgs_dir, 
                       label_file, 
                       save_dir,
                       num_samples,
                       test_configs,
                       cfg_scales):
    
    with open(label_file, "r") as json_file:
        annotation_dict =  json.loads(json_file.read())
    
    id2filename = {}
    for img in annotation_dict["images"]:
        id2filename[img["id"]] = img["file_name"]
    
    prompts_dict = {}
    for anno in annotation_dict["annotations"]:
        prompts_dict[anno["caption"]] = {"image_id": anno["image_id"],
                                        "file_name": id2filename[anno["image_id"]],
                                        "id": anno["id"]}
        
    random_promtps = random.sample(list(prompts_dict.keys()), k=num_samples)
    
    os.makedirs(f"{save_dir}/original", exist_ok=True)

    for config in test_configs:
        args.sampler = config["sampler"]
        args.use_cosine_schedule = config["use_cosine_schedule"]
        for cfg_scale in cfg_scales:
            args.cfg_scale = cfg_scale
            if args.use_cosine_schedule:
                save_fake_folder = f"{args.cfg_scale}_{args.sampler}_cosineSchedule"
            else:
                save_fake_folder = f"{args.cfg_scale}_{args.sampler}_linearSchedule"
            os.makedirs(f"{save_dir}/fake/{save_fake_folder}", exist_ok=True)
            # writer = initialize_writer(log_dir="./runs/", model_name=save_fake_folder)
            # avgClipScore = 0.0
            for prompt in tqdm(random_promtps, leave=False, desc=f"Evaluate config: cfg_scale: {args.cfg_scale}, sampler: {args.sampler}, use_cosine_schedule: {args.use_cosine_schedule}"):
                pil_img = Image.open(os.path.join(original_imgs_dir, prompts_dict[prompt]["file_name"])).resize((512, 512))
                
                fake_img = model.generate(
                                        input_image=None,
                                        img_size=(512, 512),
                                        prompt=prompt,
                                        uncond_prompt="",
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
                
                pil_img.save(os.path.join(f"{save_dir}/original", f"{prompts_dict[prompt]['image_id']}_{prompts_dict[prompt]['id']}.jpg"))
                Image.fromarray(fake_img).save(os.path.join(f"{save_dir}/fake/{save_fake_folder}", f"{prompts_dict[prompt]['image_id']}_{prompts_dict[prompt]['id']}.jpg"))
                with open(os.path.join(save_dir, f"{prompts_dict[prompt]['image_id']}_{prompts_dict[prompt]['id']}.txt"), "w") as f:
                    f.write(prompt)
                
        #         avgClipScore +=  calculate_clip_score(np.expand_dims(fake_img, axis=0), [prompt])
        #         fid_calculator.update(torch.from_numpy(np.array(pil_img)).unsqueeze(0).permute(0, 3, 1, 2), real=True)
        #         fid_calculator.update(torch.from_numpy(fake_img).unsqueeze(0).permute(0, 3, 1, 2), real=False)
                
        #     avgClipScore /= len(random_promtps)
            
        #     print(avgClipScore, fid_calculator.compute())
        #     writer.add_scalars(main_tag="CLIP Score vs FID", 
        #                        tag_scalar_dict={"clip_score": avgClipScore, "fid": fid_calculator.compute()}, 
        #                        global_step=cfg_scale)
        #     fid_calculator.reset()
        # writer.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Arguments')
    parser.add_argument('--model_path', help='Model path', metavar="", default='')
    parser.add_argument('--tokenizer_dir', metavar="", default='', help='Tokenizer dir')
    parser.add_argument('--device', metavar="", default='cpu', type=str, help='Choose device to train')
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
    
    if args.model_path and args.tokenizer_dir:
        model, tokenizer = load_model(args)
    else:
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
        
            
    # output_images = inference(args, model, tokenizer, input_image)
    test_configs = [{"sampler": "ddpm", "use_cosine_schedule": False},
                    {"sampler": "ddpm", "use_cosine_schedule": True},
                    {"sampler": "ddim", "use_cosine_schedule": False},
                    {"sampler": "ddim", "use_cosine_schedule": True}]
    cfg_scales = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    generate_fake_imgs(args, 
                        model, 
                        tokenizer,
                        original_imgs_dir="./data/coco2017/val2017/", 
                        label_file="./data/coco2017/annotations/captions_val2017.json", 
                        num_samples=1000,
                        save_dir="evaluation_data",
                        test_configs=test_configs,
                        cfg_scales=cfg_scales)
        