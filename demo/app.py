import torch
import gradio as gr
from utils import load_model
from models import get_lora_model, enable_lora
import argparse
from PIL import Image
import os


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
def initialize_model():
    args = {"model_path": "./weights/model/v1-5-pruned-emaonly.ckpt",
            "tokenizer_dir": "./weights/tokenizer/"}
    args = argparse.Namespace(**args)
    
    if not os.path.exists("./weights/model"):
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
                
    model, tokenizer = load_model(args)
    
    model.unet = get_lora_model(model.unet, rank=32, alphas=16, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out'])
    model.unet = enable_lora(model.unet, lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out'], enabled=True)
    
    if os.path.exists("./weights/model/stable_diffusion_lora_epoch_24.ckpt"):
        model.load_state_dict(torch.load("./weights/model/stable_diffusion_lora_epoch_24.ckpt", map_location="cpu")['model_state_dict'], strict=False)
        
    return model, tokenizer

def txt2img(prompt, 
            uncond_prompt, 
            n_samples,
            use_cosine_beta_schedule,
            cfg_scale,
            strength,
            inference_steps,
            sampler):
    
    args = {"img_size": 512,
            "prompt": prompt,
            "uncond_prompt": uncond_prompt,
            "do_cfg": True,
            "cfg_scale": cfg_scale,
            "device": device,
            "strength": strength,
            "num_inference_steps": inference_steps,
            "sampler": sampler, 
            "use_cosine_schedule": use_cosine_beta_schedule,
            "seed": None}
    
    args = argparse.Namespace(**args)
    output_images = []
    for i in range(n_samples):
        output_img = model.generate(
            input_image=None,
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
            tokenizer=tokenizer,
            gr_progress_bar=gr.Progress(),
        )
        output_images.append(output_img)
    
    return [Image.fromarray(img) for img in output_images]

def img2img(input_images, 
            prompt, 
            uncond_prompt, 
            n_samples,
            use_cosine_beta_schedule,
            cfg_scale,
            strength,
            inference_steps,
            sampler):
    
    args = {"img_size": 512,
            "prompt": prompt,
            "uncond_prompt": uncond_prompt,
            "do_cfg": True,
            "cfg_scale": cfg_scale,
            "device": device,
            "strength": strength,
            "num_inference_steps": inference_steps,
            "sampler": sampler, 
            "use_cosine_schedule": use_cosine_beta_schedule,
            "seed": None}
    
    args = argparse.Namespace(**args)
    
    output_images = []
    
    for i in range(n_samples):
        output_img = model.generate(
            input_image=input_images,
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
            tokenizer=tokenizer,
            batch_size=1,
            gr_progress_bar=gr.Progress()
        )
        output_images.append(output_img)
    
    
    
    return [Image.fromarray(img) for img in output_images]

def inpaint(input_image, 
            prompt, 
            uncond_prompt, 
            n_samples,
            use_cosine_beta_schedule,
            cfg_scale,
            strength,
            inference_steps,
            sampler):

    args = {"img_size": 512,
            "prompt": prompt,
            "uncond_prompt": uncond_prompt,
            "do_cfg": True,
            "cfg_scale": cfg_scale,
            "device": device,
            "strength": strength,
            "num_inference_steps": inference_steps,
            "sampler": sampler, 
            "use_cosine_schedule": use_cosine_beta_schedule,
            "seed": None}
    
    input_img, mask, masked_img = input_image['background'].convert('RGB'), input_image['layers'][0].split()[-1], input_image['composite'].convert('RGB')
     
    args = argparse.Namespace(**args)
    
    output_images = []
    for i in range(n_samples):
    
        output_img = model.inpaint(
            input_image=masked_img,
            mask=mask,
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
            tokenizer=tokenizer,
            gr_progress_bar=gr.Progress()
        )
        output_images.append(output_img)
        
    return [Image.fromarray(img) for img in output_images]
    

demo = gr.Blocks().queue()
with demo:
    with gr.Row():
        gr.Markdown("## Stable Diffusion")
    with gr.Tab(label='txt2img'):
        with gr.Row():
            with gr.Column():
                
                prompt = gr.Textbox(label="Prompt")
                uncond_prompt = gr.Textbox(label="Uncondition prompt")
                n_samples = gr.Slider(label="Number of generated images", minimum=1, maximum=5, step=1, value=1)
                with gr.Accordion(label="Advanced settings", open=False):
                    cfg_scale = gr.Slider(minimum=0, maximum=10, label='CFG Scale', step=0.1, value=7.5)
                    strength = gr.Slider(label="Strength", minimum=0, maximum=1., step=0.01, value=0.8)
                    inference_steps = gr.Slider(label="Generation Steps", minimum=0, maximum=1000, step=1, value=50)
                    
                    sampler_choices = [
                        ("DDPM", "ddpm"),
                        ("DDIM", "ddim")
                    ]
                    sampler = gr.Dropdown(label='Sampling method', choices=sampler_choices, value="ddpm")
                    use_cosine_beta_schedule = gr.Checkbox(value=False, label="Use cosine-based beta schedule")


        with gr.Row():
            generate_button = gr.Button(value="Generate")
                
        with gr.Row():
            gallery = gr.Gallery(label="Generated images", show_label=False)
            
        generate_button.click(fn=txt2img, inputs=[prompt, 
                                                    uncond_prompt, 
                                                    n_samples,
                                                    use_cosine_beta_schedule,
                                                    cfg_scale,
                                                    strength,
                                                    inference_steps,
                                                    sampler], 
                                            outputs=[gallery])
    with gr.Tab('img2img'):
        with gr.Row(equal_height=True):
            
            input_images = gr.Image(sources='upload', type="pil")
            
            with gr.Column():
                
                prompt = gr.Textbox(label="Prompt")
                uncond_prompt = gr.Textbox(label="Uncondition prompt")
                
                n_samples = gr.Slider(label="Number of generated images", minimum=1, maximum=5, step=1, value=1)
                
        with gr.Row():
            with gr.Accordion(label="Advanced Settings", open=True):
                cfg_scale = gr.Slider(minimum=0, maximum=10, label='CFG Scale', step=0.1, value=7.5)
                strength = gr.Slider(label="Strength", minimum=0, maximum=1., step=0.01, value=0.8)
                inference_steps = gr.Slider(label="Generation Steps", minimum=0, maximum=1000, step=1, value=50)
                
                sampler_choices = [
                    ("DDPM", "ddpm"),
                    ("DDIM", "ddim")
                ]
                sampler = gr.Dropdown(label='Sampling method', choices=sampler_choices, value="ddpm")
                use_cosine_beta_schedule = gr.Checkbox(value=False, label="Use cosine-based beta schedule")


        with gr.Row():
            generate_button = gr.Button(value="Generate")
                
        with gr.Row():
            gallery = gr.Gallery(label="Generated images", show_label=False)
        generate_button.click(fn=img2img, inputs=[input_images, 
                                                    prompt, 
                                                    uncond_prompt, 
                                                    n_samples,
                                                    use_cosine_beta_schedule,
                                                    cfg_scale,
                                                    strength,
                                                    inference_steps,
                                                    sampler], 
                                            outputs=[gallery])
    
    with gr.Tab("inpaint"):
        with gr.Row():
            
            input_images = gr.ImageMask(sources="upload", type="pil", crop_size=(512, 512), scale=2)
            
            with gr.Column(scale=1):
                
                prompt = gr.Textbox(label="Prompt")
                uncond_prompt = gr.Textbox(label="Unconditional prompt")
                
                n_samples = gr.Slider(label="Number of generated images", minimum=1, maximum=5, step=1, value=1)
                
                with gr.Accordion(label="Advanced Settings", open=False):
                    cfg_scale = gr.Slider(minimum=0, maximum=10, label='CFG Scale', step=0.1, value=7.5)
                    strength = gr.Slider(label="Strength", minimum=0, maximum=1., step=0.01, value=0.8)
                    inference_steps = gr.Slider(label="Generation Steps", minimum=0, maximum=1000, step=1, value=50)
                    
                    sampler_choices = [
                        ("DDPM", "ddpm"),
                        ("DDIM", "ddim")
                    ]
                    sampler = gr.Dropdown(label='Sampling method', choices=sampler_choices, value="ddpm")
                    use_cosine_beta_schedule = gr.Checkbox(value=False, label="Use cosine-based beta scheduler")
                    
        with gr.Row():
            generate_button = gr.Button(value="Generate")
                
        with gr.Row():
            gallery = gr.Gallery(label="Generated images", show_label=False)
        generate_button.click(fn=inpaint, inputs=[input_images, 
                                                    prompt, 
                                                    uncond_prompt, 
                                                    n_samples,
                                                    use_cosine_beta_schedule,
                                                    cfg_scale,
                                                    strength,
                                                    inference_steps,
                                                    sampler], 
                                            outputs=[gallery])

if __name__ == '__main__':
    model, tokenizer = initialize_model()
    demo.launch()