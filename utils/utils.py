import time

from models.diffusion import StableDiffusion
from transformers import CLIPTokenizer

def create_model(model_path: str):
    model = StableDiffusion.from_pretrained(model_path)
    return model

def create_tokenizer(tokenizer_dir):
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer

def load_model(args):
    start_time = time.time()
    model = create_model(args.model_path, sampler=args.sampler)
    print(f"Loaded model in {time.time() - start_time:.2f}s")
    start_time = time.time()
    tokenizer = create_tokenizer(args.tokenizer_dir)
    print(f"Loaded tokenizer in {time.time() - start_time:.2f}s")
    return model, tokenizer