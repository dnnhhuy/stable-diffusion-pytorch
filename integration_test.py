import torch
import unittest
from models.diffusion import StableDiffusion
from models.ddim import DDIMSampler

from transformers import CLIPTokenizer

from diffusers import StableDiffusionPipeline, AutoencoderKL

from torchvision.utils import save_image

class TestStableDiffusion(unittest.TestCase):
    def setUp(self):
        self.model = StableDiffusion(model_type='txt2img').from_pretrained("./weights/stable-diffusion-2-1")
        self.tokenizer = CLIPTokenizer.from_pretrained("./weights/stable-diffusion-2-1/tokenizer")
        self.sampler = DDIMSampler()
        
        self.unet = self.model.unet
        self.text_encoder = self.model.clip.text_model
        self.vae = self.model.vae
        
        self.unet.eval()
        self.text_encoder.eval()
        self.vae.eval()
        
        self.original_pipe = StableDiffusionPipeline.from_pretrained("./weights/stable-diffusion-2-1")
        
    def test_model_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)

    @torch.no_grad()
    def test_model_forward_pass(self):
        latent_shape = (1, 4, 64, 64)
        prompt = "A beautiful car"
        cond_tokens = torch.tensor(self.tokenizer([prompt], padding='max_length', max_length=77, truncation=True).input_ids, dtype=torch.long, device="cpu")
        
        # Encode Text
        self.original_pipe.text_encoder.to("mps")
        original_prompt_embed = self.original_pipe.text_encoder(cond_tokens.to("mps"))[0]
        self.original_pipe.text_encoder.to("cpu")
        
        self.text_encoder.to("mps")
        prompt_embed = self.text_encoder(cond_tokens.to("mps"))
        self.text_encoder.to("cpu")
        
        assert prompt_embed.shape == original_prompt_embed.shape
        assert torch.nn.functional.mse_loss(prompt_embed, original_prompt_embed).item() == 0.0

        latent_features = torch.randn(latent_shape, dtype=torch.float32, device="mps")
        
        max_timestep = self.sampler.timesteps[0].to("mps")
        max_timestep = max_timestep.unsqueeze(0)
        
        alpha_T, sigma_T = 0.0047 ** 0.5, (1 - 0.0047) ** 0.5
        self.unet.to("mps")
        pred_noise = self.unet(x=latent_features.to("mps"), timestep=max_timestep.to("mps"), cond=prompt_embed.to("mps"))
        pred_x0 = (latent_features - pred_noise * sigma_T) / alpha_T
        self.unet.to("cpu")
        
        self.original_pipe.unet.to("mps")
        original_pred_noise = self.original_pipe.unet(latent_features.to("mps"), max_timestep.to("mps"), original_prompt_embed.to("mps")).sample
        original_pred_x0 = (latent_features - original_pred_noise * sigma_T) / alpha_T
        self.original_pipe.unet.to("cpu")
        
        assert pred_noise.shape == original_pred_noise.shape
        assert pred_x0.shape == original_pred_x0.shape
        assert torch.nn.functional.mse_loss(pred_noise, original_pred_noise).item() == 0.0
        assert torch.nn.functional.mse_loss(pred_x0, original_pred_x0).item() == 0.0
        
        self.vae.to("mps")
        generated_image = self.vae.decode(pred_x0)
        self.vae.to("cpu")
        
        self.original_pipe.vae.to("mps")
        original_generated_image = self.original_pipe.vae.decode(original_pred_x0 / 0.18215).sample
        self.original_pipe.vae.to("cpu")
        
        assert generated_image.shape == original_generated_image.shape
        assert torch.nn.functional.mse_loss(generated_image, original_generated_image).item() == 0.0
        
        save_image((generated_image + 1) / 2, "generated_image.png")
        save_image((original_generated_image + 1) / 2, "original_generated_image.png")
        
if __name__ == "__main__":
    unittest.main()