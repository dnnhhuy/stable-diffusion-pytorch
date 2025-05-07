import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
import unittest

from models.clip import OpenCLIP
from models.vae import VAE
from models.unet import UNet

class TestTextEncoder(unittest.TestCase):
    def setUp(self):
        tokenizer = AutoTokenizer.from_pretrained("./weights/stable-diffusion-2-1/tokenizer")
        self.original_model = CLIPTextModel.from_pretrained("./weights/stable-diffusion-2-1/text_encoder")
        self.original_model.eval()
        self.implemented_model = OpenCLIP.from_pretrained("./weights/stable-diffusion-2-1/text_encoder")
        self.implemented_model.eval()

        self.input = tokenizer(["a diagram", "a dog", "a cat"], return_tensors="pt")
    
    def test_model_initialization(self):
        self.assertIsNotNone(self.implemented_model)
        
    def test_number_of_paramters(self):
        total_params1 = sum(p.numel() for p in self.original_model.parameters())
        total_params2 = sum(p.numel() for p in self.implemented_model.parameters())
        assert total_params1 == total_params2
        
    def test_text_encoder(self):
        with torch.no_grad(), torch.autocast("cpu"):
            text_features1 = self.original_model(**self.input).last_hidden_state
            text_features2 = self.implemented_model.encode_text(self.input.input_ids)
            
            assert text_features1.shape == text_features2.shape
            assert torch.nn.functional.mse_loss(text_features2, text_features1).item() == 0.0
        
class TestVAE(unittest.TestCase):
    def setUp(self):
        self.original_vae = AutoencoderKL.from_pretrained("./weights/stable-diffusion-2-1/vae")
        self.implemented_vae = VAE.from_pretrained("./weights/stable-diffusion-2-1/vae", device='cpu')
        self.generator1 = torch.Generator(device='mps').manual_seed(0)
        self.generator2 = torch.Generator(device='mps').manual_seed(0)
        
    def test_model_initialization(self):
        self.assertIsNotNone(self.implemented_vae)
        
    def test_number_of_paramters(self):
        total_params1 = sum(p.numel() for p in self.original_vae.parameters())
        total_params2 = sum(p.numel() for p in self.implemented_vae.parameters())
        assert total_params1 == total_params2
    
    def test_encoder(self):
        input = torch.rand(size=(1, 3, 512, 512), device='cpu')
        with torch.no_grad():
            self.original_vae.to(device="mps")
            features1 = self.original_vae.encode(input.to("mps")).latent_dist
            mean1, std1 = features1.mean, features1.std
            self.original_vae.to(device="cpu")
            self.implemented_vae.to(device="mps")
            out, mean2, std2 = self.implemented_vae.encode(input.to("mps"))
            self.implemented_vae.to("cpu")
            assert mean1.shape == mean2.shape
            assert std1.shape == std2.shape
            assert torch.nn.functional.mse_loss(mean1, mean2).item() == 0.0
            assert torch.nn.functional.mse_loss(std1, std2).item() == 0.0
            
    def test_vae(self):
        input = torch.rand(size=(1, 3, 512, 512), device='cpu')
        with torch.no_grad():
            self.original_vae.to(device="mps")
            posterior = self.original_vae.encode(input.to("mps")).latent_dist
            z = posterior.mode()
            z = z * 0.18215
            z = z / 0.18215
            features1 = self.original_vae.decode(z).sample
            self.original_vae.to(device="cpu")
            
            self.implemented_vae.to(device="mps")
            z2, mean, std = self.implemented_vae.encode(input.to("mps"), generator=self.generator2)
            features2 = self.implemented_vae.decode(mean * 0.18215)
            self.implemented_vae.to("cpu")
            assert features1.shape == features2.shape
            assert torch.nn.functional.mse_loss(features1, features2).item() == 0.0

class TestUNet(unittest.TestCase):
    def setUp(self):
        self.originalUNet = UNet2DConditionModel.from_pretrained("./weights/stable-diffusion-2-1/unet")
        self.implementedUNet = UNet.from_pretrained("./weights/stable-diffusion-2-1/unet", device='cpu')
        self.originalUNet.eval()
        self.implementedUNet.eval()
        
    def test_model_initialization(self):
        self.assertIsNotNone(self.implementedUNet)
    
    def test_numberOfParamenters(self):
        total_params1 = sum(p.numel() for p in self.originalUNet.parameters())
        total_params2 = sum(p.numel() for p in self.implementedUNet.parameters())
        assert total_params1 == total_params2
    
    def test_inference(self):
        generator = torch.Generator(device='cpu').manual_seed(0)
        z = torch.rand(size=(1, 4, 64, 64), device='cpu', generator=generator)
        timestep = torch.randint(low=0, high=1000, size=(1,), device='cpu', generator=generator)
        cond = torch.rand(size=(1, 77, 1024), device='cpu', generator=generator)
        with torch.no_grad():
            self.originalUNet.to(device="mps")
            features1 = self.originalUNet(sample=z.to("mps"), timestep=timestep.to("mps"), encoder_hidden_states=cond.to("mps")).sample
            self.originalUNet.to(device="cpu")
            self.implementedUNet.to(device="mps")
            features2 = self.implementedUNet(z.to("mps"), timestep=timestep.to("mps"), cond=cond.to("mps"))
            self.implementedUNet.to("cpu")
            assert features1.shape == features2.shape
            assert torch.nn.functional.mse_loss(features2, features1).item() == 0.0
            
if __name__ == '__main__':
    unittest.main()