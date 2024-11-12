import copy
import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, beta: float=0.995, start_ema: int=2000):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval().requires_grad(False)
        self.beta = beta
        self.step = 0
        self.start_ema = start_ema

    def update_ema(self, original_model):
        for current_param, ema_param in zip(original_model.parameters(), self.ema_model.parameters()):
            old, new = ema_param.data, current_param.data
            ema_param.data = old * self.beta + (1 - self.beta) * new

    def reset_parameters(self, original_model):
        self.ema_model = torch.load_state_dict(original_model.state_dict())
        
    def step(self, original_model):
        if self.step < self.start_ema:
            self.reset_parameters(original_model)
            self.step += 1
        else:
            self.update_ema(original_model)
            self.step += 1
            