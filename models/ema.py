class EMA:
    def __init__(self, beta, start_ema: int=2000):
        self.beta = beta
        self.step = 0
        self.start_ema = start_ema

    def update_ema(self, ema_model, original_model):
        for current_param, ema_param in zip(original_model.parameters(), ema_model.parameters()):
            old, new = ema_param.data, current_param.data
            ema_param.data = old * self.beta + (1 - self.beta) * new

    def reset_parameters(self, ema_model, original_model):
        ema_model = torch.load_state_dict(original_model.state_dict())
        
    def step(self, ema_model, original_model):
        if self.step < self.start_ema:
            self.reset_parameters(ema_model, original_model)
            self.step += 1
        else:
            self.update_ema(ema_model, original_model)
            self.step += 1
            