from model import StableDiffusion
import torch
from tqdm import tqdm


def train(model, train_dataloader, test_dataloader, epochs, device):
    results = {'train_loss': [],
              'test_loss': []}
    
    for epoch in tqdm(range(epochs)):
        train_loss = StableDiffusion.train_step(train_dataloader=train_dataloader, 
                                              device=device,
                                              optimizer=optimizer,
                                              tokenizer=tokenizer)

        test_loss = StableDiffusion.test_step(test_dataloader=test_dataloader,
                                             device=device,
                                             tokenizer=tokenizer)

        print(f"Epoch {epoch+1} | "
        f"Train loss: {train_loss} | "
        f"Test loss: {test_loss}")

        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

    return results


if __name__ == '__main__':
    model = StableDiffusion()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    train(model, epochs=10, device=device)
    