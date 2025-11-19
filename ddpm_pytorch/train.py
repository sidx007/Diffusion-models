import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import CONFIG
from .dataset import CustomMnistDataset
from .model import Unet
from .diffusion import DiffusionForwardProcess

def train(cfg):
    
    mnist_ds = CustomMnistDataset(cfg.train_csv_path)
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    
    model = Unet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()
    
    dfp = DiffusionForwardProcess()
    
    best_eval_loss = float('inf')
    
    for epoch in range(cfg.num_epochs):
        
        losses = []
        
        model.train()
        
        for imgs in tqdm(mnist_dl):
            
            imgs = imgs.to(device)
            
            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, cfg.num_timesteps, (imgs.shape[0],)).to(device)
            
            noisy_imgs = dfp.add_noise(imgs, noise, t)
            
            optimizer.zero_grad()
            
            noise_pred = model(noisy_imgs, t)
            
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        mean_epoch_loss = np.mean(losses)
        
        print('Epoch:{} | Loss : {:.4f}'.format(
            epoch + 1,
            mean_epoch_loss,
        ))
        
        if mean_epoch_loss < best_eval_loss:
            best_eval_loss = mean_epoch_loss
            torch.save(model, cfg.model_path)
            
    print(f'Done training.....')

if __name__ == "__main__":
    cfg = CONFIG()
    train(cfg)
