import torch
import torch.nn.functional as F
from tqdm import tqdm
from .dataset import CIFAR10Dataset, getTransforms
from .model import UNet
from .diffusion import q_xt_x0, n_steps

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unet = UNet(n_channels=32).to(device)
    
    batch_size = 128
    lr = 7e-5
    
    my_transforms = getTransforms()
    dataset = CIFAR10Dataset(my_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(unet.parameters(), lr=lr)
    
    epochs = 30
    epoch_loss = []
    losses = []
    
    for epoch in tqdm(range(1, epochs + 1)):
        average_train_loss = 0
        loop_train = tqdm(enumerate(data_loader, 1), total=len(data_loader), desc="Train", position=0, leave=True)
        for index, x0 in loop_train:
            optim.zero_grad()
            t = torch.randint(0, n_steps, (x0.shape[0],), dtype=torch.long).to(device)
    
            xt, noise = q_xt_x0(x0, t, device)
    
            pred_noise = unet(xt.float(), t)
            loss = F.mse_loss(noise.float(), pred_noise)
            losses.append(loss.item())
            average_train_loss+=loss.item()
    
            loss.backward()
            optim.step()
    
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                avg_train_loss="{:.4f}".format(average_train_loss / index),
                refresh=True,
            )
        epoch_loss.append(average_train_loss / len(data_loader))
    
    torch.save(unet.state_dict(), "unit.pt")

if __name__ == "__main__":
    train()
