import torch
import torchvision
from torchvision import transforms
from .model import UNet
from .diffusion import p_xt, n_steps
from .dataset import CIFAR10Dataset

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(n_channels=32).to(device)
    model.load_state_dict(torch.load("unit.pt"))
    
    trans = transforms.ToTensor()
    cifar10 = CIFAR10Dataset().cifar10
    
    j = 0
    for k in range(15):
        x = torch.randn(200, 3, 32, 32).to(device)
        print(j)
        for i in range(n_steps):
            t = torch.tensor(n_steps-i-1, dtype=torch.long).to(device)
            with torch.no_grad():
                pred_noise = model(x.float(), t.unsqueeze(0))
                x = p_xt(x, pred_noise, t.unsqueeze(0), device)
        for x0 in x:
            torchvision.utils.save_image(x0.unsqueeze(0).cpu(), "fake/" + str(j) + ".png")
            image = cifar10['train'][j]['img']
            torchvision.utils.save_image(trans(image), "real/" + str(j) + ".png")
            j+=1

if __name__ == "__main__":
    generate()
