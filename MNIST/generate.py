import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from .config import CONFIG
from .diffusion import DiffusionReverseProcess

def generate(cfg):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    drp = DiffusionReverseProcess()
    
    model = torch.load(cfg.model_path).to(device)
    model.eval()
    
    xt = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)
    
    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            xt, x0 = drp.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))

    xt = torch.clamp(xt, -1., 1.).detach().cpu()
    xt = (xt + 1) / 2
    
    return xt

if __name__ == "__main__":
    cfg = CONFIG()

    generated_imgs = []
    for i in tqdm(range(cfg.num_img_to_generate)):
        xt = generate(cfg)
        xt = 255 * xt[0][0].numpy()
        generated_imgs.append(xt.astype(np.uint8).flatten())

    generated_df = pd.DataFrame(generated_imgs, columns=[f'pixel{i}' for i in range(784)])
    generated_df.to_csv(cfg.generated_csv_path, index=False)

    fig, axes = plt.subplots(8, 8, figsize=(5, 5))

    for i, ax in enumerate(axes.flat):
        ax.imshow(np.reshape(generated_imgs[i], (28, 28)), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
