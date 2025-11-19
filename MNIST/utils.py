import numpy as np
import torch
from tqdm import tqdm
from scipy import linalg

def get_activation(dataloader, 
                   model, 
                   preprocess,
                   device = 'cpu'
                  ):
    
    model.to(device)
    model.eval()
    
    pred_arr = np.zeros((len(dataloader.dataset), 2048))
    
    batch_size = dataloader.batch_size
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            
            batch = torch.stack([preprocess(img) for img in batch]).to(device)
            
            pred = model(batch).cpu().numpy()
            
            pred_arr[i*batch_size : i*batch_size + batch.size(0), :] = pred
            
    return pred_arr

def calculate_activation_statistics(dataloader, 
                                    model, 
                                    preprocess, 
                                    device='cpu'
                                   ):
    
    act = get_activation(dataloader, 
                         model, 
                         preprocess,
                         device
                       )
    mu = np.mean(act, axis=0)
    
    sigma = np.cov(act, rowvar=False)
    
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
