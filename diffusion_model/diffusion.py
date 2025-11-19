import torch

n_steps = 1000
beta = torch.linspace(0.0001, 0.04, n_steps)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def q_xt_xtminus1(xtm1, t, device):
    c, h, w = xtm1.shape
    xtm1 = xtm1.to(device)

    eta = torch.randn(c, h, w).to(device)

    noisy = (1 - beta[t]).sqrt().reshape(1, 1, 1).to(device) * xtm1 + beta[t].sqrt().reshape(1, 1, 1).to(device) * eta
    return noisy

def q_xt_x0(x0, t, device):
    n, c, h, w = x0.shape
    x0 = x0.to(device)
    eps = torch.randn(n, c, h, w).to(device)

    a_bar = alpha_bar[t].to(device)

    noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eps
    return noisy, eps

def p_xt(xt, noise, t, device):
    alpha_t = alpha[t].to(device)
    alpha_bar_t = alpha_bar[t].to(device)
    eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** .5
    mean = 1 / (alpha_t ** 0.5) * (xt - eps_coef * noise)
    var = beta[t].to(device)
    eps = torch.randn(xt.shape, device=device)
    return mean + (var ** 0.5) * eps
