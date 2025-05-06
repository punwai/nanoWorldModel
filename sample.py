from tqdm import tqdm
from model.video_dit import DiffusionForcingDiT
import torch
from torch import nn
import os
from torchvision.utils import save_image
from configs import mnist_config

T = 1000


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    n_steps: int = 1000,
    img_dim: tuple[int, int] = (64, 64),
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
    save_images: bool = False,
):
    betas = torch.linspace(1e-4, 2e-2, n_steps, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

    x = torch.randn(batch_size, 32, 3, *img_dim, device=device)

    labels_1 = torch.arange(batch_size, device=device) % 10
    labels_2 = (torch.arange(batch_size, device=device) // 10) % 10

    for _t in tqdm(reversed(range(n_steps)), desc="Sampling..."):
        t = torch.tensor([_t], device=device).expand(batch_size)

        labels = torch.stack([labels_1, labels_2], dim=-1)
        eps = model(x, t.float() / T, labels)

        # q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \mu_{\theta}(x_t, t), \sigma_{t-1}^2 I)
        # Split the equation into multiple lines for debugging
        betas_t = betas[t][:, None, None, None, None]
        sqrt_one_minus_alphas_bar_t = sqrt_one_minus_alphas_bar[t][:, None, None, None, None]
        alphas_t = alphas[t][:, None, None, None, None]

        # compute mu_theta
        scaled_eps = (betas_t / sqrt_one_minus_alphas_bar_t) * eps
        x_minus_scaled_eps = x - scaled_eps
        mu_theta = x_minus_scaled_eps / alphas_t.sqrt()

        if _t > 0:
            # noise = torch.randn_like(x) * betas[_t].sqrt()
            # x = mu_theta + noise
            x = mu_theta
        else:
            x = mu_theta

    if save_images:
        os.makedirs("./images", exist_ok=True)
        # Save each image in the batch separately
        for i, img in enumerate(x):
            # Clamp to [0,1] for visualization if needed
            # print distribution fo img
            img = (img.clamp(-1, 1) + 1) / 2
            save_image(img, f"./images/sample_{i}.png")

    return x

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpts/model_37.pth", help="Path to checkpoint file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionForcingDiT(mnist_config)

    model.from_checkpoint(args.ckpt)

    model.to(device)
    x = sample_ddpm(model, device=device)
 