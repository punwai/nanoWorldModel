import argparse
from data.convert_dataset import BouncingMNISTDataset
from torch.utils.data import DataLoader
from model.video_dit import DiffusionForcingDiT
from sample import sample_ddpm
import torch
from torch import nn
import wandb
from configs import mnist_config
import time
import os
from torch.nn.utils import clip_grad_norm_
import math
import torch.distributed as dist

# 
# 0. set up distributed training
# 
def setup_dist():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_dist():
    dist.destroy_process_group()

# 
# 1. Set up everything
# 

args = argparse.ArgumentParser()
args.add_argument("--run_name", type=str, default="")

init_lr = 1e-4

per_device_batch_size = 8 

train_epochs = 50
T = 1000
device = "cuda"
wandb_token = "7a6d6808178f08a806911ec7263c24a59f6df7da"
model_save_steps = 200
train_log_step = 10
model_save_path = "ckpts"
sample_epochs = 30
run_name = "bouncing_mnist_video"


setup_dist()
rank = dist.get_rank()
device = torch.device("cuda", rank)

if rank == 0:
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    wandb.login(key=wandb_token)
    wandb.init(
        project="ddim", 
        config={
            "init_lr": init_lr,
            "per_device_batch_size": per_device_batch_size,
            "train_epochs": train_epochs,
            "T": T
        },
        name=run_name
    )

# 
# Diffusion parameters
# 
betas = torch.linspace(1e-4, 2e-2, T, device=device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
sqrt_alphas_bar = torch.sqrt(alphas_bar).to(device)
sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar).to(device)

# 
# Model
# 


model = DiffusionForcingDiT(mnist_config)
model.to(device)
torch.compile(model)
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[rank], output_device=rank
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9,0.999),
)

# 
# Data load
# 

start_time = time.time()

dataset = BouncingMNISTDataset("data/bouncing_mnist")

consistent_data = [(item["video"], item["label"]) for item in dataset if item["video"].shape == (32, 3, 64, 64)]
dataset.videos, dataset.labels = zip(*consistent_data) if consistent_data else ([], [])

train_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=True
)

def collate_fn(batch):
    vids = torch.stack([item["video"] for item in batch], dim=0)
    labs = torch.stack([item["label"] for item in batch], dim=0)
    return {"video": vids, "label": labs}


train_loader = DataLoader(
    dataset,
    collate_fn=collate_fn,
    batch_size=per_device_batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

overall_steps = train_epochs * len(train_loader)
total_steps = 0

for epoch in range(train_epochs):
    from tqdm import tqdm

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}", unit="batch")):
        total_steps += 1

        x = batch["video"].to(device)
        labels = batch["label"].to(device)

        b = x.shape[0]
        t = torch.randint(0, T, (b,)).to(device)

        noise = torch.randn_like(x)
        noised_x = sqrt_alphas_bar[t][:,None,None,None,None] * x + sqrt_one_minus_alphas_bar[t][:,None,None,None,None] * noise
        t_emb = t.float() / T

        labels = labels

        loss = nn.functional.mse_loss(
            model(noised_x, t_emb, labels), 
            noise
        )

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if rank == 0:
            if total_steps % model_save_steps == 0:
                torch.save(model.state_dict(), f"{model_save_path}/model_{total_steps}.pth")
            if total_steps % train_log_step == 0:
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                wandb.log({
                    "loss": loss.item(),
                    "grad_norm": grad_norm
                })

    if (epoch + 1) % sample_epochs == 0:
        sample_ddpm(model, device=device, save_images=True)



