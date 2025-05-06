import argparse
import csv, json, os
import numpy as np
from datasets import load_dataset
from PIL import Image
from multiprocessing import Pool, Manager

def generate_bouncing_mnist(images, *,     # list[PIL.Image]  (len ≥1)
                            num_frames=100,
                            canvas_size=64,
                            velocity_range=(2, 4),
                            seed=None):
    """
    Return np.uint8 array → (T, H, W) in [0,255]  (grayscale).
    Earlier images in the list are drawn last → always on top.
    """
    rng = np.random.default_rng(seed)
    # Normalize digits to boolean masks & store dims
    masks   = [np.array(im.resize((28, 28))).astype(bool)     for im in images]
    heights = [m.shape[0] for m in masks]
    widths  = [m.shape[1] for m in masks]

    # Random initial (x,y)  (top‑left) and (vx,vy)
    xs = rng.integers(0, canvas_size - np.array(widths))
    ys = rng.integers(0, canvas_size - np.array(heights))
    vxs = rng.choice([-1, 1], size=len(images)) * rng.integers(*velocity_range, size=len(images))
    vys = rng.choice([-1, 1], size=len(images)) * rng.integers(*velocity_range, size=len(images))

    frames = np.zeros((num_frames, canvas_size, canvas_size), dtype=np.uint8)

    for t in range(num_frames):
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        # earlier digits last ⇒ on top
        for idx in reversed(range(len(images))):
            x, y, m = xs[idx], ys[idx], masks[idx]
            h, w    = m.shape
            canvas[y:y+h, x:x+w][m] = 255  # copy white pixels

        frames[t] = canvas

        # update positions & bounce
        xs += vxs
        ys += vys
        for i in range(len(images)):
            if xs[i] < 0 or xs[i] + widths[i]  > canvas_size:
                vxs[i] *= -1
                xs[i]   = np.clip(xs[i], 0, canvas_size - widths[i])
            if ys[i] < 0 or ys[i] + heights[i] > canvas_size:
                vys[i] *= -1
                ys[i]   = np.clip(ys[i], 0, canvas_size - heights[i])

    return frames

from datasets import load_dataset
mnist = load_dataset("ylecun/mnist", split="train")     # quicker mirror than ylecun/*
# shuffle
mnist = mnist.shuffle(seed=42)

# save as GIF

import os
from multiprocessing import Pool
from PIL import Image
import numpy as np

def process_batch(start_idx, end_idx, seed, output_dir, return_list):
    rng = np.random.default_rng(seed + start_idx)
    local_log = []                         # collect (fname, [d1, d2]) pairs
    for idx in range(start_idx, end_idx):
        sel = rng.choice(len(mnist), size=2, replace=False).tolist()  # convert to Python ints
        imgs  = [mnist[j]["image"]  for j in sel]
        labs  = [mnist[j]["label"]  for j in sel]   # ← labels

        vid   = generate_bouncing_mnist(imgs, num_frames=32,
                                        seed=int(seed + idx))
        pil_frames = [Image.fromarray(frame) for frame in vid]
        fname = f"bouncing_{idx}.gif"
        pil_frames[0].save(os.path.join(output_dir, fname),
                           save_all=True, append_images=pil_frames[1:],
                           duration=40, loop=0)
        local_log.append((fname, labs))
    return_list += local_log               # push to shared list

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    manager = Manager()
    label_log = manager.list()             # shared among workers

    # split work
    batch_size, rem = divmod(args.dataset_size, args.num_workers)
    batches, start = [], 0
    for i in range(args.num_workers):
        end = start + batch_size + (1 if i < rem else 0)
        batches.append((start, end, args.seed, args.output_dir, label_log))
        start = end

    with Pool(args.num_workers) as pool:
        pool.starmap(process_batch, batches)

    # write master label file (both CSV and JSONL for convenience)
    csv_path  = os.path.join(args.output_dir, "labels.csv")
    json_path = os.path.join(args.output_dir, "labels.jsonl")
    with open(csv_path,  "w", newline="") as f_csv, \
         open(json_path, "w")             as f_json:
        writer = csv.writer(f_csv)
        writer.writerow(["filename", "digit_1", "digit_2"])
        for fname, labs in label_log:
            writer.writerow([fname, *labs])
            f_json.write(json.dumps({"filename": fname,
                                     "digits": labs}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="bouncing_mnist")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mnist = load_dataset("ylecun/mnist", split="train").shuffle(seed=42)
    main(args)
