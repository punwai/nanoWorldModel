# 
# given a directory containing bouncing_mnist videos, where each
# gif is labeled `bouncing_<idx>.gif`, and a labels.csv file, where each
# line is of the form `bouncing_<idx>.gif,<digit_1>,<digit_2>`, convert
# the dataset into a torch dataset.
# 
import argparse
import itertools
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageSequence

class BouncingMNISTDataset(Dataset):

    def __init__(self, data_dir, dataset_size=None):
        self.data_dir = data_dir
        self.dataset_size = dataset_size
        labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        # Preload all gifs and labels into memory
        self.videos = []
        self.labels = []
        from tqdm import tqdm

        labels_iter = labels_df.iterrows()
        total_size = len(labels_df)
        if dataset_size is not None:
            labels_iter = itertools.islice(labels_iter, min(dataset_size, total_size))
            total_size = dataset_size

        for _, row in tqdm(labels_iter, total=total_size, desc="Loading videos"):
            gif_path = os.path.join(data_dir, row["filename"])
            with Image.open(gif_path) as img:
                frames = []
                # Iterate through all frames of the gif
                for frame in ImageSequence.Iterator(img):
                    arr = np.array(frame.convert("L"))
                    frames.append(torch.from_numpy(arr))
                video_tensor = torch.stack(frames).float() / 255.0 * 2 - 1  # shape: (T, V, H, W)
            video_tensor = video_tensor[:,None,:,:].repeat(1, 3, 1, 1)  # shape: (T, 3, H, W)
            self.videos.append(video_tensor)
            self.labels.append(torch.tensor(
                [row["digit_1"], row["digit_2"]], dtype=torch.long
            ))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return {
            "video": self.videos[idx],
            "label": self.labels[idx]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    dataset = BouncingMNISTDataset(args.data_dir)
    print(len(dataset))
    print(dataset[0]["video"].shape)
    print(dataset[0]["label"])
