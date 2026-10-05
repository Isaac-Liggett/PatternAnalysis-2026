from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.paths = [
            p for p in Path(root_dir).rglob("*")
            if p.suffix.lower() == ".png"
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # scales to [0, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img)


def get_dataloader(root_dir, image_size=64, batch_size=64, num_workers=0, shuffle=True):
    dataset = ImageFolderDataset(root_dir, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # avoids a ragged final batch messing with BatchNorm stats
    )