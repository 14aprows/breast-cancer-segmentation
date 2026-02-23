import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class BUSISegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)

            for file in os.listdir(class_path):
                if file.endswith(".png") and not file.endswith("_mask.png"):
                    img_path = os.path.join(class_path, file)
                    mask_path = os.path.join(
                        class_path,
                        file.replace(".png", "_mask.png")
                    )

                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        mask = mask.float()

        return image, mask