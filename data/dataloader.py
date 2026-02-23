from torch.utils.data import DataLoader, random_split, Subset
from configs.config import Config
from data.dataset import BUSISegmentationDataset
from data.preprocessing import get_train_transforms, get_val_transforms

def get_dataloaders(data_root, batch_size=8, num_workers=3):
    dataset = BUSISegmentationDataset(
        Config.DATA_ROOT,
        transform=None
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_dataset = BUSISegmentationDataset(
        data_root,
        transform=get_train_transforms()
    )

    val_dataset = BUSISegmentationDataset(
        data_root,
        transform=get_val_transforms()
    )

    train_dataset = Subset(train_dataset, train_subset.indices)
    val_dataset = Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader