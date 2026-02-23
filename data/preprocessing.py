import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import Config

def get_train_transforms():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        A.Affine(
            scale=(0.95, 1.05),          
            translate_percent=(0.0, 0.05),  
            rotate=(-10, 10),            
            shear=(-5, 5),               
            p=0.5
        ),

        A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),

        A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])