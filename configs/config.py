import torch 
import os 

class Config:
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_ROOT = "/Dataset_BUSI_with_GT"
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    IMAGE_SIZE = 256

    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    T_MAX = 30

    SAVE_DIR = "checkpoints"