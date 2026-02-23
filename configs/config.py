import torch 
import os 

class Config:
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ))
    DATA_ROOT = os.path.join(DATA, 'Dataset_BUSI_with_GT')

    BATCH_SIZE = 8
    NUM_WORKERS = 4
    IMAGE_SIZE = 256

    IN_CHANNELS = 3
    OUT_CHANNELS = 1

    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    T_MAX = 30

    SAVE_DIR = "checkpoints"