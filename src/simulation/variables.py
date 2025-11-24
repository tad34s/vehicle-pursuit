import numpy as np

MODEL_PATH = "./models"

# Hyperparameters
NUM_TRAINING_EXAMPLES = 7500
MAX_TRAINED_EPISODES = 500
NUM_DATALOADER_WORKERS = 4


ACTION_OPTIONS = [
    np.array([[1, 0, 1, 0]]),
    np.array([[1, 0, 0, 0]]),
    np.array([[1, 0, 0, 1]]),
]
MIRRORED_ACTIONS = [2, 1, 0]

# DepthNet

IMAGE_SIZE = (512, 512)

DATASET_LOCATION = "dataset"
