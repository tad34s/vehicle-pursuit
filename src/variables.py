import numpy as np

MODEL_PATH = "./models"

# Hyperparameters
EPISODE_LENGTH = 5000
MAX_TRAINED_EPISODES = 500
NUM_DATALOADER_WORKERS = 4


ACTION_OPTIONS = [
    np.array([[1, 0, 1, 0]]),
    np.array([[1, 0, 0, 0]]),
    np.array([[1, 0, 0, 1]]),
]
MIRRORED_ACTIONS = [2, 1, 0]
