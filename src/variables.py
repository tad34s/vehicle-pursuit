import numpy as np

MODEL_PATH = "./models"

VISUAL_INPUT_SHAPE = (1, 100, 160)
NONVISUAL_INPUT_SHAPE = (6,)

# Hyperparameters
EPISODE_LENGTH = 500
MAX_TRAINED_EPISODES = 500


ACTION_OPTIONS = [
    np.array([[1, 0, 1, 0]]),
    np.array([[1, 0, 0, 0]]),
    np.array([[1, 0, 0, 1]]),
]
MIRRORED_ACTIONS = [2, 1, 0]
