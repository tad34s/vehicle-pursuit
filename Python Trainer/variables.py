
MAX_TRAINED_EPOCHS = 500
VIS_INPUT_SHAPE = (1, 64, 64)

# Learning hyperparameters
START_TEMPERATURE = 20
REDUCE_TEMPERATURE = 1 / 15 # 0.75
NUM_TRAINING_EXAMPLES = 100
DISCOUNT = 0.95 # devalues future reward
LEARNING_RATE = 0.0005


# Reward
REWARD_SAME_ACTION = 2.0 # will be added to the reward for sticking with the same action
