REDUCE_TEMPERATURE = 1 / 8
START_TEMPERATURE = 50
LEARNING_RATE_QNET = 0.0005
LEARNING_RATE_DEPTH_NET = 0.0005

# Reward
DISCOUNT = 0.95  # devalues future reward
REWARD_MAX = 10  # aprox. the maximum reward it will get for staying on the line
STEERING_DISCOUNT = 0.9  # multiplier for steering


# Input sizes
VISUAL_INPUT_SHAPE = (3, 128, 128)
NONVISUAL_INPUT_SHAPE = (6,)
