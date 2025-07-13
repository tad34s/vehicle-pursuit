REDUCE_TEMPERATURE = 1 / 25
START_TEMPERATURE = 20
LEARNING_RATE_QNET = 0.0005
LEARNING_RATE_DEPTH_NET = 0.0005

# Reward
DISCOUNT = 0.95  # devalues future reward
REWARD_MAX = 5  # aprox. the maximum reward it will get for staying on the line
STEERING_DISCOUNT = 0.8  # multiplier for steering


# Input sizes
VISUAL_INPUT_SHAPE = (1, 100, 160)
NONVISUAL_INPUT_SHAPE = (6,)
