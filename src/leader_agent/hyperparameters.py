# Training
REDUCE_TEMPERATURE = 1 / 25
START_TEMPERATURE = 15
LEARNING_RATE = 0.0005

# Reward
DISCOUNT = 0.95  # devalues future reward
REWARD_MAX = 5  # aprox. the maximum reward it will get for staying on the line
STEERING_DISCOUNT = 0.8  # multiplier for steering

# Input sizes
VISUAL_INPUT_SHAPE = (1, 100, 160)
NONVISUAL_INPUT_SHAPE = (1,)

# Image preprocessing
BLUR_INTENSITY = 1.2
NOISE_OPACITY = 0.004
NOISE_INTESITY = 8
