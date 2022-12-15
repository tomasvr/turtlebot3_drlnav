# general settings
ENABLE_BACKWARD          = False
ENABLE_STACKING          = False
ENABLE_VISUAL            = False    # Meant to be used only during evaluation/testing phase
ENABLE_TRUE_RANDOM_GOALS = False    # If false, goals are taken randomly from a list of known valid goal positions
MODEL_STORE_INTERVAL = 100          # Store the model weights every N episodes

# DRL parameters
ACTION_SIZE     = 2         # Not used for DQN, see DQN_ACTION_SIZE
HIDDEN_SIZE     = 512       # Number of neurons in hidden layers

BATCH_SIZE      = 128       # Number of samples per training batch
BUFFER_SIZE     = 1000000   # Number of samples stored in replay buffer before FIFO
DISCOUNT_FACTOR = 0.99
LEARNING_RATE   = 0.003
TAU             = 0.003

OBSERVE_STEPS   = 25000     # At training start random actions are taken for N steps for better exploration
STEP_TIME       = 0.01      # Delay between steps, can be set to 0
EPSILON_DECAY   = 0.9995    # Epsilon decay per step
EPSILON_MINIMUM = 0.05

# DQN parameters
DQN_ACTION_SIZE = 5
TARGET_UPDATE_FREQUENCY = 1000

# DDPG parameters

# TD3 parameters
POLICY_NOISE            = 0.2
POLICY_NOISE_CLIP       = 0.5
POLICY_UPDATE_FREQUENCY = 2

# drl_environment
REWARD_FUNCTION = "A"           # Defined in reward.py
EPISODE_TIMEOUT_SECONDS = 50    # Number of seconds after which episode timeout occurs
ENABLE_MOTOR_NOISE = False      # Add normally distributed noise to motor output to simulate hardware imperfections

# Stacking
STACK_DEPTH = 3                 # Number of subsequent frames processed per step
FRAME_SKIP  = 4                 # Number of frames skipped in between subsequent frames