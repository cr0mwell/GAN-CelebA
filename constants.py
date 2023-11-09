import os

# Dataset
SAMPLES_DIR = './src/samples'
MODELS_DIR = './src/models'
DATASET_DIR = '../../CV/datasets'
DATASET = 'CelebA'
SEP = os.path.sep
DS_PATH = f'{DATASET_DIR}{SEP}{DATASET}'
IMG_RES = 64
IMG_PATH = f'{DS_PATH}{SEP}{IMG_RES}x{IMG_RES}'

# Model optimization
EPOCHS = 50
BATCH_SIZE = 100
SAMPLE_INTERVAL = 400
ADAM_MOMENTUM = 0.5
EMA = 0.99
LATENT_DIM = 100
FILTERS = 128
DISC_STEPS = 5
DROPOUT_VAL = 0.3

# Resolution of image for the Kernel Inception Distance metric
KID_IMG_RES = 299

# Image preprocessing
MEAN, VAR = 0, 0.001
SIGMA = VAR ** 0.5

# Adaptive discriminator augmentation
MAX_TRANSLATION = 0.125
MAX_ROTATION = 0.125
MAX_ZOOM = 0.25
TARGET_ACCURACY = 0.85
STEPS = 1000
