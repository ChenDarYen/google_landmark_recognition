import os
import multiprocessing

# dir
TRAIN_SCV = '../input/landmark-recognition-2020/train.csv'
TEST_CSV = '../input/landmark-recognition-2020/sample_submission.csv'
TRAIN_DIR = '../input/landmark-recognition-2020/train/'
TEST_DIR = '../input/landmark-recognition-2020/test/'

# general
IS_TRAIN = True
RECORD_PATH = ''
CPU_NUM = multiprocessing.cpu_count()
LOG_STEPS = 10
MIN_SAMPLES_PER_CLASS = 143

# training
EPOCHS = 5
BATCH_SIZE = 64
INPUT_SIZE = 64

# model
COMPOUND_COEF = 1
DROPOUT_RATE = 0.25

# optimizer
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
LR = 0.256
STEP_SIZE = 2
GAMMA = 0.98

# inference
NUM_PRED = 20



