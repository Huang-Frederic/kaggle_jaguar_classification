import os

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR      = os.path.join(PROJECT_ROOT, 'data')
CLEANED_DIR   = os.path.join(DATA_DIR, 'cleaned')
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'outputs')
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submissions')
LOG_DIR       = os.path.join(PROJECT_ROOT, 'tensorboards')

IMG_SIZE    = 224
NUM_CLASSES = 31
BATCH_SIZE  = 32
SEED        = 42
