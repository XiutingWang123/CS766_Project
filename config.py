import os

"""
Path and parameters
"""

DATA_PATH = 'data'

PASCAL_PATH_TEST = os.path.join(DATA_PATH, 'test')

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

#WEIGHTS_FILE = None
#WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

#FLIPPED = True


"""
 Network parameter
"""


IMAGE_SIZE = 448

GRID_SIZE = 7

BOXES_PER_GRID = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1.0
COORD_SCALE = 5.0


"""
 Solver parameter
"""

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 64

MAX_ITER = 15000

SUMMARY_ITER = 100

SAVE_ITER = 1000


"""
 test parameter
"""

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5

IOU_OVERLAP_THRESHOLD = 0.5