# 32739

TRAIN_IMAGES_DIR = "D:\datasets\\ai_challenger_scene\scene_train_images_20170904"



# 4982

VAL_IMAGES_DIR = 'D:\datasets\\ai_challenger_scene\scene_validation_images_20170908'




JSON_TRAIN = 'D:\datasets\\ai_challenger_scene\scene_train_annotations_20170904.json'



JSON_VAL = 'D:\datasets\\ai_challenger_scene\scene_validation_annotations_20170908.json'


LOG_DIR = "logs"



NUM_CLASS = 80

# HEIGHT = 32

# WIDTH = 32
WIDTH = 224
HEIGHT = 224

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 53879

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 7120
#一般取值为64，128，256，512，1024
# BATCH_SIZE = 1024
BATCH_SIZE = 32
# NUM_EPOCHS = 2000
NUM_EPOCHS = 1
STEPS_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
MAX_STEPS = NUM_EPOCHS * (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE)


# 正则化惩罚系数
L2_WEIGHT_DECAY = 1e-4
#
BATCH_NORM_DECAY = 0.9

BATCH_NORM_EPSILON = 1e-5

MODEL_INPUT_SHAPE = (224, 224, 3)

