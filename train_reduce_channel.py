import tensorflow as tf
import matplotlib.pyplot as plt
from model_reduce_channel import resnet_model_keras
from tensorflow.keras import backend
from tensorflow.keras import layers, models, optimizers, losses, metrics, regularizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from input import load_data
from common import JSON_TRAIN
from common import TRAIN_IMAGES_DIR
from common import JSON_VAL
from common import VAL_IMAGES_DIR
from common import NUM_CLASS
from common import LOG_DIR
'''
from common import NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
from common import NUM_EPOCHS
from common import BATCH_SIZE

from common import STEPS_PER_EPOCH_FOR_TRAIN
from common import STEPS_PER_EPOCH_FOR_EVAL
'''
from common import L2_WEIGHT_DECAY
from common import MODEL_INPUT_SHAPE
import math
import os

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 53879

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 7120
#一般取值为64，128，256，512，1024
# BATCH_SIZE = 1024
BATCH_SIZE = 128
# NUM_EPOCHS = 2000
NUM_EPOCHS = 500
STEPS_PER_EPOCH_FOR_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
STEPS_PER_EPOCH_FOR_EVAL = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / BATCH_SIZE
MAX_STEPS = NUM_EPOCHS * (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE)

# NUM_EPOCHS_1 = 50
# NUM_EPOCHS_2 = 100
NUM_EPOCHS_1 = 500
NUM_EPOCHS_2 = 100


def resnet50_model(nb_classes=80):
    # 创建模型，定义输入输
    model = resnet_model_keras(nb_classes)
    '''
    model.load_weights(filepath="weights\\trained_weights\\trained_weights_final.h5",
                       by_name=True)
    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    # for layer in base_model.layers[ : -4]:
    for layer in model.layers:
        layer.trainable = False
        # print(layer.name)
    '''
    return model


def load_input():
    # load data

    next_feature, next_label = load_data(JSON_TRAIN, TRAIN_IMAGES_DIR,
                                         is_training=True, batch_size=BATCH_SIZE)

    x_test, y_test = load_data(JSON_VAL, VAL_IMAGES_DIR,
                               is_training=False, batch_size=BATCH_SIZE)

    return next_feature, next_label, x_test, y_test


def do_train(nb_classes, log_dir):
    # 结束当前的TF计算图，并新建一个。有效的避免模型/层的混乱
    backend.clear_session()
    tf.reset_default_graph()
    with tf.device("/gpu:1"):

        next_feature, next_label, x_test, y_test = load_input()

        model = resnet50_model(nb_classes)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', verbose=1, save_weights_only=True,
                                 save_best_only=True, period=50)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)
    # math.ceil向上取整，一个epoch需要将数据集整个遍历一遍
    steps_per_epoch = math.ceil(STEPS_PER_EPOCH_FOR_TRAIN)
    val_steps = math.ceil(STEPS_PER_EPOCH_FOR_EVAL)

    # steps_per_epoch = math.ceil(500/BATCH_SIZE)
    # val_steps= math.ceil(500/BATCH_SIZE)

    # validation_steps = int(320/BATCH_SIZE)
    print("steps_per_epoch:", steps_per_epoch)

    def top3_accuracy(y, predic):
        return metrics.top_k_categorical_accuracy(y, predic, k=3)

    # top3 = metrics.top_k_categorical_accuracy(labels, predic, k=3)
    # adm = optimizers.Adam()
    # loss = losses.categorical_crossentropy()
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    TRAIN_STAGE_1_ENABLE = True
    if TRAIN_STAGE_1_ENABLE:
        print("train stage1: -----------------------------------------------------------------")
        # 可观察模型结构
        model.summary()
        model.compile(optimizer=optimizers.Adam(lr=1e-1),
                      loss="categorical_crossentropy",
                      metrics=[metrics.categorical_accuracy, top3_accuracy])

        hist1 = model.fit(next_feature, next_label, batch_size=None, epochs=NUM_EPOCHS_1,
                          validation_data=(x_test, y_test),
                          verbose=1,
                          callbacks=[checkpoint, reduce_lr, early_stopping],
                          initial_epoch=0,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=val_steps)
        # duration_time = time.time() - start_time

        plt.plot(hist1.history["loss"])
        plt.plot(hist1.history["val_loss"])
        plt.title("model loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + "/fig/model_loss_stage1.png")
        # plt.show()

        # """
        plt.plot(hist1.history["categorical_accuracy"])
        plt.plot(hist1.history["val_categorical_accuracy"])
        plt.title("model accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + "/fig/model_top1_stage1.png")
        # plt.show()

        plt.plot(hist1.history["top3_accuracy"])
        plt.plot(hist1.history["val_top3_accuracy"])
        plt.title("model accuracy3")
        plt.ylabel('accuracy3')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + "/fig/model_top3_stage1.png")
        # plt.show()

        model.save_weights(log_dir + '/trained_weights_stage1.h5')



    # 评估模型
    print("evaluate:")
    loss, accuracy, top3 = model.evaluate(x_test, y_test, verbose=1, steps=val_steps)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    print('Test accuracy of top3:', top3)

    return


# 使用第一张GPU
GPU_IDX = '1'

log_dir = LOG_DIR + "/resnet50_reduce_channel/"
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)

tf.gfile.MakeDirs(log_dir)
tf.gfile.MakeDirs(log_dir + 'fig/')

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDX
do_train(NUM_CLASS, log_dir)
backend.clear_session()