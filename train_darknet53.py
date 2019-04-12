import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from darknet53_model import darknet53_model_keras
from keras import optimizers, losses, metrics, regularizers
# from tensorflow.keras import optimizers, losses, metrics, regularizers
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from input import load_data
from common import JSON_TRAIN
from common import TRAIN_IMAGES_DIR
from common import JSON_VAL
from common import VAL_IMAGES_DIR
from common import NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
from common import NUM_EPOCHS
from common import BATCH_SIZE
from common import NUM_CLASS
from common import LOG_DIR
from common import STEPS_PER_EPOCH_FOR_TRAIN
from common import STEPS_PER_EPOCH_FOR_EVAL

from common import L2_WEIGHT_DECAY
from common import MODEL_INPUT_SHAPE
import math
import os

# NUM_EPOCHS_1 = 50
# NUM_EPOCHS_2 = 100
NUM_EPOCHS_1 = 5
NUM_EPOCHS_2 = 10

def darknet53_model(nb_classes=80):
    # 创建模型，定义输入输
    '''create the training model'''
    K.clear_session()  # get a new session
    model = darknet53_model_keras(nb_classes)
    # 可观察模型结构
    model.summary()
    model.load_weights(filepath="weights/darknet53_weights.h5",
                       by_name=True)
    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    # for layer in base_model.layers[ : -4]:
    for layer in model.layers[:-2]:
        layer.trainable = False
        # print(layer.name)

    return model

def do_train(nb_classes, log_dir):
    # create model
    model = darknet53_model(nb_classes)
    # 可观察模型结构
    model.summary()

    # load data
    next_feature, next_label = load_data(JSON_TRAIN, TRAIN_IMAGES_DIR,
                                         is_training=True, batch_size=BATCH_SIZE)

    x_test, y_test = load_data(JSON_VAL, VAL_IMAGES_DIR,
                               is_training=False, batch_size=BATCH_SIZE)
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5',
                                 monitor='val_loss', verbose=1, save_weights_only=True,
                                 save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    # math.ceil向上取整，一个epoch需要将数据集整个遍历一遍
    steps_per_epoch = math.ceil(STEPS_PER_EPOCH_FOR_TRAIN)
    val_steps = math.ceil(STEPS_PER_EPOCH_FOR_EVAL)

    steps_per_epoch = math.ceil(500/BATCH_SIZE)
    val_steps= math.ceil(500/BATCH_SIZE)

    # validation_steps = int(320/BATCH_SIZE)
    print("steps_per_epoch:", steps_per_epoch)


    def top3_accuracy(y, predic):
        return metrics.top_k_categorical_accuracy(y, predic, k=3)
    # top3 = metrics.top_k_categorical_accuracy(labels, predic, k=3)
    adm = optimizers.Adam()
    # loss = losses.categorical_crossentropy()
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=optimizers.Adam(lr=1e-3),
                      loss="categorical_crossentropy",
                      metrics=[metrics.categorical_accuracy, top3_accuracy])

        hist1 = model.fit(next_feature, next_label, batch_size=None, epochs=NUM_EPOCHS_1,
                          validation_data=(x_test, y_test),
                          verbose=1,
                          callbacks=[checkpoint],
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
        plt.show()

        # """
        plt.plot(hist1.history["categorical_accuracy"])
        plt.plot(hist1.history["val_categorical_accuracy"])
        plt.title("model accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + "/fig/model_top1_stage1.png")
        plt.show()

        plt.plot(hist1.history["top3_accuracy"])
        plt.plot(hist1.history["val_top3_accuracy"])
        plt.title("model accuracy3")
        plt.ylabel('accuracy3')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + "/fig/model_top3_stage1.png")
        plt.show()

        model.save_weights(log_dir + 'trained_weights_stage1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        # 可观察模型结构
        model.summary()
        model.compile(optimizer=optimizers.Adam(lr=1e-4),
                      loss="categorical_crossentropy",
                      metrics=[metrics.categorical_accuracy, top3_accuracy])

        hist2 = model.fit(next_feature, next_label, batch_size=None, epochs=NUM_EPOCHS_2,
                          validation_data=(x_test, y_test),
                          verbose=1,
                          callbacks=[checkpoint, reduce_lr, early_stopping],
                          initial_epoch=NUM_EPOCHS_1,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=val_steps)
        # duration_time = time.time() - start_time

        plt.plot(hist2.history["loss"])
        plt.plot(hist2.history["val_loss"])
        plt.title("model loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + "/fig/model_loss_stage2.png")
        plt.show()

        # """
        plt.plot(hist2.history["categorical_accuracy"])
        plt.plot(hist2.history["val_categorical_accuracy"])
        plt.title("model accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + "/fig/model_top1_stage2.png")
        plt.show()

        plt.plot(hist2.history["top3_accuracy"])
        plt.plot(hist2.history["val_top3_accuracy"])
        plt.title("model accuracy3")
        plt.ylabel('accuracy3')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(log_dir + "/fig/model_top3_stage2.png")
        plt.show()

        model.save_weights(log_dir + 'trained_weights_final.h5')

    # 评估模型
    print("evaluate:")
    loss, accuracy , top3 = model.evaluate(x_test, y_test, verbose=1, steps=val_steps)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    print('Test accuracy of top3:', top3)

    return

# 使用第一张GPU
GPU_IDX = '0'

log_dir = LOG_DIR + "/darknet53_" + GPU_IDX + "/"
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)

tf.gfile.MakeDirs(log_dir)
tf.gfile.MakeDirs(log_dir + 'fig/')


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDX
do_train(NUM_CLASS, log_dir)