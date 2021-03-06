import tensorflow as tf
from tensorflow.python.keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from input import load_data
from model import resnet_model_keras
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
import time
from datetime import datetime
import math

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from common import JSON_TRAIN
from common import TRAIN_IMAGES_DIR
from common import JSON_VAL
from common import VAL_IMAGES_DIR
from common import NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
from common import STEPS_PER_EPOCH_FOR_TRAIN
from common import STEPS_PER_EPOCH_FOR_EVAL

from common import NUM_EPOCHS
from common import BATCH_SIZE
from common import NUM_CLASS
from common import LOG_DIR

def loss(logits, labels):

    return

def do_train(num_class, log_dir):

    with tf.device("/cpu:0"):
        # load data
        next_feature, next_label = load_data(JSON_TRAIN, TRAIN_IMAGES_DIR,
                                             is_training=True, batch_size=BATCH_SIZE,
                                             num_epochs=NUM_EPOCHS)

        x_test, y_test = load_data(JSON_VAL, VAL_IMAGES_DIR,
                                   is_training=False, batch_size=BATCH_SIZE,
                                   num_epochs=1)
        # 创建模型，定义输入输出
        model = resnet_model_keras(num_class)
        model.load_weights(filepath="weights\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                           by_name=True)
    #
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', verbose=1, save_weights_only=True,
                                 save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    parallel_model = multi_gpu_model(model, gpus=2)
    # loss_fn = losses.sparse_categorical_crossentropy()
    adm = optimizers.Adam()
    # metrics_fn = metrics.sparse_categorical_crossentropy()
    # 定义优化器，loss function，评估函数metrics
    model.compile(optimizer=adm,
                  loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"]
                  )
    # 训练模型
    # 记录运行计算图一次的时间
    # start_time = time.time()
    # math.ceil向上取整，一个epoch需要将数据集整个遍历一遍
    # steps_per_epoch = math.ceil(STEPS_PER_EPOCH_FOR_TRAIN)
    # val_steps = math.ceil(STEPS_PER_EPOCH_FOR_EVAL)

    steps_per_epoch = math.ceil(500/BATCH_SIZE)
    val_steps = math.ceil(500/BATCH_SIZE)

    # validation_steps = int(320/BATCH_SIZE)
    print("steps_per_epoch:",steps_per_epoch)

    hist = model.fit(next_feature, next_label, batch_size=None, epochs=NUM_EPOCHS,
                     validation_data=(x_test, y_test),
                     verbose=1,
                     callbacks=[checkpoint],
                     steps_per_epoch=steps_per_epoch,
                     validation_steps=val_steps)
    """
    hist = model.fit(next_feature, next_label, batch_size=None, epochs=NUM_EPOCHS,
                     verbose=1,
                     callbacks=[logging, checkpoint, reduce_lr, early_stopping],
                     steps_per_epoch=steps_per_epoch)
    """
    # duration_time = time.time() - start_time
    print("history loss:")
    # print(hist.history.keys())
    print(hist.history["loss"])
    print("history val_loss:")
    print(hist.history["loss"])

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOG_DIR + "/fig/res_model_loss.png")
    plt.show()

    # """
    plt.plot(hist.history["categorical_accuracy"])
    plt.plot(hist.history["val_categorical_accuracy"])
    plt.title("model accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOG_DIR + "/fig/res_model_top1.png")
    plt.show()

    plt.plot(hist.history["top3_accuracy"])
    plt.plot(hist.history["val_top3_accuracy"])
    plt.title("model accuracy3")
    plt.ylabel('accuracy3')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOG_DIR + "/fig/res_model_top3.png")
    plt.show()
    # """

    # 评估模型
    print("evaluate:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1, steps=val_steps)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    # 保存模型
    # model.save_weights(log_dir + "\scence_resnet_model.h5")



    return

log_dir = LOG_DIR + "/resnet50/"
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)

tf.gfile.MakeDirs(log_dir)
do_train(NUM_CLASS, log_dir)
