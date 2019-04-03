import tensorflow as tf
import matplotlib.pyplot as plt
# from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
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

import math


def resnet50_model(lr=0.005, decay=1e-6, momentum=0.9, nb_classes=80):

    base_model = ResNet50(weights="D:\githubProjects\\resnet-scence-classification\weights\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                          include_top=False, pooling='avg',
                         input_shape=(224, 224, 3), classes=nb_classes)
    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    for layer in base_model.layers[ : -8]:
        layer.trainable = False
        # print(layer.name)

    x = base_model.output

    # 添加自己的分类层
    predic = layers.Dense(nb_classes, activation='softmax')(x)

    tl_model = models.Model(base_model.input, predic)

    # 可观察模型结构
    tl_model.summary()
    def top3_accuracy(y, predic):
        return metrics.top_k_categorical_accuracy(y, predic, k=3)
    # top3 = metrics.top_k_categorical_accuracy(labels, predic, k=3)
    adm = optimizers.Adam()

    tl_model.compile(optimizer=adm,
                  loss="categorical_crossentropy",
                  metrics=[metrics.categorical_accuracy, top3_accuracy]
                  )

    return tl_model

def train(log_dir):
    # load data
    next_feature, next_label = load_data(JSON_TRAIN, TRAIN_IMAGES_DIR,
                                         is_training=True, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)

    x_test, y_test = load_data(JSON_VAL, VAL_IMAGES_DIR,
                               is_training=False, batch_size=BATCH_SIZE, num_epochs=1)

    tl_model = resnet50_model()

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5',
                                 monitor='loss', verbose=1, save_weights_only=True,
                                 save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
    # math.ceil向上取整，一个epoch需要将数据集整个遍历一遍
    steps_per_epoch = math.ceil(STEPS_PER_EPOCH_FOR_TRAIN)
    val_steps = math.ceil(STEPS_PER_EPOCH_FOR_EVAL)

    steps_per_epoch = math.ceil(500/BATCH_SIZE)
    val_steps= math.ceil(500/BATCH_SIZE)

    # validation_steps = int(320/BATCH_SIZE)
    print("steps_per_epoch:", steps_per_epoch)
    hist = tl_model.fit(next_feature, next_label, batch_size=None, epochs=NUM_EPOCHS,
                        validation_data=(x_test, y_test),
                        verbose=1,
                        callbacks=[checkpoint],
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps)
    # duration_time = time.time() - start_time
    print("history:")
    # print(hist.history.keys())
    print(hist.history["loss"])
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOG_DIR + "/fig/model_loss.png")
    plt.show()

    # """
    plt.plot(hist.history["categorical_accuracy"])
    plt.plot(hist.history["val_categorical_accuracy"])
    plt.title("model accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOG_DIR + "/fig/model_top1.png")
    plt.show()

    plt.plot(hist.history["top3_accuracy"])
    plt.plot(hist.history["val_top3_accuracy"])
    plt.title("model accuracy3")
    plt.ylabel('accuracy3')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOG_DIR + "/fig/model_top3.png")
    plt.show()
    # """
    # 评估模型
    print("evaluate:")
    loss, accuracy , top3 = tl_model.evaluate(x_test, y_test, verbose=1, steps=val_steps)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    print('Test accuracy of top3:', top3)

    return

log_dir = LOG_DIR + "/tl"
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)

tf.gfile.MakeDirs(log_dir)
train(log_dir)