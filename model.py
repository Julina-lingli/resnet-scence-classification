import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import utils
from tensorflow.python.keras import layers, models, regularizers
from common import L2_WEIGHT_DECAY
from common import BATCH_NORM_DECAY
from common import BATCH_NORM_EPSILON
from common import MODEL_INPUT_SHAPE


def _conv_bn(input_tensor, filters, kernel_size, strides, con_name, bn_name):

    # 创建一个conv2d-BN-relu小模块
    #
    # he_normal(seed=None)
    # He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入
    x = layers.Conv2D(filters, kernel_size, stride= strides,
                      padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=con_name)(input_tensor)
    #
    x = layers.BatchNormalization(axis=3,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name)(x)

    return x

def identity_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments

        input_tensor: input tensor

        kernel_size: default 3, the kernel size of

            middle conv layer at main path

        filters: list of integers, the filters of 3 conv layer at main path

        stage: integer, current stage label, used for generating layer names

        block: 'a','b'..., current block label, used for generating layer names



    # Returns

        Output tensor for the block.

    """
    con_name_base = "res" +str(stage) + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    filters1, filters2, filters3 = filters

    # 创建第一个conv2d-BN-relu小模块，卷积核大小为(1, 1)
    x = _conv_bn(input_tensor, filters1, (1, 1), strides, con_name_base + "2a", bn_name_base + "2a")
    x = layers.Activation("relu")(x)
    # 创建第二个conv2d-BN-relu小模块，卷积核大小为(3, 3)
    x = _conv_bn(x, filters2, kernel_size, strides, con_name_base + "2b", bn_name_base + "2b")
    x = layers.Activation("relu")(x)
    # 创建第3个conv2d-BN-relu小模块，卷积核大小为(1, 1)
    x = _conv_bn(x, filters3, (1, 1), strides, con_name_base + "2c", bn_name_base + "2c")

    #add
    x = layers.add([x, input_tensor])
    x = layers.Activation("relu")(x)
    return x

def conv_block(input_tensor,

               kernel_size,

               filters,

               stage,

               block,

               strides=(2, 2)):

    """A block that has a conv layer at shortcut.



    # Arguments

    input_tensor: input tensor

    kernel_size: default 3, the kernel size of

    middle conv layer at main path

    filters: list of integers, the filters of 3 conv layer at main path

    stage: integer, current stage label, used for generating layer names

    block: 'a','b'..., current block label, used for generating layer names

    strides: Strides for the second conv layer in the block.



    # Returns

    Output tensor for the block.



    Note that from stage 3,

    the second conv layer at main path is with strides=(2, 2)

    And the shortcut should have strides=(2, 2) as well

    """


    con_name_base = "res" + str(stage) + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    filters1, filters2, filters3 = filters
    x = _conv_bn(input_tensor, filters1, (1, 1), (1, 1), con_name_base + "2a", bn_name_base + "2a")
    x = layers.Activation("relu")(x)

    x = _conv_bn(x, filters2, kernel_size, strides, con_name_base + "2b", bn_name_base + "2b")
    x = layers.Activation("relu")(x)

    x = _conv_bn(x, filters3, (1, 1), (1, 1), con_name_base + "2c", bn_name_base + "2c")

    shortcut = _conv_bn(input_tensor, filters3, (1, 1), strides, con_name_base + "s1", bn_name_base + "s1")

    x = layers.add([x, shortcut])

    x = layers.Activation("relu")(x)

    return x

def resnet_model_keras():

    input_shape = MODEL_INPUT_SHAPE
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
