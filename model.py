import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import utils
from tensorflow.python.keras import layers, models, regularizers
from common import L2_WEIGHT_DECAY
from common import BATCH_NORM_DECAY
from common import BATCH_NORM_EPSILON
from common import MODEL_INPUT_SHAPE


def _conv_bn(input_tensor, filters, kernel_size, strides, con_name, bn_name, padding="valid"):

    # 创建一个conv2d-BN-relu小模块
    #
    # he_normal(seed=None)
    # He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入
    x = layers.Conv2D(filters, kernel_size, strides= strides,
                      padding=padding,
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

def _identity_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
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
    con_name_base = "res" +str(stage)  + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    filters1, filters2, filters3 = filters

    # 创建第一个conv2d-BN-relu小模块，卷积核大小为(1, 1)
    x = _conv_bn(input_tensor, filters1, (1, 1), strides, con_name_base + "2a", bn_name_base + "2a")
    x = layers.Activation("relu")(x)
    # 创建第二个conv2d-BN-relu小模块，卷积核大小为(3, 3)
    x = _conv_bn(x, filters2, kernel_size, strides, con_name_base + "2b", bn_name_base + "2b", padding="same")
    x = layers.Activation("relu")(x)
    # 创建第3个conv2d-BN-relu小模块，卷积核大小为(1, 1)
    x = _conv_bn(x, filters3, (1, 1), strides, con_name_base + "2c", bn_name_base + "2c")

    #add
    x = layers.add([x, input_tensor])
    x = layers.Activation("relu")(x)
    return x

def _conv_block(input_tensor,

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


    con_name_base = "res" + str(stage)  + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    filters1, filters2, filters3 = filters
    x = _conv_bn(input_tensor, filters1, (1, 1), (1, 1), con_name_base + "2a", bn_name_base + "2a")
    x = layers.Activation("relu")(x)

    x = _conv_bn(x, filters2, kernel_size, strides, con_name_base + "2b", bn_name_base + "2b", padding="same")
    x = layers.Activation("relu")(x)

    x = _conv_bn(x, filters3, (1, 1), (1, 1), con_name_base + "2c", bn_name_base + "2c")

    shortcut = _conv_bn(input_tensor, filters3, (1, 1), strides, con_name_base + "s1", bn_name_base + "s1", padding="same")

    x = layers.add([x, shortcut])

    x = layers.Activation("relu")(x)

    return x

def resnet_model_keras(num_classes):

    input_shape = MODEL_INPUT_SHAPE
    img_input = layers.Input(shape=input_shape)
    # 不考虑stride，要保证输入输出的维度一样，则卷积核大小和padding的数量对应关系为：
    # f=7时，p=3；f=5时，p=2；f=3时，p=1；
    # 将图片的尺寸由224，224，3变为227，227，3，保证stride前，输入输出的维度一致
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    # 残差块的输入进行预处理
    x = _conv_bn(x, 64, (7, 7), (2, 2), "conv1", "bn_conv1", padding="valid")
    x = layers.Activation("relu")(x)
    x = layers.ZeroPadding2D(padding=(1,1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(x)

    # Only the first block per block_layer uses projection_shortcut and strides
    # 每个block_layer中的第一个block在实现shortcut时，残差网络和输入的维度不同，故需要用projection_shortcut
    # 进行线性投影，即应用_conv_block, stride = 1,保证输入输出一致
    x = _conv_block(x, (3, 3), [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = _identity_block(x, (3, 3), [64, 64, 256], stage=2, block='b')
    x = _identity_block(x, (3, 3), [64, 64, 256], stage=2, block='c')

    # block_layer 2
    x = _conv_block(x, (3, 3), [128, 128, 512], stage=3, block='a')
    x = _identity_block(x, (3, 3), [128, 128, 512], stage=3, block='b')
    x = _identity_block(x, (3, 3), [128, 128, 512], stage=3, block='c')
    x = _identity_block(x, (3, 3), [128, 128, 512], stage=3, block='d')

    # block_layer 3
    x = _conv_block(x, 3, [256, 256, 1024], stage=4, block='a')

    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='b')

    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='c')

    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='d')

    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='e')

    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # block_layer 4
    x = _conv_block(x, 3, [512, 512, 2048], stage=5, block='a')

    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block='b')

    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # average pool,代替fc层
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    x = layers.Dense(

        num_classes, activation='softmax',

        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),

        bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),

        name='fc1000')(x)

    # 创建模型
    resnet_models = models.Model(img_input, x, name="resnet50")
    # print(resnet_models.summary())
    return resnet_models