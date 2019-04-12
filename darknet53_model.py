from functools import wraps
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Add
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.regularizers import l2
# from tensorflow.keras.models import Model
from keras.models import Model
from common import L2_WEIGHT_DECAY
from common import BATCH_NORM_DECAY
from common import BATCH_NORM_EPSILON
# from common import MODEL_INPUT_SHAPE
MODEL_INPUT_SHAPE = (416, 416, 3)
L2_WEIGHT_DECAY = 5e-4
LEAKY_ALPHA = 0.1

@wraps(Conv2D)
def Darknet53Conv2D(*args, **kwargs):
    """Wrapper to set Darknet53 parameters for Convolution2D."""
    darknet53_conv_kwargs = {'kernel_regularizer': l2(L2_WEIGHT_DECAY)}
    darknet53_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet53_conv_kwargs['use_bias'] = False
    darknet53_conv_kwargs.update(kwargs)
    print("darknet53_conv_kwargs:", darknet53_conv_kwargs)
    return Conv2D(*args, **darknet53_conv_kwargs)

def Darknet53Conv2D_BN_Leaky(x, num_filters, kernel_size, strides=(1, 1)):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    x = Darknet53Conv2D(num_filters, kernel_size, strides=strides)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=LEAKY_ALPHA)(x)
    return x

def Darknet53Res_unit(x, num_filters):
    # strides=(1,1),采用same卷积，输出和输入尺寸一致
    res = Darknet53Conv2D_BN_Leaky(x, num_filters//2, (1, 1))
    res = Darknet53Conv2D_BN_Leaky(res, num_filters, (3, 3))
    x = Add()([x, res])
    return x


def Darknet53Resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = Darknet53Conv2D_BN_Leaky(x, num_filters, (3,3), strides=(2,2))
    for i in range(num_blocks):
        y = Darknet53Res_unit(x, num_filters)
        x = y
    return x

def darknet53_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    # DBL
    x = Darknet53Conv2D_BN_Leaky(x, 32, (3,3))
    # res1
    x = Darknet53Resblock_body(x, 64, 1)
    # res2
    x = Darknet53Resblock_body(x, 128, 2)
    # res8
    x = Darknet53Resblock_body(x, 256, 8)
    # res8
    x = Darknet53Resblock_body(x, 512, 8)
    # res4
    x = Darknet53Resblock_body(x, 1024, 4)
    return x

def darknet53_model_keras(num_classes):
    input_shape = MODEL_INPUT_SHAPE
    img_input = Input(shape=input_shape)

    x = darknet53_body(img_input)

    x = GlobalAveragePooling2D(name='avg_pool')(x)

    x = Dense(num_classes, activation='softmax', use_bias=False,
              kernel_regularizer=l2(L2_WEIGHT_DECAY),
              name="FC80")(x)

    darknet53_model = Model(img_input, x, name="darknet53")

    return darknet53_model