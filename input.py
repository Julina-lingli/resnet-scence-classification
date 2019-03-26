
import tensorflow as tf
import os
import json
from PIL import Image
import numpy as np


from common import JSON_TRAIN
from common import TRAIN_IMAGES_DIR
from common import HEIGHT
from common import WIDTH
from common import MAX_STEPS
from common import NUM_EPOCHS
from common import BATCH_SIZE
from common import LOG_DIR

def get_image_label_from_json(json_path, images_dir):
    with open(json_path, "r") as f:
        image_label_list = json.load(f)
        print(image_label_list[:2])
        image_label_dict = dict()
        for info in image_label_list:
            image_label_dict[info["image_id"]] = int(info["label_id"])

        image_name_list = list(image_label_dict.keys())
        label_list = list(image_label_dict.values())
        filenames = []
        for image_name in image_name_list:
            filename = os.path.join(images_dir, image_name)
            filenames.append(filename)
        print(type(filenames), type(label_list))
    return filenames, label_list

def _preprocess_image(image, is_training):

    """Preprocess a single image of layout [height, width, depth]."""
    print("_preprocess_image", image)
    if is_training:

        # Randomly flip the image horizontally.
        # 随机水平翻转图像
        image = tf.image.random_flip_left_right(image)
        # 在某范围随机调整图片亮度
        # image = tf.image.random_brightness(image, max_delta=63)
        # 在某范围随机调整图片对比度
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    # 减去平均值并除以像素的方差，白化操作：均值变为0，方差变为1
    image = tf.image.per_image_standardization(image)

    return image

# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label, is_training):
    print("_parse_function filename:", filename)
    image_string = tf.read_file(filename)
    print("image_string", image_string)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # image_decoded = tf.image.decode_image(image_string, channels=3)
    # image_decoded = tf.image.decode_image(image_string)
    print("image_decoded", image_decoded)
    # image_resized = tf.image.resize_images(image_decoded, [HEIGHT, WIDTH])
    new_image = tf.image.resize_image_with_crop_or_pad(image_decoded, HEIGHT, WIDTH)

    new_image = _preprocess_image(new_image, is_training)

    print("new_image", new_image)

    new_image = tf.cast(new_image, tf.uint8)
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, 80)

    return new_image, label


"""
def img_resize(imgpath, img_size):
    img = Image.open(imgpath)
    if (img.width > img.height):
        scale = float(img_size) / float(img.height)
        img = np.array(cv2.resize(np.array(img), (
            int(img.width * scale + 1), img_size))).astype(np.float32)
    else:
        scale = float(img_size) / float(img.width)
        img = np.array(cv2.resize(np.array(img), (
            img_size, int(img.height * scale + 1)))).astype(np.float32)
    img = (img[
           (img.shape[0] - img_size) // 2:
           (img.shape[0] - img_size) // 2 + img_size,
           (img.shape[1] - img_size) // 2:
           (img.shape[1] - img_size) // 2 + img_size,
           :] - 127) / 255
    return img
"""

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    """ 
    lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充,
    调整图像尺寸 具体而言,此时某个边正好可以等于目标长度,
    另一边小于等于目标长度 将缩放后的数据拷贝到画布中心,返回完成缩放 
    """

    iw, ih = image.size
    w, h = size #inp_dim是需要resize的尺寸（如416*416）
    # 取min(w/img_w, h/img_h)这个比例来缩放，缩放后的尺寸为new_w, new_h,
    # 即保证较短的边缩放后正好等于目标长度(需要的尺寸)，另一边的尺寸缩放后还没有填充满.
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    # 将图片按照纵横比不变来缩放为new_w x new_h，768 x 576的图片缩放成416x312.
    image = image.resize((nw,nh), Image.BICUBIC)

    # 创建一个画布, 将resized_image数据拷贝到画布中心。
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _py_parse_function(filename, label):
    # import cv2
    # image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
    print("filename shape:", filename.shape)
    print("filename:", filename)
    filename = list(filename)
    image = Image.open(filename)
    print("image:", image)
    image.show()
    """
    # 另外一种显示图片
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()
    """

    image_resized = letterbox_image(image, (WIDTH, HEIGHT))
    print("image_resized:", image_resized)
    # image_resized.show()
    # PLT处理图像后，最后需要配合np.array使用，shape转换为（HEIGHT，WIDTH，CHANNEL）,以便后面卷积操作
    image = np.array(image_resized, dtype="float32")
    print(image.shape)
    # print("image_resized1:", image_resized)
    """
    from matplotlib import pyplot as plt
    plt.imshow(image_resized)
    plt.show()
    """
    # image = _preprocess_image(image, is_training)

    print("label shape:", label.shape)

    return image, label

# _parse_function("D:\datasets\\ai_challenger_scene\scene_train_images_20170904\\00000ae5e7fcc87222f1fb6f221e7501558e5b08.jpg",1)

def load_data(json_path, images_dir, is_training, batch_size, num_epochs):
    #
    filenames, label_list = get_image_label_from_json(json_path, images_dir)
    print("filenames", filenames[:2])
    # 图片文件的列表
    # filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
    filenames_tensor = tf.constant(filenames[:500])
    print("filenames_tensor:", filenames_tensor)
    # label[i]就是图片filenames[i]的label
    labels_tensor = tf.constant(label_list[:500])

    # features_placeholder = tf.placeholder(filenames_tensor.dtype, filenames_tensor.shape)
    # labels_placeholder = tf.placeholder(labels_tensor.dtype, labels_tensor.shape)
    # 将filenames, label_list分割组合成一个个(filename, label)这样的元组
    # 此时dataset中的一个元素是(filename, label)，shape为：((), ()), types: (tf.string, tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames_tensor, labels_tensor))
    print("dataset：",dataset)
    # 基于 tf.placeholder() 张量定义 Dataset，并使用可初始化 Iterator，然后在初始化 dataset 的 Iterator 时将 NumPy 数组供给程序。
    # dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    """
    # tf.py_func可以定义自己的图片处理函数，但是这里没有处理好
    dataset = dataset.map(
        lambda filename, label: tf.py_func(
            _py_parse_function, [filenames_tensor, labels_tensor], [tf.uint8, labels_tensor.dtype]))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(lambda x, y: _preprocess_image(x, y, is_training))
    """
    # 采用tf的标准图像解码函数，将dataset数据集中的每个元素都应用于_parse_function
    dataset = dataset.map(lambda x, y:_parse_function(x, y, is_training))
    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    # dataset = dataset.shuffle(buffersize=10).batch(batch_size).repeat(num_epochs)
    # 划分batch，epoch
    # dataset = dataset.batch(batch_size).repeat(num_epochs)
    dataset = dataset.batch(batch_size).repeat()
    # 初始化迭代器，遍历一遍数据集
    iterator = dataset.make_one_shot_iterator()
    #
    # iterator = dataset.make_initializable_iterator()
    """
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                              labels_placeholder: labels})
    """
    # 返回的one_element为batch_size个（_labels, _features）
    next_feature, next_label = iterator.get_next()
    # Display the training images in the visualizer.
    # f.summary.image（）中传入的第一个参数是命名，第二个是图片数据，第三个是最多展示的张数，此处为10张
    # Tensor必须是4-D形状[batch_size, height, width, channels]
    tf.summary.image('images', next_feature, 10)
    print("next_label", next_label)
    print("next_feature", next_feature)

    """
    #
    #测试图片在tensorboard中的显示
    # # 汇总记录节点
    summary_merge = tf.summary.merge_all()


    # test
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
       
        # sess.run(iterator.initializer, feed_dict={features_placeholder: filenames,
        #                                              labels_placeholder: label_list})
        
        #
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        steps =0
        try:
            while steps < 2:
                summary, new_image = sess.run([summary_merge, next_feature])
                print("_features", new_image)
                print(new_image.shape)
                # 运行并写入日志
                summary_writer.add_summary(summary)
                from matplotlib import pyplot as plt
                plt.imshow(new_image[0])
                plt.show()
                print("labels:", sess.run(next_label))
                steps +=1

            summary_writer.close()

        except tf.errors.OutOfRangeError:
            print("End of dataset per epoch")

    """

    return next_feature, next_label

# load_data(JSON_TRAIN, TRAIN_IMAGES_DIR, is_training=True, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)












