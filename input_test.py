import tensorflow as tf
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# 创建0-10的数据集，每个batch取个数。
dataset = tf.data.Dataset.range(10).batch(6)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(2):
        value = sess.run(next_element)
        print(value)


print('---------------------------')
dataset2 = tf.data.Dataset.range(10).shuffle(buffer_size=100, seed=1000)
dataset2 = dataset2.map(lambda x: x + random.randint(10, 20))
dataset2 = dataset2.batch(6).repeat()

iterator2 = dataset2.make_one_shot_iterator()
next_element2 = iterator2.get_next()

with tf.Session() as sess2:
    for i in range(10):
        value2 = sess2.run(next_element2)
        print(value2)
        print('-----------------------')