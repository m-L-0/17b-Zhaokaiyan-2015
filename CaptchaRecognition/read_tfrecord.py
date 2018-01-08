import tensorflow as tf
import matplotlib.pyplot as plt


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename = tf.matching_files(filename)
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()  # 文件读取器
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example, features={
                                        'label': tf.FixedLenFeature([], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)  # 用解码器 tf.decode_raw 解码。
    image = tf.cast(image, dtype='float32')*(1/255)-0.5  # 归一化处理
    image = tf.reshape(image, [40, 50, 3])  # 恢复数据形状
    image = tf.split(image, 3, 2)[0]
    label = tf.cast(features['label'], tf.int32)
    return image, label
list0 = []
image, label = read_and_decode('train*')
image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=10, capacity=2000, min_after_dequeue=1900, num_threads=2)
init = tf.local_variables_initializer()
with tf.Session() as sess:
    # 用训练集进行训练
    sess.run(init)
    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)   # 启动QueueRunner, 此时文件名队列已经进队
    for x in range(1):  # 规定出队数量
        img, label = sess.run([image_batch, label_batch])
        print("正确")
        print(label.shape, img.shape)
        for j in range(10):
            print(label[j])
            list0 = list(str(label[j]))  # 将验证码转化成列表list0
            if len(list0) is not 4:      # 对于不够4位的补0
                for m in range(4-len(list0)):
                    list0.append('None')
            print(list0)
            list1 = []  # 定义一个list1列表
            for n in range(4):  # 给列表添加个子列表,且每个列表内含10个0,代表0-9的位置
                list2 = []
                for m in range(10):
                    list2.append(0)
                list1.append(list2)
            for k in range(4):
                if list0[k] != 'None':
                    list1[k][int(list0[k])] = 1  # 让验证码的每位数，在list1中的子列表中显示出位置
            print(list1)
            im = img[j].reshape(40, 50)
            plt.imshow(im)
            plt.show()
    coord.request_stop()
    coord.join(threads)
