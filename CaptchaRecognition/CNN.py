# 卷积神经网络
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ckpt_path', '/home/artemis/下载/yanzheng/cnn',"""存放模型的目录""")

tf.app.flags.DEFINE_string('model_name', 'yanzhenma',"""模型的名称""")

# 定义模型
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 参数分别指定了卷积核的尺寸、多少个channel、filter的个数即产生特征图的个数


# 2x2最大池化，即将一个2x2的像素块降为1x1的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著的特征。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_path)
if not ckpt:
    if not os.path.exists(FLAGS.ckpt_path):
        os.mkdir(FLAGS.ckpt_path)

  
sess = tf.InteractiveSession()  
# 在设计网络结构前，先定义输入的placeholder，x是特征，y是真实的label
x_image = tf.placeholder(tf.float32, [None, 40, 50, 1])
y = tf.placeholder(tf.int64, [None, 4, 11])


# 定义CNN结构
# 定义第一个卷积层，使用前面写好的d数进行参数初始化，包括weight和bias
W_conv1 = weight_variable([3, 3, 1, 32]) 
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# -1*20*28*64
# 池化
h_pool2 = max_pool_2x2(h_conv2)
# -1*10*14*96

# 定义第三个卷积层
W_conv3 = weight_variable([3, 3, 64, 96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# -1*10*14*96
# 池化
h_pool3 = max_pool_2x2(h_conv3)
# -1*5*7*96

# 全连接层
W_fc1 = weight_variable([5*7*96, 512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool3, [-1, 5*7*96])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# --正则化--
# 为了减轻过拟合，使用Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Dropout层输出连接一个Softmax层,得到最后的概率输出
# 输出层一
W_fc_1 = weight_variable([512, 11])
b_fc_1 = bias_variable([11])
pred_1 = tf.matmul(h_fc1_drop, W_fc_1) + b_fc_1


# 输出层二
W_fc_2 = weight_variable([512, 11])
b_fc_2 = bias_variable([11])
pred_2 = tf.matmul(h_fc1_drop, W_fc_2) + b_fc_2


# 输出层三
W_fc_3 = weight_variable([512, 11])
b_fc_3 = bias_variable([11])
pred_3 = tf.matmul(h_fc1_drop, W_fc_3) + b_fc_3


# 输出层四
W_fc_4 = weight_variable([512, 11])
b_fc_4 = bias_variable([11])
pred_4 = tf.matmul(h_fc1_drop, W_fc_4) + b_fc_4

pred = tf.concat([pred_1, pred_2, pred_3, pred_4], 1)
pred = tf.reshape(pred, [-1, 4, 11])

cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred[:, 0], labels=y[:, 0]))
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred[:, 1], labels=y[:, 1]))
cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred[:, 2], labels=y[:, 2]))
cost4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred[:, 3], labels=y[:, 3]))
cost = tf.reduce_sum([cost1, cost2, cost3, cost4])


# 输出层和交叉熵的定义
out_op = tf.train.AdamOptimizer().minimize(cost)

# 预测正确率
# d = tf.argmax(pred, 2)
corr = tf.equal(tf.argmax(pred, 2), tf.argmax(y, 2)) 
accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
saver = tf.train.Saver(max_to_keep=2)


# 训练模型
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

# def to_reshape(label, batch_size):
#     lab = np.zeros([len(label), 4, 11])
#     for i in range(len(label)):
#         for m, n in enumerate(str(label[i])+''*(4-len(str(label[i])))):
#             if n == '':
#                 lab[i][m][10] = 1
#             else:
#                 lab[i][m][int(n)] = 1
#     return lab
# 改变标签的格式
def to_reshape(label, batc_size):
    list0 = []
    list1 = []
    label_list = []
    for j in range(200):
        list0 = list(str(label[j]))  # 将验证码转化成列表list0
        if len(list0) is not 4:  # 对于不够4位的补0
            for m in range(4 - len(list0)):
                list0.append('None')
        list1 = []  # 定义一个list1列表
        for n in range(4):  # 给列表添加个子列表,且每个列表内含11个0,代表0-9和None的位置
            list2 = []
            for m in range(11):
                list2.append(0)
            list1.append(list2)
        for k in range(4):
            if list0[k] != 'None':
                list1[k][int(list0[k])] = 1  # 让验证码的每位数，在list1中的子列表中显示出位置
            else:
                list1[k][10] = 1
        label_list.append(list1)
    return label_list
# 将label转化为(?, 4, 10)
batch_size = 200
# 加载训练集数据
image, label = read_and_decode('train*')
image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                  capacity=5000, min_after_dequeue=1900, num_threads=2)

# 加载验证集数据
vimage, vlabel = read_and_decode('validation*')
vimage_batch, vlabel_batch = tf.train.shuffle_batch([vimage, vlabel], batch_size=batch_size,
                                                  capacity=5000, min_after_dequeue=1900, num_threads=2)
# 加载测试数据
timage, tlabel = read_and_decode('test*')
timage_batch, tlabel_batch = tf.train.shuffle_batch([timage, tlabel], batch_size=batch_size,
                                                  capacity=5000, min_after_dequeue=1900, num_threads=2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(FLAGS.ckpt_path, sess.graph)
    
    ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    step = 0
    if ckpt:
        saver.restore(sess=sess, save_path=ckpt)
        step = int(ckpt[len(os.path.join(FLAGS.ckpt_path, FLAGS.model_name)) + 1:])
    # 用训练集进行训练
    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)   # 启动QueueRunner, 此时文件名队列已经进队
    # for x in range(20000):  # 规定出队数量
    #     img, label = sess.run([image_batch, label_batch])
    #     Tlabel = to_reshape(label, batch_size)
    #     vimg, vlabel = sess.run([vimage_batch, vlabel_batch])
    #     Vlabel = to_reshape(vlabel, batch_size)
    img, label = sess.run([image_batch, label_batch])
    timg, tlabel = sess.run([timage_batch, tlabel_batch])
    tlabel = to_reshape(tlabel, batch_size)

        # if x % 100 == 0:  # 每100次训练，对准确率进行一次测试
        #     train_accuracy, train_op, train_cost = sess.run([accuracy, out_op, cost],
        #                                                     feed_dict={x_image: img, y: Tlabel, keep_prob: 1.0})  # 测试集精准度
        #     ckptname = os.path.join(FLAGS.ckpt_path, FLAGS.model_name)
        #     saver.save(sess, ckptname, global_step=x)
        #     val_accuracy, val_op, val_cost = sess.run([accuracy, out_op, cost],
        #                                                     feed_dict={x_image: vimg, y: Vlabel, keep_prob: 1.0})
        #     print("step: %d  训练集 ACCURACY: %.3f     COST: %.3f" % (x, train_accuracy, train_cost))
        #     print("step: %d  验证集 ACCURACY: %.3f     COST: %.3f" % (x, val_accuracy, val_cost))
    test_accuracy, test_op, test_cost = sess.run([accuracy, out_op, cost],
                                            feed_dict={x_image: timg, y: tlabel, keep_prob: 1.0})  # 测试集精准度
    print("测试集 ACCURACY: %.3f     COST: %.3f" % (test_accuracy, test_cost))                                                        
    coord.request_stop()
    coord.join(threads)
