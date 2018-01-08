import csv
import tensorflow as tf
import random
from PIL import Image


# 读取csv文件
csv_reader = csv.reader(open('./data/captcha/labels/labels.csv', encoding='utf-8'))
data = []  # 定义一个数据列表
i = 0
j = 0
m = 0
# 将图片名和标签名存入列表
for row in csv_reader:
    data.append(row)

# 生成训练集tfrecords文件
train_num = random.sample(data, int(0.8*len(data)))
while len(train_num) >= 4000:
    writer = tf.python_io.TFRecordWriter("train" + str(i) + ".tfrecords")  # 创建一个train.tfrecords文件
    train_data = train_num[:4000]
    for name, label in train_data:
        img = Image.open(name)  # 定义操作对象
        img = img.resize((40, 50))  # 将图片改为40,50
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
    del(train_num[:4000])
    i += 1
# # ----------------------------------------------------------------------------------------------
# 生成测试集tfrecord文件
test_num = random.sample(data, int(0.1*len(data)))
while len(test_num) >= 4000:
    writer = tf.python_io.TFRecordWriter("test" + str(j) + ".tfrecords")  # 创建一个test.tfrecords文件
    test_data = test_num[:4000]
    for name, label in test_data:
        img = Image.open(name)  # 定义操作对象
        img = img.resize((40, 50))  # 将图片改为40,50
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
    del (test_num[:4000])
    j += 1
# # ------------------------------------------------------------------------------------------------
# # 生成验证集tfrecord文件
validation_num = random.sample(data, int(0.1*len(data)))
writer = tf.python_io.TFRecordWriter("./validation" + str(m) + ".tfrecords")  # 创建一个validation .tfrecord文件
while len(validation_num) >= 4000:
    validation_data = validation_num[:4000]
    for name, label in validation_data:
        img = Image.open(name)  # 定义操作对象
        img = img.resize((40, 50))  # 将图片改为40,50
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
    del (validation_num[:4000])
    m += 1
