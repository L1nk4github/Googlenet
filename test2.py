import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

test_path = 'test_set' # 你可以修改这个路径为你的数据集路径
training_path = 'training_set'

# 创建一个ImageDataGenerator对象，用于对图像进行预处理和增强
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, # 将像素值归一化到0-1之间
    rotation_range=10, # 随机旋转图像的角度范围
    width_shift_range=0.1, # 随机水平平移图像的比例范围
    height_shift_range=0.1, # 随机垂直平移图像的比例范围
    zoom_range=0.1, # 随机缩放图像的比例范围
    horizontal_flip=True, # 随机水平翻转图像
    validation_split=0.2 # 划分验证集的比例
)

# 使用datagen.flow_from_directory方法来从数据集路径中加载数据，指定图像大小，批次大小，类别模式和子集类型
train_generator = datagen.flow_from_directory(
    training_path, # 数据集路径
    target_size=(224,224), # 图像大小为224x224
    batch_size=32, # 批次大小为32
    class_mode='sparse', # 类别模式为稀疏，即用整数来表示类别标签
    subset='training' # 子集类型为训练集
)

validation_generator = datagen.flow_from_directory(
    test_path, # 数据集路径
    target_size=(224,224), # 图像大小为224x224
    batch_size=32, # 批次大小为32
    class_mode='sparse', # 类别模式为稀疏，即用整数来表示类别标签
    subset='validation' # 子集类型为验证集
)


# 定义一个函数来创建一个卷积层，输入是一个张量x，输出是一个卷积后的张量y
def conv_layer(x, filters, kernel_size, strides, padding, activation):
  y = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x) # 创建一个卷积层，并且对x进行卷积操作
  y = tf.keras.layers.BatchNormalization()(y) # 创建一个批量归一化层，并且对y进行归一化操作
  y = tf.keras.layers.Activation(activation)(y) # 创建一个激活层，并且对y进行激活操作
  return y

# 定义一个函数来创建一个最大池化层，输入是一个张量x，输出是一个池化后的张量y
def max_pool_layer(x, pool_size, strides, padding):
  y = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding)(x) # 创建一个最大池化层，并且对x进行池化操作
  return y

# 定义一个函数来创建一个平均池化层，输入是一个张量x，输出是一个池化后的张量y
def avg_pool_layer(x, pool_size, strides, padding):
  y = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(x) # 创建一个平均池化层，并且对x进行池化操作
  return y

# 定义一个函数来创建一个全连接层，输入是一个张量x，输出是一个全连接后的张量y
def dense_layer(x, units, activation):
  y = tf.keras.layers.Dense(units=units, activation=activation)(x) # 创建一个全连接层，并且对x进行全连接操作
  return y

# 定义一个函数来创建一个inception模块，输入是一个张量x，输出是一个inception后的张量y
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
  # 创建四个分支
  branch_1 = conv_layer(x, filters=filters_1x1, kernel_size=1, strides=1, padding="same", activation="relu") # 第一个分支是1x1卷积
  branch_2 = conv_layer(x, filters=filters_3x3_reduce, kernel_size=1, strides=1, padding="same", activation="relu") # 第二个分支是1x1卷积后接3x3卷积
  branch_2 = conv_layer(branch_2, filters=filters_3x3, kernel_size=3, strides=1, padding="same", activation="relu")
  branch_3 = conv_layer(x, filters=filters_5x5_reduce, kernel_size=1, strides=1, padding="same", activation="relu") # 第三个分支是1x1卷积后接5x5卷积
  branch_3 = conv_layer(branch_3, filters=filters_5x5, kernel_size=5, strides=1, padding="same", activation="relu")
  branch_4 = max_pool_layer(x, pool_size=3, strides=1, padding="same") # 第四个分支是3x3最大池化后接1x1卷积
  branch_4 = conv_layer(branch_4, filters=filters_pool_proj, kernel_size=1, strides=1, padding="same", activation="relu")

  # 合并四个分支
  y = tf.keras.layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4]) # 按照通道维度进行拼接，得到inception的输出
  return y

# 定义一个函数来创建一个googlenet模型，输入是一个张量x，输出是一个模型对象model
def create_googlenet(input_shape, num_classes):
  # 创建模型的第一部分，包括7x7卷积，最大池化和LRN
  inputs = tf.keras.Input(shape=input_shape) # 创建一个输入层，并且指定输入形状
  x = conv_layer(inputs, filters=64, kernel_size=7, strides=2, padding="same", activation="relu") # 7x7卷积
  x = max_pool_layer(x, pool_size=3, strides=2, padding="same") # 最大池化
  x = tf.nn.lrn(x) # LRN

  # 创建模型的第二部分，包括1x1卷积，3x3卷积，最大池化和LRN
  x = conv_layer(x, filters=64, kernel_size=1, strides=1, padding="same", activation="relu") # 1x1卷积
  x = conv_layer(x, filters=192, kernel_size=3, strides=1, padding="same", activation="relu") # 3x3卷积
  x = max_pool_layer(x, pool_size=3, strides=2, padding="same") # 最大池化
  x = tf.nn.lrn(x) # LRN

  # 创建模型的第三部分，包括两个inception模块和一个最大池化层
  x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32) # 第一个inception模块
  x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64) # 第二个inception模块
  x = max_pool_layer(x, pool_size=3, strides=2, padding="same") # 最大池化层

  # 创建模型的第四部分，包括五个inception模块和一个平均池化层
  x = inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16,
                       filters_5x5=48, filters_pool_proj=64)  # 第一个inception模块
  x = inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24,
                       filters_5x5=64, filters_pool_proj=64)  # 第二个inception模块
  x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24,
                       filters_5x5=64, filters_pool_proj=64)  # 第三个inception模块
  x = inception_module(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32,
                       filters_5x5=64, filters_pool_proj=64)  # 第四个inception模块
  x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32,
                       filters_5x5=128, filters_pool_proj=128)  # 第五个inception模块
  x = avg_pool_layer(x, pool_size=7, strides=1, padding="valid")  # 平均池化层

  # 创建模型的第五部分，包括一个dropout层，一个全连接层和一个softmax层
  x = tf.keras.layers.Dropout(rate=0.4)(x)  # 创建一个dropout层，并且对x进行dropout操作
  x = tf.keras.layers.Flatten()(x)  # 创建一个展平层，并且对x进行展平操作
  x = dense_layer(x, units=num_classes, activation="softmax")  # 创建一个全连接层，并且对x进行全连接操作，得到输出

  # 创建模型对象，指定输入和输出
  model = tf.keras.Model(inputs=inputs, outputs=x)

  # 返回模型对象
  return model


# 定义一些超参数
learning_rate = 0.01  # 学习率
batch_size = 32  # 批次大小
epochs = 10  # 迭代次数

# 定义输入形状和类别数
input_shape = (224,224,3) # 图像大小为224x224，通道数为3
num_classes = train_generator.num_classes # 类别数为训练集中不同标签的个数

# 调用函数创建googlenet模型对象
model = create_googlenet(input_shape=input_shape, num_classes=num_classes)

# 打印模型的结构
model.summary()

# 定义一个优化器对象，使用随机梯度下降(SGD)算法，并且指定学习率
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# 定义一个损失函数对象，使用交叉熵损失函数，并且指定标签是稀疏的
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# 定义一个评估指标对象，使用准确率指标，并且指定名称为accuracy
metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

# 编译模型，指定优化器，损失函数和评估指标
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# 训练模型，指定训练集，批次大小，迭代次数和验证集
history = model.fit(train_generator, batch_size=batch_size, epochs=epochs, validation_data=validation_generator)

# 测试模型，指定测试集
model.evaluate(validation_generator)

# # 预测模型，指定一个图片文件路径
# image_path = "image2.jpg" # 你可以修改这个路径为你想要预测的图片文件路径
# image = load_image(image_path) # 调用之前定义的函数加载图片
# image = np.expand_dims(image, axis=0) # 增加一个批次维度
# prediction = model.predict(image) # 对图片进行预测，得到一个概率分布
# label = np.argmax(prediction) # 取概率最大的索引作为标签
# print("The predicted label is:", label) # 打印预测的标签

