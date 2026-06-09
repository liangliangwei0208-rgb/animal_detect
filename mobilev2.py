import os
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# 显存不足，切换为cpu训练，若想gpu训练，注释掉这两行代码:
'''os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"'''

# 显存自适应分配
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 设置混合精度策略
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

np.set_printoptions(threshold=np.inf)

# 自制数据集：
# 数据文件夹
data_dir = "./pic_object"
VALIDATION_SPLIT = 0.15
BATCH_SIZE = 32  # GTX 1650 只有约4GB显存，批量太大容易显存不足
# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):  #  返回该文件下的目录
        fpath = os.path.join(data_dir, fname)  # 将文件路径和目录内容合并为图片路径
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas, dtype='float32')  # 降低精度，防止超载内存
    labels = np.array(labels, dtype='int')

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    # 只打印每类数量，避免一次输出几千个标签导致终端刷屏
    label_ids, label_counts = np.unique(labels, return_counts=True)
    print("label counts:", dict(zip(label_ids.tolist(), label_counts.tolist())))
    return fpaths, datas, labels

fpaths, datas, labels = read_data(data_dir)

np.random.seed(200)
np.random.shuffle(datas)
np.random.seed(200)
np.random.shuffle(labels)

# 读取自制训练集：
# 先在CPU/Numpy里手动切分训练集和验证集，避免Keras validation_split在GPU上搬运整份大数组
split_index = int(len(datas) * (1 - VALIDATION_SPLIT))
x_train = datas[:split_index]
y_train = labels[:split_index]   # 0：bird； 1：butterfly； 2：cat； 3：dog； 4：tigger
x_val = datas[split_index:]
y_val = labels[split_index:]
print("train samples: {}\tvalidation samples: {}".format(len(x_train), len(x_val)))

# 数据增强：
image_gen_train = ImageDataGenerator(
    rescale=1. / 1,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5,  # 将图像随机缩放阈量50％
    fill_mode='nearest',
)
image_gen_train.fit(x_train)

mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
                                                           include_top=False,
                                                           weights='imagenet',
                                                           pooling='avg')

mobilenet.trainable = True

model = tf.keras.Sequential()
model.add(mobilenet)
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

checkpoint_save_path = "./checkpoint1/trash_model.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model weights-----------------')
    # 先加载模型权重，再compile优化器，避免旧checkpoint里的优化器状态和混合精度LossScaleOptimizer冲突
    load_status = model.load_weights(checkpoint_save_path)
    if hasattr(load_status, 'expect_partial'):
        load_status.expect_partial()

# 使用混合精度训练优化器
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 保存模型：
cp_callback_list = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=0),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=0.00002,
                                     patience=5,
                                     verbose=1,
                                     mode='min',
                                     baseline=None,
                                     restore_best_weights=False),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.50,
                                         patience=3,
                                         verbose=1,
                                         mode='min',
                                         min_delta=0.0001,
                                         cooldown=1,
                                         min_lr=0)
]

# fit中执行训练过程：告知训练集和测试集的输入特征和标签；告知每个batch是多少，告知要迭代多少次数据集
# 告知测试集，告知数据集迭代次数，用测试集验证准确率；告知1次数据迭代，打印出验证的数据；使用回调函数，实现断点续训
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=200, validation_data=(x_val, y_val), validation_freq=1,
                    callbacks=cp_callback_list)

# 保存h5模型
H5_MODEL_PATH = "animal.h5"
model.save(filepath=H5_MODEL_PATH)
print('saved h5 model!')

model.summary()  # 打印出神经网络的结构和参数

# 显示训练集和验证集的acc和loss曲线  数据可视化：
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)  # 一行两列图像显示数据
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("val_loss2.6.png")  # 保存训练时数据图片
plt.show()
