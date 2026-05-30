import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


CLASSES = ("bird", "butterfly", "cat", "dog", "tiger")
IMAGE_SIZE = (224, 224)
MODEL_PATH = "animal.h5"


def resize_img(img, size=IMAGE_SIZE):
    """按比例缩放图片，并用白色背景补齐到模型需要的尺寸。"""
    iw, ih = img.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    img = img.resize((nw, nh), Image.BICUBIC)
    new_img = Image.new("RGB", size, (255, 255, 255))
    new_img.paste(img, ((w - nw) // 2, (h - nh) // 2))
    return new_img


def build_model(weights_path=MODEL_PATH):
    """重新搭建 TF2.2 兼容的网络结构，再从 animal.h5 加载权重。"""
    # animal.h5 是较新 Keras 保存的完整模型，TF2.2 直接 load_model 会不兼容；
    # 这里用同样的 MobileNetV2 分类结构加载权重，可以继续复用 tf22 的 GPU 环境。
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights=None,
        pooling="avg",
    )
    model = tf.keras.Sequential(
        [
            mobilenet,
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.load_weights(weights_path)
    return model


def preprocess_image(image_path):
    # CLI 和 GUI 都走同一套预处理，避免两种入口预测结果不一致。
    with Image.open(image_path) as img:
        img = resize_img(img.convert("RGB"), IMAGE_SIZE)
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    return img_arr[tf.newaxis, ...]


def predict_image(model, image_path):
    # 返回类别名、最高置信度，以及完整的 softmax 输出。
    result = model.predict(preprocess_image(image_path))
    class_index = int(tf.argmax(result, axis=1)[0])
    score = float(np.max(result[0]))
    return CLASSES[class_index], score, result


def draw_prediction(image_path, label, score):
    # GUI 展示用：把预测类别和置信度画到图片左上角。
    with Image.open(image_path) as img:
        display_img = resize_img(img.convert("RGB"), (400, 400))
    display_arr = cv2.cvtColor(np.asarray(display_img), cv2.COLOR_RGB2BGR)
    cv2.putText(
        display_arr,
        "{:.2f}%".format(score * 100),
        (8, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        display_arr,
        label,
        (8, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return cv2.cvtColor(display_arr, cv2.COLOR_BGR2BGRA)


class Ui_MainWindow(object):
    def __init__(self, model, parent=None):
        # PyQt5 只在启动 GUI 时导入，这样命令行预测不依赖图形界面初始化。
        from PyQt5 import QtCore, QtWidgets

        super(Ui_MainWindow, self).__init__()
        self.model = model
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.widget = QtWidgets.QWidget(parent)
        self.timer_camera = QtCore.QTimer()
        self.timer_camera_capture = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()

    def set_ui(self):
        QtWidgets = self.QtWidgets
        layout_main = QtWidgets.QHBoxLayout()
        layout_fun_button = QtWidgets.QVBoxLayout()

        self.pushButton = QtWidgets.QPushButton("打开图片")
        self.pushButton.setMinimumHeight(50)

        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setMinimumHeight(50)
        self.lineEdit.setFixedSize(140, 30)

        self.label = QtWidgets.QLabel()
        self.label.setFixedSize(641, 481)
        self.label.setAutoFillBackground(False)

        layout_fun_button.addWidget(self.pushButton)
        layout_fun_button.addWidget(self.lineEdit)
        layout_fun_button.addStretch(1)

        layout_main.addLayout(layout_fun_button)
        layout_main.addWidget(self.label)

        self.widget.setLayout(layout_main)
        self.widget.setWindowTitle("动物识别")

    def slot_init(self):
        self.pushButton.clicked.connect(self.button_open_image_click)

    def button_open_image_click(self):
        from PyQt5 import QtGui, QtWidgets

        # 用户选择图片后，调用和 CLI 相同的 predict_image 函数。
        self.label.clear()
        self.lineEdit.clear()
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.widget,
            "打开图片",
            "",
            "Images (*.jpg *.jpeg *.png);;All Files (*)",
        )
        if not image_path:
            return

        label, score, result = predict_image(self.model, os.path.expanduser(image_path))
        print("image:", image_path)
        print("scores:", result)
        print("result:", label, "{:.2f}%".format(score * 100))

        image_bgra = draw_prediction(image_path, label, score)
        qt_img = QtGui.QImage(
            image_bgra.data,
            image_bgra.shape[1],
            image_bgra.shape[0],
            QtGui.QImage.Format_RGB32,
        )
        self.label.setPixmap(QtGui.QPixmap.fromImage(qt_img))
        self.lineEdit.setText(label)

    def show(self):
        self.widget.show()


def run_gui(model):
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow(model)
    ui.show()
    return app.exec_()


def parse_args():
    parser = argparse.ArgumentParser(description="Animal image classifier")
    parser.add_argument("--image", help="Path to an image for command-line prediction.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to animal.h5 weights.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("labels:", CLASSES)
    model = build_model(args.model)
    if args.image:
        label, score, _ = predict_image(model, args.image)
        print("image:", args.image)
        print("result:", label)
        print("confidence: {:.2f}%".format(score * 100))
        return 0
    return run_gui(model)


if __name__ == "__main__":
    sys.exit(main())
