import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from torch.nn import SELU

# 定義讀取影像的函數
def read_images_from_folder(folder, img_size=(100, 100)):
    imgs, labels = [], []
    for label in range(1, 33):  # 根據需求設置範圍
        label_folder = os.path.join(folder, str(label))
        for filename in os.listdir(label_folder):
            if filename.endswith('.png'):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 改為讀取彩色圖像
                img = cv2.resize(img, img_size)
                imgs.append(img)
                labels.append(label - 1)  # 將標籤從1-32轉為0-31
    return np.array(imgs), np.array(labels)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'wave_foto')
model_weights_path = 'cnnwavefoto.h5'

# 檢查是否已經存在模型文件
if os.path.exists(model_weights_path):
    # 如果模型文件存在，手動構建模型架構
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='selu', input_shape=(100, 100, 3)))  # 改為3通道輸入
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='softmax'))

    # 加載模型權重
    model.load_weights(model_weights_path)
    print("Loaded model weights from disk")

    # 編譯模型
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
else:
    # 從資料夾中讀取影像和標籤
    images, labels = read_images_from_folder(data_path)

    # 重新塑形影像數據以適應卷積神經網絡的輸入格式
    images = images.reshape((images.shape[0], 100, 100, 3))  # 改為彩色圖像的形狀 (100, 100, 3)

    # 分割數據集為訓練集和測試集，比例為7:3
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=42)

    # 打印數據集形狀以確認正確性
    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)

    # 構建卷積神經網絡模型
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='selu', input_shape=(100, 100, 3)))  # 改為3通道輸入
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='selu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='softmax'))

    # 編譯模型
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    # 訓練模型
    train_history = model.fit(train_images, train_labels, epochs=1000, batch_size=2048, validation_data=(test_images, test_labels))

    # 保存模型權重
    model.save_weights(model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

    # 可視化訓練過程
    plt.subplot(1, 2, 1)
    plt.title("loss/epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_history.epoch, train_history.history['loss'], linestyle="--", marker='o')
    for a, b in zip(train_history.epoch, train_history.history['loss']):
        plt.text(a, b + 0.1, round(b, 2), ha='center', va='bottom', fontsize=10)

    # 可視化訓練準確度
    plt.subplot(1, 2, 2)
    plt.title("Train Accuracy / Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.plot(train_history.epoch, train_history.history['sparse_categorical_accuracy'], linestyle="--", marker='o')
    for a, b in zip(train_history.epoch, train_history.history['sparse_categorical_accuracy']):
        plt.text(a, b + 0.003, round(b, 2), ha='center', va='bottom', fontsize=10)
    plt.show()

# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.2f}")