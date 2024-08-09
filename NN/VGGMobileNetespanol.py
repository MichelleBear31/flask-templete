import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def audio_to_mfcc(audio_path, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def save_mfcc_image(mfcc, output_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc, aspect='auto', origin='lower')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
def load_data(input_dir, output_dir):
    images = []
    labels = []
    for label in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, label)
        if os.path.isdir(class_dir):
            label_output_dir = os.path.join(output_dir, label)
            create_output_dir(label_output_dir)
            for file in os.listdir(class_dir):
                audio_path = os.path.join(class_dir, file)
                if audio_path.endswith('.wav'):
                    try:
                        mfcc = audio_to_mfcc(audio_path)
                        output_path = os.path.join(label_output_dir, file.replace('.wav', '.png'))
                        save_mfcc_image(mfcc, output_path)
                        image = img_to_array(load_img(output_path))
                        images.append(image)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    return images, labels, lb

def create_datagen():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

def create_model(base_model, num_classes):
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

def compile_and_train_model(model, train_generator, learning_rate, epochs, steps_per_epoch):
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)
    return model

def save_model(model, filename):
    model.save(filename)

def evaluate_model(model, test_generator, steps):
    loss, accuracy = model.evaluate(test_generator, steps=steps)
    return loss, accuracy
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, 'train')
output_dir = os.path.join(current_dir, 'VGG_MobileNet_mfcc_images')

def main():
    logging.basicConfig(level=logging.INFO)
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    create_output_dir(output_dir)
    
    # 加载和预处理数据
    images, labels = load_data(input_dir, output_dir)
    if images.size == 0 or labels.size == 0:
        print("No data loaded. Please check the input directory and audio files.")
        return
    
    images, labels, lb = preprocess_data(images, labels)
    
    learning_rate = 0.0001
    epochs = 5
    batch_size = 1
    steps_per_epoch = 47 * 3  # 47個資料夾，每個資料夾3個音檔
    
    # 加載和預處理數據
    images, labels = load_data(input_dir, output_dir)
    if images.size == 0 or labels.size == 0:
        print("No data loaded. Please check the input directory and audio files.")
        return
    
    images, labels, lb = preprocess_data(images, labels)
    
    datagen = create_datagen()
    train_generator = datagen.flow(images, labels, batch_size=batch_size)

    # 訓練VGG16模型
    base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=images.shape[1:])
    model_vgg = create_model(base_model_vgg, len(lb.classes_))
    model_vgg = compile_and_train_model(model_vgg, train_generator, learning_rate, epochs, steps_per_epoch)
    save_model(model_vgg, 'VGG16.keras')

    # 訓練MobileNet模型
    base_model_mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=images.shape[1:])
    model_mobilenet = create_model(base_model_mobilenet, len(lb.classes_))
    model_mobilenet = compile_and_train_model(model_mobilenet, train_generator, learning_rate, epochs, steps_per_epoch)
    save_model(model_mobilenet, 'MobileNet.keras')

    # 評估模型
    vgg_loss, vgg_accuracy = evaluate_model(model_vgg, train_generator, steps_per_epoch)
    print(f'VGG16 - Loss: {vgg_loss}, Accuracy: {vgg_accuracy}')

    mobilenet_loss, mobilenet_accuracy = evaluate_model(model_mobilenet, train_generator, steps_per_epoch)
    print(f'MobileNet - Loss: {mobilenet_loss}, Accuracy: {mobilenet_accuracy}')

if __name__ == "__main__":
    main()