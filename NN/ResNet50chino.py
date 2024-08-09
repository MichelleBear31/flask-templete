import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import logging

logging.basicConfig(level=logging.INFO)
current_dir = os.path.dirname(os.path.abspath(__file__))

def audio_to_mfcc(audio_path, sr=44100):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return mfcc

def save_mfcc_image(mfcc, output_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis=None, y_axis=None)
    plt.axis('off')  # 移除座標軸
    plt.gca().set_position([0, 0, 1, 1])  # 擴展圖片填滿整個畫布
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            class_output_dir = os.path.join(output_dir, class_dir)
            if not os.path.exists(class_output_dir):
                os.makedirs(class_output_dir)

            for audio_file in os.listdir(class_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(class_path, audio_file)
                    mfcc = audio_to_mfcc(audio_path)
                    output_path = os.path.join(class_output_dir, f"{os.path.splitext(audio_file)[0]}.png")
                    save_mfcc_image(mfcc, output_path)
                    logging.info(f"Processed {audio_file} to {output_path}")

def create_simple_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir, model_path='ResNet50.keras', epochs=10, batch_size=256):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',  # 改為 categorical 以支持多類別
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',  # 改為 categorical 以支持多類別
        subset='validation'
    )

    model = create_simple_model(num_classes=train_generator.num_classes)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('ResNet50.keras', save_best_only=True)
        ]
    )

    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    return history

def process_test_audio(audio_path, model_path='ResNet50.keras'):
    temp_image_path = 'temp_mfcc.png'
    
    mfcc = audio_to_mfcc(audio_path)
    save_mfcc_image(mfcc, temp_image_path)
    
    model = load_model(model_path)
    
    img = load_img(temp_image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    
    os.remove(temp_image_path)
    
    return prediction

def main():
    # 設定音訊與圖片資料夾
    input_dir = os.path.join(current_dir, 'chinotrain')  # 使用 chinotrain 資料夾
    output_dir = os.path.join(current_dir, 'Resnet_mfcc_images')
    
    # 將音訊轉換為 MFCC 圖片
    process_audio_files(input_dir, output_dir)

    # 訓練階段
    mfcc_images_dir = output_dir
    
    if not os.path.exists(mfcc_images_dir):
        logging.error(f"Directory not found: {mfcc_images_dir}")
        return

    # 確認資料夾中圖片結構是否正確
    class_dirs = [d for d in os.listdir(mfcc_images_dir) if os.path.isdir(os.path.join(mfcc_images_dir, d))]
    logging.info(f"Found {len(class_dirs)} classes")

    try:
        history = train_model(mfcc_images_dir, epochs=10, batch_size=256)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return

    test_audio_path = os.path.join(current_dir,'static', 'audio', 'user_input.wav')
    model_path = 'ResNet50.keras'

    try:
        prediction = process_test_audio(test_audio_path, model_path)
        predicted_class = np.argmax(prediction)  # 找到機率最高的類別
        confidence = prediction[0][predicted_class]  # 取得該類別的機率
        logging.info(f"Test audio prediction class: {predicted_class + 1}, confidence: {confidence * 100:.2f}%")  # +1 是因為類別從 1 開始編號
        
        # 驗證訓練集上的結果
        model = load_model(model_path)
        train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            mfcc_images_dir,
            target_size=(128, 128),
            batch_size=256,
            class_mode='categorical'
        )
        train_predictions = model.predict(train_generator)
        logging.info(f"Training set predictions: Min={np.min(train_predictions):.4f}, Max={np.max(train_predictions):.4f}, Mean={np.mean(train_predictions):.4f}")
    except Exception as e:
        logging.error(f"An error occurred during prediction for test audio: {str(e)}")
        
if __name__ == "__main__":
    main()