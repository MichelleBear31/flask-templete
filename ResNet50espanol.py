import os
import librosa
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
    #mfcc=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)    
    return mfcc

def save_mfcc_image(mfcc, output_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis=None, y_axis=None) # 移除座標
    plt.axis('off')  # 移除座標軸
    plt.gca().set_position([0, 0, 1, 1])  # 擴展圖片填滿整個畫布
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for audio_file in os.listdir(input_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(input_dir, audio_file)
            mfcc = audio_to_mfcc(audio_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.png")
            save_mfcc_image(mfcc, output_path)
            logging.info(f"Processed {audio_file} to {output_path}")
    
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    for layer in base_model.layers:
        layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir, model_path='ResNet50.keras', epochs=None, batch_size=None):
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
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
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
    
    return prediction[0][0]

def main():
    #音訊轉MFCC圖片
    input_dir = os.path.join(current_dir, 'train')
    output_dir = os.path.join(current_dir, 'Resnet_mfcc_arrays')
    
    process_audio_files(input_dir, output_dir)

    # 訓練階段

    mfcc_images_dir = os.path.join(current_dir,'Resnet_mfcc_images')
    
    if not os.path.exists(mfcc_images_dir):
        logging.error(f"Directory not found: {mfcc_images_dir}")
        return

    image_files = [f for f in os.listdir(mfcc_images_dir) if f.endswith('.png')]
    logging.info(f"Found {len(image_files)} images")

    class_dir = os.path.join(mfcc_images_dir, 'class')
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    for image_file in image_files:
        src = os.path.join(mfcc_images_dir, image_file)
        dst = os.path.join(class_dir, image_file)
        if not os.path.exists(dst):
            os.rename(src, dst)

    try:
        history = train_model(mfcc_images_dir, epochs=3, batch_size=262144)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return

    test_audio_path = os.path.join('static','audio','user_input.wav')
    model_path = 'ResNet50.keras'

    try:
        result = process_test_audio(test_audio_path, model_path)
        logging.info(f"Test audio raw prediction: {result}")
        logging.info(f"Test audio prediction result: {result * 100:.2f}%")
        model = load_model(model_path)
        train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            mfcc_images_dir,
            target_size=(128, 128),
            batch_size=262144,
            class_mode='binary'
        )
        train_predictions = model.predict(train_generator)
        logging.info(f"Training set predictions: Min={np.min(train_predictions):.4f}, Max={np.max(train_predictions):.4f}, Mean={np.mean(train_predictions):.4f}")
    except Exception as e:
        logging.error(f"An error occurred during prediction for test audio: {str(e)}")
if __name__ == "__main__":
    main()