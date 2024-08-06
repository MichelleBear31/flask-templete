import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO)

def audio_to_mfcc(audio_path, sr=44100, min_length=22050):
    """
    Convert audio file to MFCC with error handling for short files.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) < min_length:
            raise ValueError(f"Audio file {audio_path} is too short for processing.")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return mfcc
    except Exception as e:
        logging.error(f"Error processing audio file {audio_path}: {str(e)}")
        return None

def save_mfcc_image(mfcc, output_path):
    """
    Save MFCC as an image.
    """
    if mfcc is None:
        logging.warning(f"Skipping saving MFCC image for {output_path} due to previous errors.")
        return

    plt.figure(figsize=(10, 10))
    librosa.display.specshow(mfcc, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_audio_files(input_dir, output_dir):
    """
    Process audio files and save MFCC images, with error handling.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for category in os.listdir(input_dir):
        try:
            category_int = int(category)
            if category_int < 1 or category_int > 37:
                raise ValueError(f"Invalid category number: {category}")
        except ValueError as e:
            logging.warning(f"Ignoring invalid directory: {category}")
            continue

        category_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)

        for audio_file in os.listdir(category_dir):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(category_dir, audio_file)
                try:
                    mfcc = audio_to_mfcc(audio_path)
                    if mfcc is not None:
                        output_path = os.path.join(category_output_dir, f"{os.path.splitext(audio_file)[0]}.png")
                        save_mfcc_image(mfcc, output_path)
                        logging.info(f"Processed {audio_file} to {output_path}")
                    else:
                        logging.warning(f"Skipping {audio_file} due to MFCC processing error.")
                except Exception as e:
                    logging.error(f"Failed to process {audio_file}: {str(e)}")

def create_model(model_path='cnn.keras'):
    """
    Create and compile the CNN model using Input layer.
    """
    num_classes = 37
    model = Sequential()
    model.add(Input(shape=(128, 128, 3)))  # Use Input layer for the first layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    return model

def train_model(train_dir, model_path='cnn.keras', epochs=None, batch_size=None):
    """
    Train the model with data augmentation and error handling.
    """
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
    
    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=batch_size or 32,  # 如果batch_size为None，使用默认值32
            class_mode='sparse',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=batch_size or 32,  # 如果batch_size为None，使用默认值32
            class_mode='sparse',
            subset='validation',
            shuffle=True
        )
    except Exception as e:
        logging.error(f"Error initializing data generators: {str(e)}")
        return None
    
    model = create_model(model_path)
    
    # 将生成器转换为tf.data.Dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).repeat()

    validation_dataset = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).repeat()
    
    # 计算实际的步数
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    
    try:
        history = model.fit(
            train_dataset,
            epochs=epochs or 100,  # 如果epochs为None，使用默认值100
            validation_data=validation_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
                tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
        logging.info(f"Model saved to {model_path}")
        return history
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return None

def process_test_audio(audio_path, model_path='cnn.keras'):
    """
    Process a test audio file and make predictions.
    """
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
    
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    return predicted_class, confidence

def main():
    """
    Main function to process audio and train the model.
    """
    # 音訊轉MFCC圖片
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'chinotrain')
    output_dir = os.path.join(current_dir, 'cnn_mfcc_images')
    
    process_audio_files(input_dir, output_dir)

    # 訓練階段
    mfcc_images_dir = output_dir
    
    if not os.path.exists(mfcc_images_dir):
        logging.error(f"Directory not found: {mfcc_images_dir}")
        return

    test_audio_path = os.path.join(current_dir, 'static', 'audio', 'user_input.wav')
    model_path = os.path.join(current_dir, 'cnn.keras')

    try:
        history = train_model(mfcc_images_dir, model_path=model_path, epochs=100, batch_size=2048)
        if history is not None:
            logging.info("Training completed successfully")
            if os.path.exists(model_path):
                logging.info(f"模型文件已成功创建: {model_path}")
            else:
                logging.error(f"模型文件未能创建: {model_path}")
        else:
            logging.error("Training failed.")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return

    try:
        predicted_class, confidence = process_test_audio(test_audio_path, model_path)
        logging.info(f"Test audio prediction result: Class {predicted_class} with {confidence * 100:.2f}% confidence")
        
        model = load_model(model_path)
        train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            mfcc_images_dir,
            target_size=(128, 128),
            batch_size=2048,
            class_mode='sparse'
        )
        train_predictions = model.predict(train_generator)
        logging.info(f"Training set predictions: Min={np.min(train_predictions):.4f}, Max={np.max(train_predictions):.4f}, Mean={np.mean(train_predictions):.4f}")
    except Exception as e:
        logging.error(f"An error occurred during prediction for test audio: {str(e)}")

if __name__ == "__main__":
    main()