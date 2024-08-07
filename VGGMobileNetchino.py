import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import math
from sklearn.model_selection import train_test_split


def audio_to_mfcc(audio_path, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def save_mfcc_image(mfcc, output_path):
    plt.figure(figsize=(1, 1))
    plt.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
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
                        output_path = os.path.join(
                            label_output_dir, file.replace('.wav', '.png'))
                        save_mfcc_image(mfcc, output_path)
                        image = img_to_array(
                            load_img(output_path, target_size=(128, 128)))
                        images.append(image)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
    return np.array(images), np.array(labels)


def preprocess_data(images, labels):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    images = images.astype('float32') / 255.0  # Normalize the images
    return images, labels, lb


def create_datagen():
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def create_mobilenet_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Fine-tune only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


def create_vgg_model(num_classes):
    base_model = VGG16(weights='imagenet',
                       include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Fine-tune only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


def compile_and_train_model(model, train_generator, validation_generator, learning_rate, epochs, steps_per_epoch, validation_steps, model_path='model.keras'):
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator.repeat(),  # Ensure the generator repeats
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator.repeat(),  # Ensure the generator repeats
        validation_steps=validation_steps
    )

    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    return model, history


def plot_comparison(history1, history2, title, metric='accuracy'):
    plt.plot(history1.history[metric], label='MobileNetV2')
    plt.plot(history1.history[f'val_{metric}'], label='MobileNetV2 Val')
    plt.plot(history2.history[metric], label='VGG16')
    plt.plot(history2.history[f'val_{metric}'], label='VGG16 Val')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'chinotrain')
    output_dir = os.path.join(current_dir, 'VGG_MobileNet_mfcc_images')

    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    create_output_dir(output_dir)

    # Load and preprocess data
    images, labels = load_data(input_dir, output_dir)
    if images.size == 0 or labels.size == 0:
        print("No data loaded. Please check the input directory and audio files.")
        return

    images, labels, lb = preprocess_data(images, labels)

    # Log data size
    logging.info(f"Loaded {len(images)} images with {len(labels)} labels.")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    # Training parameters
    learning_rate = 0.0001  # Adjusted learning rate
    epochs = 100
    batch_size = 2048  # Reduce batch size to better handle small datasets
    steps_per_epoch = max(1, len(X_train) // batch_size)
    validation_steps = max(1, len(X_val) // batch_size)

    # Data augmentation and generator creation
    datagen = create_datagen()
    train_generator = datagen.flow(
        X_train, y_train, batch_size=batch_size, shuffle=True)
    validation_generator = datagen.flow(
        X_val, y_val, batch_size=batch_size, shuffle=False)

    # Train MobileNetV2 model
    mobilenet_model = create_mobilenet_model(len(lb.classes_))
    mobilenet_model, mobilenet_history = compile_and_train_model(
        mobilenet_model,
        train_generator,
        validation_generator,
        learning_rate,
        epochs,
        steps_per_epoch,
        validation_steps,
        'MobileNetV2_model.keras'
    )

    # Train VGG16 model
    vgg_model = create_vgg_model(len(lb.classes_))
    vgg_model, vgg_history = compile_and_train_model(
        vgg_model,
        train_generator,
        validation_generator,
        learning_rate,
        epochs,
        steps_per_epoch,
        validation_steps,
        'VGG16_model.keras'
    )

    # Compare Models
    plot_comparison(mobilenet_history, vgg_history,
                    'Model Accuracy Comparison')
    plot_comparison(mobilenet_history, vgg_history,
                    'Model Loss Comparison', metric='loss')


if __name__ == "__main__":
    main()