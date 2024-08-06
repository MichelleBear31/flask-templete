import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import logging
import pickle

logging.basicConfig(level=logging.INFO)

def audio_to_mfcc(audio_path, sr=44100, max_len=87):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mean = np.mean(mfcc, axis=1, keepdims=True)
    cms_mfccs = mfcc - mean
    return cms_mfccs.T  # Transpose to shape (time, features)

def save_mfcc_array(mfcc, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(mfcc, f)

def process_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        category_output_dir = os.path.join(output_dir, category)
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)
        
        for audio_file in os.listdir(category_path):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(category_path, audio_file)
                mfcc = audio_to_mfcc(audio_path)
                output_path = os.path.join(category_output_dir, f"{os.path.splitext(audio_file)[0]}.pkl")
                save_mfcc_array(mfcc, output_path)
                logging.info(f"Processed {audio_file} to {output_path}")

def load_data(data_dir, max_len=87):
    X, y = [], []
    categories = sorted(os.listdir(data_dir))
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        for file in os.listdir(category_path):
            if file.endswith('.pkl'):
                with open(os.path.join(category_path, file), 'rb') as f:
                    mfcc = pickle.load(f)
                    if mfcc.shape[0] == max_len:
                        X.append(mfcc)
                        y.append(idx)  # Use index of category as label
                    else:
                        logging.warning(f"Skipping {file} due to incorrect shape: {mfcc.shape}")
    
    X = np.array(X)
    y = np.array(y)
    logging.info(f"Final data shapes: X={X.shape}, y={y.shape}")
    return X, y, categories
def create_model(num_classes=37):
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, 20), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Use softmax for multi-class classification
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir, model_path='LSTM_model.keras', epochs=20, batch_size=64):
    X, y, categories = load_data(train_dir)
    
    model = create_model(num_classes=len(categories))
    
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        ]
    )

    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    return history, categories

def process_test_audio(audio_path, model_path='LSTM_model.keras', categories=None):
    mfcc = audio_to_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    
    model = load_model(model_path)
    
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    confidence = prediction[0][predicted_class]
    
    category_name = categories[predicted_class] if categories else predicted_class
    
    return category_name, confidence

def main():
    # 音訊轉MFCC特徵
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'chinotrain')  # Update to new directory
    output_dir = os.path.join(current_dir, 'LSTM_mfcc_arrays')
    
    process_audio_files(input_dir, output_dir)

    # 訓練階段
    try:
        # Define the model path in the current directory
        model_path = os.path.join(current_dir, 'LSTM_model.keras')
        history, categories = train_model(output_dir, model_path=model_path,epochs=1020, batch_size=64)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return

    test_audio_path = os.path.join(current_dir,'static', 'audio', 'user_input.wav')
    model_path = os.path.join(current_dir, 'LSTM_model.keras')
    try:
        predicted_class, confidence = process_test_audio(test_audio_path, model_path, categories)
        logging.info(f"Test audio predicted class: {predicted_class}")
        logging.info(f"Test audio confidence level: {confidence * 100:.2f}%")
    except Exception as e:
        logging.error(f"An error occurred during prediction for test audio: {str(e)}")

if __name__ == "__main__":
    main()