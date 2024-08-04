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

def audio_to_mfcc(audio_path, sr=44100,max_len=87):
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
    
    for audio_file in os.listdir(input_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(input_dir, audio_file)
            mfcc = audio_to_mfcc(audio_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.pkl")
            save_mfcc_array(mfcc, output_path)
            logging.info(f"Processed {audio_file} to {output_path}")

def create_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, 20), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data(data_dir, max_len=87):  # 更新 max_len 為 87
    X, y = [], []
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(data_dir, file), 'rb') as f:
                mfcc = pickle.load(f)
                if mfcc.shape[0] == max_len:  # 確保 MFCC 特徵長度為 87
                    X.append(mfcc)
                    y.append(1 if 'positive' in file else 0)
                else:
                    print(f"Skipping {file} due to incorrect shape: {mfcc.shape}")
    X = np.array(X)
    y = np.array(y)
    print(f"Final data shapes: X={X.shape}, y={y.shape}")
    return X, y

def train_model(train_dir, model_path='LSTM_model.keras', epochs=2, batch_size=32):
    X, y = load_data(train_dir)
    
    model = create_model()
    
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )

    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    return history

def process_test_audio(audio_path, model_path='LSTM_model.keras'):
    mfcc = audio_to_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    
    model = load_model(model_path)
    
    prediction = model.predict(mfcc)
    
    return prediction[0][0]

def main():
    #音訊轉MFCC特徵
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'train')
    output_dir = os.path.join(current_dir, 'LSTM_mfcc_arrays')
    
    process_audio_files(input_dir, output_dir)

    # 訓練階段
    try:
        history = train_model(output_dir, epochs=2, batch_size=32)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return

    test_audio_path = os.path.join('static','audio','user_input.wav')
    model_path = 'LSTM_model.keras'

    try:
        result = process_test_audio(test_audio_path, model_path)
        logging.info(f"Test audio raw prediction: {result}")
        logging.info(f"Test audio prediction result: {result * 100:.2f}%")
    except Exception as e:
        logging.error(f"An error occurred during prediction for test audio: {str(e)}")

if __name__ == "__main__":
    main()