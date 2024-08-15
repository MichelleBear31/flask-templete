import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization,Layer
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import logging
import pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
#print("Num GPUs Available: ", len(gpus))
tf.config.experimental.set_visible_devices(devices=gpus[0:2],device_type='GPU')
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
                output_path = os.path.join(category_output_dir, f"{os.path.splitext(audio_file)[0]}.pkl")
                if os.path.exists(output_path):
                    logging.info(f"PKL file already exists for {audio_file}, skipping processing.")
                    continue
                
                audio_path = os.path.join(category_path, audio_file)
                mfcc = audio_to_mfcc(audio_path)
                save_mfcc_array(mfcc, output_path)
                logging.info(f"Processed {audio_file} to {output_path}")

def split_dataset(input_dir, output_train_dir, output_test_dir, test_size=0.3):
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
    
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        files = [f for f in os.listdir(category_path) if f.endswith('.pkl')]
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
        
        category_train_dir = os.path.join(output_train_dir, category)
        category_test_dir = os.path.join(output_test_dir, category)
        
        if not os.path.exists(category_train_dir):
            os.makedirs(category_train_dir)
        if not os.path.exists(category_test_dir):
            os.makedirs(category_test_dir)
        
        for file in train_files:
            src = os.path.join(category_path, file)
            dest = os.path.join(category_train_dir, file)
            try:
                os.rename(src, dest)
            except FileExistsError:
                logging.warning(f"File already exists: {dest}, skipping.")
        
        for file in test_files:
            src = os.path.join(category_path, file)
            dest = os.path.join(category_test_dir, file)
            try:
                os.rename(src, dest)
            except FileExistsError:
                logging.warning(f"File already exists: {dest}, skipping.")
        
        logging.info(f"Category {category}: {len(train_files)} files for training, {len(test_files)} files for testing.")

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

@register_keras_serializable(package="Custom", name="Attention")
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[-1],),  # 确保与输入的最后一个维度匹配
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        alpha = tf.keras.backend.softmax(e, axis=1)
        context = tf.keras.backend.sum(alpha * x, axis=1)
        return context

def create_model_with_attention(num_classes=2):
    inputs = tf.keras.Input(shape=(87, 20))  # 创建输入层
    x = LSTM(128, return_sequences=True)(inputs)  # 确保提供了 units 参数
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=True)(x)  # 确保提供了 units 参数
    x = Attention()(x)  # 添加注意力层
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # 输出层
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)  # 创建模型
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_model_with_attention(train_dir, model_path='LSTM_model_with_attention.keras', history_path='LSTM_history_with_attention.pkl', epochs=2000, batch_size=128):
    X, y, categories = load_data(train_dir)
    
    model = create_model_with_attention(num_classes=len(categories))
    
    # 在每个 epoch 结束时保存激活值
    class SaveActivations(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # 使用模型获取中间层的输出
            intermediate_model = tf.keras.Model(inputs=model.input,
                                                outputs=model.get_layer('attention').output)
            activations = intermediate_model.predict(X)
            np.save(f'activations_epoch_{epoch}.npy', activations)
            logging.info(f"Saved activations for epoch {epoch}")

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            SaveActivations(),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
        ]
    )
    
    model.save(model_path)
    logging.info(f"Model with attention saved to {model_path}")
    
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    logging.info(f"Training history with attention saved to {history_path}")

    return history, categories

def create_model(num_classes=3):
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, 20), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Use softmax for multi-class classification
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir, model_path='LSTM_model.keras', history_path='LSTM_history.pkl', epochs=2000, batch_size=128):
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
    
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    logging.info(f"Training history saved to {history_path}")

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

def visual_speature():
        # 加载某个 epoch 的激活值
    activations = np.load('activations_epoch_499.npy')
    # 可视化激活值
    plt.imshow(activations.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Attention Layer Activations')
    plt.xlabel('Time Steps')
    plt.ylabel('Attention Units')
    plt.show()

def main():
    # 音訊轉MFCC特徵
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'chinotrain')
    output_mfcc_dir = os.path.join(current_dir, 'LSTM_mfcc_arrays')
    
    model_path = os.path.join(current_dir, 'LSTM_model_with_attention.keras')
    history_path = os.path.join(current_dir, 'LSTM_history_with_attention.pkl')
    
    if os.path.exists(model_path) or os.path.exists(history_path):
        retrain = input("Model or history files already exist. Do you want to retrain? (y/n): ").strip().lower()
        
        if retrain == 'y':
            logging.info("Retraining the model...")
            process_audio_files(input_dir, output_mfcc_dir)
            
            # 分割資料集
            output_train_dir = os.path.join(current_dir, 'LSTM_mfcc_train')
            output_test_dir = os.path.join(current_dir, 'LSTM_mfcc_test')
            split_dataset(output_mfcc_dir, output_train_dir, output_test_dir, test_size=0.3)
            
            try:
                history, categories = train_model_with_attention(output_train_dir, model_path=model_path, history_path=history_path, epochs=600, batch_size=128)
                logging.info("Training completed successfully")
            except Exception as e:
                logging.error(f"An error occurred during training: {str(e)}")
                return
        else:
            logging.info("Skipping retraining...")
            try:
                with open(history_path, 'rb') as f:
                    history = pickle.load(f)
                categories = sorted(os.listdir(os.path.join(current_dir, 'LSTM_mfcc_train')))
                logging.info("Model and history loaded successfully")
            except Exception as e:
                logging.error(f"An error occurred while loading model or history: {str(e)}")
                return
    else:
        logging.info("No existing model or history found, proceeding with training...")
        process_audio_files(input_dir, output_mfcc_dir)
        
        # 分割資料集
        output_train_dir = os.path.join(current_dir, 'LSTM_mfcc_train')
        output_test_dir = os.path.join(current_dir, 'LSTM_mfcc_test')
        split_dataset(output_mfcc_dir, output_train_dir, output_test_dir, test_size=0.3)
        
        try:
            history, categories = train_model_with_attention(output_train_dir, model_path=model_path, history_path=history_path, epochs=650, batch_size=128)
            logging.info("Training completed successfully")
        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            return

    test_audio_path = os.path.join(current_dir, 'static', 'audio', 'test1.wav')
    try:
        predicted_class, confidence = process_test_audio(test_audio_path, model_path, categories)
        logging.info(f"Test audio predicted class: {predicted_class}")
        logging.info(f"Test audio confidence level: {confidence * 100:.2f}%")
    except Exception as e:
        logging.error(f"An error occurred during prediction for test audio: {str(e)}")

if __name__ == "__main__":

    visual_speature()

    main()