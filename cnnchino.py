import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Bidirectional, Input, Reshape
from tensorflow.keras.utils import to_categorical

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'chinotrain')  # Update to new directory
test_file_path = os.path.join(current_dir, 'static', 'audio', 'user_input.wav')

# Function for Cepstral Mean Subtraction
def cepstral_mean_subtraction(mfccs):
    mean = np.mean(mfccs, axis=1, keepdims=True)
    cms_mfccs = mfccs - mean
    return cms_mfccs

# Function to extract features directly from audio samples
def extract_features(y, sr, n_mfcc=13, n_chroma=12, n_contrast=7):
    # Choose n_fft based on signal length
    n_fft = min(len(y), 1024)  # Use 1024 or the length of the signal, whichever is smaller

    # MFCC with CMS
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    mfcc_cms = cepstral_mean_subtraction(mfcc)
    mfcc_mean = np.mean(mfcc_cms, axis=1)

    # Delta and Delta-Delta Features
    delta_mfcc = librosa.feature.delta(mfcc_cms)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

    delta2_mfcc = librosa.feature.delta(mfcc_cms, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
    contrast_mean = np.mean(contrast, axis=1)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)

    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff, axis=1)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth, axis=1)

    # Combine all features
    features = np.concatenate((mfcc_mean, delta_mfcc_mean, delta2_mfcc_mean, chroma_mean, 
                               contrast_mean, tonnetz_mean, rolloff_mean, bandwidth_mean))

    return features

# Data Augmentation function
def augment_audio(y, sr):
    # Time Stretch
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)

    # Pitch Shift
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

    # Adding Noise
    noise = np.random.normal(0, 0.01, len(y))
    y_noise = y + noise

    return [y, y_stretch, y_shift, y_noise]

# Initialize lists to hold features and labels
features = []
labels = []

# Loop through each directory
for folder_num in range(1, 38):  # 1 to 37
    folder_path = os.path.join(data_path, str(folder_num))
    for file in glob.glob(os.path.join(folder_path, '*.wav')):
        y, sr = librosa.load(file, sr=None)
        augmented_audios = augment_audio(y, sr)
        for aug_y in augmented_audios:
            feature_vector = extract_features(aug_y, sr)  # Pass audio data directly
            features.append(feature_vector)
            labels.append(folder_num)

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Reshape data for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Updated CNN Model with Batch Normalization and LSTM
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1, 1)))
model.add(Conv2D(32, kernel_size=(2, 1), activation='relu'))
model.add(BatchNormalization())  # Normalize activations
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(2, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

model.add(Flatten())

# Reshape the flattened output for LSTM
model.add(Reshape((model.layers[-1].output_shape[1], 1)))  # Reshape to (timesteps, features)

# Add Bidirectional LSTM for temporal patterns
model.add(Bidirectional(LSTM(64)))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {score[1] * 100:.2f}%")

# Predictions and Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(classification_report(y_true, y_pred_classes))

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Test new audio files
def predict_audio(file_path, model):
    # Load audio file and extract features
    y, sr = librosa.load(file_path, sr=None)
    features = extract_features(y, sr)  # Pass audio data directly
    features = features.reshape(1, features.shape[0], 1, 1)
    prediction = model.predict(features)
    predicted_class = le.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

# Example: Test a new audio file
predicted_class = predict_audio(test_file_path, model)
print(f"Predicted Class: {predicted_class}")