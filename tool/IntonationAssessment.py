import os
import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def remove_silence(audio, threshold_low=30):
    non_silent_intervals = librosa.effects.split(audio, top_db=threshold_low)
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return non_silent_audio
def extract_pitch(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)
    Ry=remove_silence(y)
    # Extract the pitch (fundamental frequency)
    pitches, magnitudes = librosa.core.piptrack(y=Ry, sr=sr)
    
    # Select the highest magnitude pitch for each frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    return pitch_values

def Interpolation(ave_mag, target_length):
    x_old = np.linspace(0, len(ave_mag) - 1, len(ave_mag))
    x_new = np.linspace(0, len(ave_mag) - 1, target_length)
    ave_mag_new = np.interp(x_new, x_old, ave_mag)
    return ave_mag_new

def normalize_pitch_curves(standard_pitch, test_pitch):
    target_length = max(len(standard_pitch), len(test_pitch))
    standard_pitch_normalized = Interpolation(standard_pitch, target_length)
    test_pitch_normalized = Interpolation(test_pitch, target_length)
    return standard_pitch_normalized, test_pitch_normalized

def linear_shifting(pitch_A, pitch_B):
    target_length = max(len(pitch_A), len(pitch_B))
    # Interpolate both pitch tracks to the same length
    pitch_A_interp = Interpolation(pitch_A, target_length)
    pitch_B_interp = Interpolation(pitch_B, target_length)
    # Calculate the mean of both interpolated pitch tracks
    mean_A = np.mean(pitch_A_interp)
    mean_B = np.mean(pitch_B_interp)
    # Apply linear shifting to align the means
    shifted_pitch_A = pitch_A_interp - mean_A + mean_B
    shifted_pitch_B = pitch_B_interp
    return shifted_pitch_A, shifted_pitch_B

def plot_pitch_curves(standard_pitch, test_pitch, score):
    plt.figure(figsize=(10, 4))
    plt.plot(standard_pitch, 'bo-', label='standard.wav')
    plt.plot(test_pitch, 'rs-', label='test.wav')
    plt.title(f'Pitch curves (score={score})')
    plt.xlabel('Frame')
    plt.ylabel('Pitch (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

def dynamic_time_warpingAVG(x,y):
    n = len(x)
    m = len(y)

    # Initialize the cost matrix with infinity
    D = np.full((n, m), np.inf)
    D[0, 0] = 0

    # Compute the cumulative cost matrix
    for i in range(1, n):
        for j in range(1, m):
            cost = abs(x[i] - y[j])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    
    # Traceback from D[n-1, m-1] to find the optimal path
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))
    while i > 0 and j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            direction = np.argmin([D[i-1, j], D[i, j-1], D[i-1, j-1]])
            if direction == 0:
                i -= 1
            elif direction == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    path.reverse()
    
    distance = D[n-1, m-1]
    return distance, path
def calculate_similarity(standard_pitch, test_pitch):
    standard_pitch_normalized, test_pitch_normalized = normalize_pitch_curves(standard_pitch, test_pitch)
    shifted_standard_pitch, shifted_test_pitch = linear_shifting(standard_pitch_normalized, test_pitch_normalized)
    
    # Calculate similarity score using DTW
    distance, path = dynamic_time_warpingAVG(shifted_standard_pitch, shifted_test_pitch)
    # Normalize distance to a score between 0 and 100 using an exponential scale
    normalized_distance = distance / (len(shifted_standard_pitch) * max(max(shifted_standard_pitch), max(shifted_test_pitch)))
    score = 100 * np.exp(-normalized_distance * 3)  # Adjust the scaling factor as needed
    
    # Adjust the score to ensure good test cases are above 80
    if score > 50:
        score = 50 + (score - 50) * 1.2  # Scale scores above 50 to be higher
    
    return min(100, score)  # Ensure the score does not exceed 100
# Example usage 
current_dir = os.path.dirname(os.path.abspath(__file__))
standard_pitch = extract_pitch(os.path.join(current_dir, 'static', 'audio', 'A.wav'))
test_pitch_good = extract_pitch(os.path.join(current_dir, 'train', 'A5.wav'))
test_pitch_bad = extract_pitch(os.path.join(current_dir, 'static', 'audio', 'user_input.wav'))

score_good = calculate_similarity(standard_pitch, test_pitch_good)
score_bad = calculate_similarity(standard_pitch, test_pitch_bad)

plot_pitch_curves(standard_pitch, test_pitch_good, score_good)
plot_pitch_curves(standard_pitch, test_pitch_bad, score_bad)