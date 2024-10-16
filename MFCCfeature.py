import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import stft
from scipy.fftpack import dct
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=44100)#波型資料，採樣率
    return audio, sr
def AveMag(y,sr):#音量強度曲線
    # 计算平均振幅
    hop_length = 86#重疊部分
    frame_length = 256
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    D=librosa.stft(y)
    magnitude,phase = librosa.magphase(D)
    ave_mag=np.mean(magnitude,axis=0)
    plt.plot(ave_mag,marker='o')
    plt.title('Average magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Average magnitude')
    plt.tight_layout()
    plt.show()
    return ave_mag
def extract_pitch(y, sr):#基頻軌跡(pitch tracking)
    S = np.abs(librosa.stft(y))
    pitches, magnitudes = librosa.core.piptrack(S=S, sr=sr,n_fft=2048, hop_length=84, fmin=20.0, fmax=3000.0, threshold=0.1, win_length=None, window='hann', center=True, pad_mode='constant', ref=np.mean)
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_value = pitches[index, t]
        if pitch_value > 0:
            pitch.append(pitch_value)
    return np.array(pitch)
def Interpolation(ave_mag, target_length):
    x_old = np.linspace(0, len(ave_mag) - 1, len(ave_mag))
    x_new = np.linspace(0, len(ave_mag) - 1, target_length)
    ave_mag_new = np.interp(x_new, x_old, ave_mag)
    return ave_mag_new
def linear_scaling(ave_mag1, ave_mag2):
    # 创建矩阵A和向量y
    N = len(ave_mag1)
    A = np.vstack([ave_mag2, np.ones(N)]).T
    y = ave_mag1

    # 使用最小二乘法求解theta
    theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    # 调整后的音量强度曲线
    ave_mag2_adjusted = ave_mag2 * theta[0] + theta[1]
    return ave_mag2_adjusted, ave_mag1
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
def extract_mfcc(y, sr):
    # 計算STFT
    n_fft = 2048  # STFT的窗口大小
    hop_length = 512  # 跳步大小
    f, t, Zxx = stft(y, sr, nperseg=n_fft, noverlap=hop_length)
    S = np.abs(Zxx)**2

    # 計算Mel濾波器頻譜
    n_mels = 40  # Mel濾波器的數量
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048)
    mel_spectrogram = np.dot(mel_basis, S)

    # 對Mel頻譜進行對數轉
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    # 計算DCT以得到MFCC（DCT-II型）
    n_mfcc = 39  # 希望的MFCC維度
    mfcc = dct(log_mel_spectrogram, type=2, axis=0, norm='ortho')[:n_mfcc]

    # 繪製每一個MFCC維度的特徵
    for i in range(n_mfcc):
        plt.figure(figsize=(10, 2))
        plt.plot(mfcc[i], label=f'MFCC Coefficient {i+1}')
        plt.title(f'MFCC Coefficient {i+1} Over Time')
        plt.xlabel('Time')
        plt.ylabel(f'Coefficient {i+1} Value')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return mfcc
def cepstral_mean_subtraction(mfccs):
    mean = np.mean(mfccs, axis=1, keepdims=True)
    cms_mfccs = mfccs - mean
    return cms_mfccs
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path_A=os.path.join(current_dir,'static','audio','E.wav')
    audio_A, sr_A = load_audio(audio_file_path_A)
    mfccs_A = extract_mfcc(audio_A, sr_A)
    # mfccs_A_cms = cepstral_mean_subtraction(mfccs_A)
    # # 繪製MFCC熱力圖
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mfccs_A_cms, x_axis='time', sr=sr_A, hop_length=512, cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('MFCC Heatmap')
    # plt.xlabel('Time')
    # plt.ylabel('MFCC Coefficients')
    # plt.tight_layout()
    # plt.show()