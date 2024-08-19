from flask import Flask, render_template, request, redirect, url_for,jsonify
import numpy as np
import librosa
import sounddevice as sd
from scipy.ndimage import maximum_filter1d
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import pyaudio
import pylab as pl
import pyloudnorm as pyln
import glob
import tqdm
import wave
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
from scipy.optimize import minimize
from sympy import symbols, Eq, solve
app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
# 載入音檔
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=44100)#波型資料，採樣率
    return audio, sr
def record_audio_to_file(filename, duration=3, channels=1, rate=44100, frames_per_buffer=1):
    """Record user's input audio and save it to the specified file."""
    FORMAT = pyaudio.paInt16
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT, channels=channels, rate=rate, frames_per_buffer=frames_per_buffer, input=True)
    print("Start recording...")

    frames = []
    # Record for the given duration
    for _ in range(0, int(rate / frames_per_buffer * duration)):
        data = stream.read(frames_per_buffer)
        frames.append(data)
    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio recorded and saved to {filename}")
    return filename
def End_pont_detection(y,sr):
    # Zero-mean subtraction
    y = y - np.mean(y)
    # Framing
    frame_size = 256
    hop_length = 128
    frames = librosa.util.frame(y, frame_length=frame_size, hop_length=hop_length)

    # Calculate volume (sum of absolute values)
    volume = np.sum(np.abs(frames), axis=0)

    # Calculate zero crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_size, hop_length=hop_length)[0]

    # Calculate volume thresholds
    volume_th1 = np.max(volume) * 0.1
    volume_th2 = np.median(volume) * 0.1
    #volume_th3 = np.min(volume) * 10
    #volume_th4 = volume[0] * 5

    # Calculate ZCR thresholds
    zcr_th1 = np.max(zcr) * 0.1
    zcr_th2 = np.median(zcr) * 0.1
    #zcr_th3 = np.min(zcr) * 10
    #zcr_th4 = zcr[0] * 5
    # Debug: Print threshold values
    # print(f"Volume thresholds: {volume_th1}, {volume_th2}, {volume_th3}, {volume_th4}")
    # print(f"ZCR thresholds: {zcr_th1}, {zcr_th2}, {zcr_th3}, {zcr_th4}")
    print(f"Volume thresholds: {volume_th1}, {volume_th2}")
    print(f"ZCR thresholds: {zcr_th1}, {zcr_th2}")

    # Find endpoint indices based on volume thresholds
    index1 = np.where(volume > volume_th1)
    index2 = np.where(volume > volume_th2)
    # index3 = np.where(volume > volume_th3)
    # index4 = np.where(volume > volume_th4)

    # Find endpoint indices based on ZCR thresholds
    zcr_index1 = np.where(zcr > zcr_th1)
    zcr_index2 = np.where(zcr > zcr_th2)
    # zcr_index3 = np.where(zcr > zcr_th3)
    # zcr_index4 = np.where(zcr > zcr_th4)
    # Debug: Print indices
    # print(f"ZCR indices for threshold 4: {zcr_index4}")

    def frame_to_sample_index(indices, hop_length):
        return indices * hop_length

    # Convert frame indices to sample indices for volume
    end_point1 = frame_to_sample_index(index1[0], hop_length)
    end_point2 = frame_to_sample_index(index2[0], hop_length)
    # end_point3 = frame_to_sample_index(index3[0], hop_length)
    # end_point4 = frame_to_sample_index(index4[0], hop_length)

    # Convert frame indices to sample indices for ZCR
    zcr_end_point1 = frame_to_sample_index(zcr_index1[0], hop_length)
    zcr_end_point2 = frame_to_sample_index(zcr_index2[0], hop_length)
    # zcr_end_point3 = frame_to_sample_index(zcr_index3[0], hop_length)
    # zcr_end_point4 = frame_to_sample_index(zcr_index4[0], hop_length)

    # Check if zcr_end_point4 is empty
    # if len(zcr_end_point4) == 0:
    #     print("No ZCR endpoints found for threshold 4.")
    #     return

    # Plotting
    time = np.arange(len(y)) / sr
    frame_time = frame_to_sample_index(np.arange(len(volume)), hop_length) / sr
    zcr_frame_time = frame_to_sample_index(np.arange(len(zcr)), hop_length) / sr

    plt.figure(figsize=(12, 10))

    # Waveform plot
    plt.subplot(4, 1, 1)
    plt.plot(time, y)
    plt.ylabel('Amplitude')
    plt.title('Waveform and EP (method=volZcr)')
    plt.axvline(x=time[end_point1[0]], color='m')
    plt.axvline(x=time[end_point2[0]], color='g')
    # plt.axvline(x=time[end_point3[0]], color='k')
    # plt.axvline(x=time[end_point4[0]], color='r')
    plt.axvline(x=time[end_point1[-1]], color='m')
    plt.axvline(x=time[end_point2[-1]], color='g')
    # plt.axvline(x=time[end_point3[-1]], color='k')
    # plt.axvline(x=time[end_point4[-1]], color='r')
    # plt.legend(['Waveform', 'Boundaries by threshold 1', 'Boundaries by threshold 2', 'Boundaries by threshold 3', 'Boundaries by threshold 4'])
    plt.legend(['Waveform', 'Boundaries by threshold 1', 'Boundaries by threshold 2'])
    # Volume plot
    plt.subplot(4, 1, 2)
    plt.plot(frame_time, volume, '.-')
    plt.ylabel('Volume')
    plt.axhline(y=volume_th1, color='m')
    plt.axhline(y=volume_th2, color='g')
    # plt.axhline(y=volume_th3, color='k')
    # plt.axhline(y=volume_th4, color='r')
    # plt.legend(['Volume', 'Threshold 1', 'Threshold 2', 'Threshold 3', 'Threshold 4'])
    plt.legend(['Volume', 'Threshold 1', 'Threshold 2'])
    # Zero Crossing Rate plot
    plt.subplot(4, 1, 3)
    plt.plot(zcr_frame_time, zcr, '.-')
    plt.ylabel('ZCR')
    plt.axhline(y=zcr_th1, color='m')
    plt.axhline(y=zcr_th2, color='g')
    # plt.axhline(y=zcr_th3, color='k')
    # plt.axhline(y=zcr_th4, color='r')
    # plt.legend(['ZCR', 'Threshold 1', 'Threshold 2', 'Threshold 3', 'Threshold 4'])
    plt.legend(['ZCR', 'Threshold 1', 'Threshold 2'])

    # Waveform after EPD plot
    plt.subplot(4, 1, 4)
    plt.plot(time, y)
    plt.ylabel('Amplitude')
    plt.title('Waveform after EPD')
    plt.axvline(x=time[zcr_end_point1[0]], color='m')
    plt.axvline(x=time[zcr_end_point2[0]], color='g')
    # plt.axvline(x=time[zcr_end_point3[0]], color='k')
    # plt.axvline(x=time[zcr_end_point4[0]], color='r')
    plt.axvline(x=time[zcr_end_point1[-1]], color='m')
    plt.axvline(x=time[zcr_end_point2[-1]], color='g')
    # plt.axvline(x=time[zcr_end_point3[-1]], color='k')
    # plt.axvline(x=time[zcr_end_point4[-1]], color='r')
    # plt.legend(['Waveform', 'Boundaries by threshold 1', 'Boundaries by threshold 2', 'Boundaries by threshold 3', 'Boundaries by threshold 4'])
    plt.legend(['Waveform', 'Boundaries by threshold 1', 'Boundaries by threshold 2'])
    plt.tight_layout()
    plt.show()
def remove_silence(audio, threshold_low=30):
    non_silent_intervals = librosa.effects.split(audio, top_db=threshold_low)
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return non_silent_audio
def Frame(y,sr):
    hop_length = 86#重疊部分
    frame_length = 256
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return frames
def amplitude(y,sr):#響度曲線
    amplitude_db=librosa.amplitude_to_db(librosa.stft(y),ref=np.max)#響度
    # 計算時間軸
    time = np.linspace(0, len(y) / sr, amplitude_db.shape[1])
    # 計算平均分貝值
    mean_amplitude_db = np.mean(amplitude_db, axis=0)
    # 繪製折線圖
    plt.figure(figsize=(10, 6))
    plt.plot(time, mean_amplitude_db)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Amplitude in dB over Time')
    plt.grid(True)
    plt.show()
    return amplitude
def ACF(frame):
    frame_len=len(frame)
    acf=np.zeros(frame_len)
    for i in range(frame_len):
        acf[i]=np.sum(frame[i:frame_len]*frame[0:frame_len-i])
    return acf
def AMDF(frame):
    frame_len = len(frame)
    amdf = np.zeros(frame_len)
    for tau in range(frame_len):
        amdf[tau] = np.sum(np.abs(frame[:frame_len-tau] - frame[tau:frame_len])) / frame_len
    return amdf
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
def extract_mfcc(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs
def cepstral_mean_subtraction(mfccs):
    mean = np.mean(mfccs, axis=1, keepdims=True)
    cms_mfccs = mfccs - mean
    return cms_mfccs
def dynamic_time_warpingMFCC(x, y):
    distance, path = fastdtw(x.T, y.T, dist=euclidean)
    return distance, path
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
# 計算相似度分數
def calculateAB():
    # 定義符號
    a, b = symbols('a b')
    # 定義方程式
    eq1 = Eq(1 + a * (5)**b, 10/9)
    eq2 = Eq(1 + a * (6)**b, 5/3)
    # 求解方程
    solution = solve((eq1, eq2), (a, b))
def distance_to_score(dist, a, b):
    return 100 / (1 + a * (dist ** b))
def calculate_distances(ave_mag_A, ave_mag_B, lin_pitch_A, lin_pitch_B, mfccs_A, mfccs_B):
    distanceAVG, pathAVG = dynamic_time_warpingAVG(ave_mag_A, ave_mag_B)
    distancePIT, pathPIT = dynamic_time_warpingAVG(lin_pitch_A, lin_pitch_B)
    distanceMFCC, pathMFCC = dynamic_time_warpingMFCC(mfccs_A, mfccs_B)
    return distanceAVG, distancePIT, distanceMFCC
def optimize_parameters(distAVG, distPIT, distMFCC, initial_guess):
    result = minimize(lambda params: -scoring_function(params, distAVG, distPIT, distMFCC), initial_guess, method='SLSQP', bounds=[(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, 1), (0, 1), (0, 1)], constraints={'type': 'eq', 'fun': lambda params: params[6] + params[7] + params[8] - 1})
    return result.x
def scoring_function(params, distAVG, distPIT, distMFCC):
    a1, b1, a2, b2, a3, b3, w1, w2, w3 = params
    scoreAVG = distance_to_score(distAVG, a1, b1)
    scorePIT = distance_to_score(distPIT, a2, b2)
    scoreMFCC = distance_to_score(distMFCC, a3, b3)
    total_score = w1 * scoreAVG + w2 * scorePIT + w3 * scoreMFCC
    return total_score
@app.route('/')
def index():
    return render_template('Spanish.html')

@app.route('/speech_file', methods=['POST'])
def receive_speech_file():
    global selected_file,audio_file_path_A
    selected_file = request.json.get('speechFile')
    audio_file_path_A = os.path.join(current_dir,'static','audio', selected_file)
    return jsonify({'message': 'Received speechFile successfully'})

@app.route('/record_audio', methods=['POST'])
def record_audio():
    global audio_file_path_B
    audio_file_path_B = os.path.join(current_dir,'static','audio','user_input.wav')
    record_audio_to_file(audio_file_path_B, duration=3, channels=1, rate=44100, frames_per_buffer=1)
    return jsonify({'message': 'Audio recorded successfully'})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_A, sr_A = load_audio(audio_file_path_A)
    audio_B, sr_B = load_audio(audio_file_path_B)
    audio_pre_A = remove_silence(audio_A)
    audio_pre_B = remove_silence(audio_B)
    pitch_A = extract_pitch(audio_pre_A, sr_A)#pitch tracking A 基頻軌跡
    pitch_B = extract_pitch(audio_pre_B, sr_B)#pitch tracking B 基頻軌跡
    ave_mag_A = AveMag(audio_pre_A, sr_A)#音量強度曲線
    ave_mag_B = AveMag(audio_pre_B, sr_B)#音量強度曲線
    
    # 插值处理
    target_length = max(len(ave_mag_A), len(ave_mag_B))
    ave_mag_A_interp = Interpolation(ave_mag_A, target_length)
    ave_mag_B_interp = Interpolation(ave_mag_B, target_length)
    #基頻軌跡插值處理
    pitch_A_interp = Interpolation(pitch_A, target_length)
    pitch_B_interp = Interpolation(pitch_B, target_length)
    lin_pitch_A, lin_pitch_B = linear_shifting(pitch_A_interp, pitch_B_interp)
    ave_mag_B_scaled_ajt, ave_mag_A_scaled_ajt = linear_scaling(ave_mag_A_interp, ave_mag_B_interp)
     # 確保輸入為一維
    ave_mag_A_scaled = np.array(ave_mag_A_scaled_ajt).flatten()
    ave_mag_B_scaled = np.array(ave_mag_B_scaled_ajt).flatten()
    mfccs_A = extract_mfcc(audio_pre_A, sr_A)
    mfccs_B = extract_mfcc(audio_pre_B, sr_B)
    # 使用 CMS 消除通道效應
    mfccs_A_cms = cepstral_mean_subtraction(mfccs_A)
    mfccs_B_cms = cepstral_mean_subtraction(mfccs_B)
    #音量強度曲線的計算 DTW 距離和路徑
    distanceAVG, pathAVG = dynamic_time_warpingAVG(ave_mag_A_scaled, ave_mag_B_scaled)
    #基頻軌跡的 DTW 距離和路徑
    distancePIT, pathPIT = dynamic_time_warpingAVG(lin_pitch_A, lin_pitch_B)
    #MFCC的計算 DTW 距離和路徑
    distanceMFCC, pathMFCC = dynamic_time_warpingMFCC(mfccs_A_cms, mfccs_B_cms)
    score = distance_to_score(distanceAVG, 0.00000000150193575922916, 9.82746911958941)
    
    return jsonify({"score": round(score, 3)})

#if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    audio_file_path_A = os.path.join(current_dir,'static','audio','F.wav')
    audio_A, sr_A = load_audio(audio_file_path_A)
    #print(audio_A)
    audio_file_path_B=os.path.join(current_dir,'static','audio','A.wav')
    # record_audio_to_file('D:\\系統檔 Documents\\GitHub\\Alcoho.github.io\\flask-templete\\static\\audio\\user_input.wav',duration=3, channels=1, rate=44100, frames_per_buffer=1)
    audio_B, sr_B = load_audio(audio_file_path_B)
    # 使用 os.path.join 來組合完整的檔案路徑
    file_path = os.path.join(current_dir, 'audio_B.txt')
    #np.savetxt(file_path, audio_A)
    # audio_pre_A=remove_silence(audio_A)
    # audio_pre_B=remove_silence(audio_B)
    # End_pont_detection(audio_A,sr_A)
    # End_pont_detection(audio_B,sr_B)
    # pitch_A=extract_pitch(audio_pre_A,sr_A)#pitch tracking A 基頻軌跡
    # pitch_B=extract_pitch(audio_pre_B,sr_B)#pitch tracking B
    # print(pitch_A)
    # print(pitch_B)
    # librosa.display.waveshow(audio_pre_A, sr=sr_A)
    # plt.title('Teacher_audio')
    # plt.show()
    # plt.figure()
    # librosa.display.waveshow(audio_pre_B, sr=sr_B)
    # plt.title('Student_audio')
    # plt.show()
    # amplitude(audio_pre_B,sr_B)
    # ave_mag_A =AveMag(audio_pre_A,sr_A)#音量強度曲線
    # ave_mag_B = AveMag(audio_pre_B,sr_B)#音量強度曲線
    # #音量強度曲線插值處理
    # target_length = max(len(ave_mag_A), len(ave_mag_B))
    # ave_mag_A_interp = Interpolation(ave_mag_A, target_length)
    # ave_mag_B_interp = Interpolation(ave_mag_B, target_length)
    # #基頻軌跡插值處理
    # target_length = max(len(pitch_A), len(pitch_B))
    # pitch_A_interp = Interpolation(pitch_A, target_length)
    # pitch_B_interp = Interpolation(pitch_A, target_length)
    # lin_pitch_A,lin_pitch_B=linear_shifting(pitch_A_interp,pitch_B_interp)
    # # 线性缩放
    # ave_mag_B_scaled_ajt, ave_mag_A_scaled_ajt = linear_scaling(ave_mag_A_interp, ave_mag_B_interp)
    # # 确保输入为1维
    # ave_mag_A_scaled = np.array(ave_mag_A_scaled_ajt).flatten()
    # ave_mag_B_scaled = np.array(ave_mag_B_scaled_ajt).flatten()
    # plt.plot(ave_mag_A_scaled, label='Scaled Audio A')
    # plt.plot(ave_mag_B_scaled, label='Scaled Audio B')
    # plt.legend()
    # plt.title('Scaled Average Magnitude')
    # plt.xlabel('Frames')
    # plt.ylabel('Scaled Average Magnitude')
    # plt.show()

    # #提取 MFCC 特徵
    # mfccs_A= extract_mfcc(audio_pre_A, sr_A)
    # mfccs_B= extract_mfcc(audio_pre_B, sr_B)
    # # 使用 CMS 消除通道效應
    # mfccs_A_cms = cepstral_mean_subtraction(mfccs_A)
    # mfccs_B_cms = cepstral_mean_subtraction(mfccs_B)
    # #音量強度曲線的計算 DTW 距離和路徑
    # distanceAVG, pathAVG = dynamic_time_warpingAVG(ave_mag_A_scaled, ave_mag_B_scaled)
    # #基頻軌跡的 DTW 距離和路徑
    # distancePIT, pathPIT = dynamic_time_warpingAVG(lin_pitch_A, lin_pitch_B)
    # #MFCC的計算 DTW 距離和路徑
    # distanceMFCC, pathMFCC = dynamic_time_warpingMFCC(mfccs_A_cms, mfccs_B_cms)

    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(ave_mag_A, label='Original A')
    # plt.plot(ave_mag_B, label='Original B')
    # plt.legend()

    # plt.subplot(3, 1, 2)
    # plt.plot(ave_mag_A_interp, label='Interpolated A')
    # plt.plot(ave_mag_B_interp, label='Interpolated B')
    # plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.plot(ave_mag_A_scaled, label='Scaled A')
    # plt.plot(ave_mag_B_scaled, label='Scaled B')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    # # 打印結果
    # print(f"The distanceAVG between the two audio files is: {distanceAVG}")
    # print(f"The DTW path is: {pathAVG}")
    # print(f"The distancePIT between the two audio files is: {distancePIT}")
    # print(f"The DTW path is: {pathPIT}")
    # print(f"The distanceMFCC between the two audio files is: {distanceMFCC}")
    # print(f"The DTW path is: {pathMFCC}")
    # score=distance_to_score(distanceAVG, 0.00000000150193575922916, 9.82746911958941)
    # print(score)
    # score=distance_to_score(distancePIT, 0.00000000150193575922916, 9.82746911958941)
    # print(score)
    # #return 100 / (1 + a * (dist ** b))
    # # Assuming ave_mag_A_scaled, ave_mag_B_scaled, lin_pitch_A, lin_pitch_B, mfccs_A_cms, mfccs_B_cms are defined and populated with your data.
    # distAVG, distPIT, distMFCC = calculate_distances(ave_mag_A_scaled, ave_mag_B_scaled, lin_pitch_A, lin_pitch_B, mfccs_A_cms, mfccs_B_cms)
    # # Initial guess for optimization: [a1, b1, a2, b2, a3, b3, w1, w2, w3]
    # #a b為距離轉成分數的參數,w為為三個特徵的權重
    # #為了求得我們設計了以下實驗：首先我們先收集10句CNN互動英語的句字，當成標準語音，再請實驗室同學依此10句錄音，當成測試語音，總共收集了320句測試語音，每一句跟標準答案比
    # initial_guess = [1, 1, 1, 1, 1, 1, 0.7, 0.2, 0.1]
    # optimized_params = optimize_parameters(distAVG, distPIT, distMFCC, initial_guess)
    # print("Optimized parameters:", optimized_params)
    # total_score=scoring_function(optimized_params, distAVG, distPIT, distMFCC)
    # print("Total score:", total_score)


    # # 預處理雜音
    # audio_preA2 = preprocess_audio(audio_A)
    # audio_preB2 = preprocess_audio(audio_B)
    # # 去除靜音部分
    # audio_A1 = remove_silence(audio_preA2)
    # audio_B1 = remove_silence(audio_preB2)
    # # 提取 MFCC 特徵
    # mfccs_A= extract_mfcc(audio_A1, sr_A)
    # mfccs_B= extract_mfcc(audio_B1, sr_B)
    # # 使兩個音檔的 MFCC 特徵具有相同的維度
    # min_length = min(mfccs_A.shape[1], mfccs_B.shape[1])
    # mfccs_A1 = mfccs_A[:, :min_length]
    # mfccs_B1 = mfccs_B[:, :min_length]
    # # 正規化 MFCC 特徵
    # mfccs_A_normalized = normalize_mfcc(mfccs_A1)
    # mfccs_B_normalized = normalize_mfcc(mfccs_B1)
    # print(mfccs_A1)
    # print('\n')
    # print(mfccs_B1)
    # plt.figure()
    # librosa.display.waveshow(mfccs_A_normalized, sr=sr_A)
    # plt.title('Teacher')
    # plt.show()

    # plt.figure()
    # librosa.display.waveshow(audio_B, sr=sr_B)
    # plt.title('Student_audio')
    # plt.show()


    # # 計算相似度分數
    # score = 100-compute_similarity_score(mfccs_A_normalized, mfccs_B_normalized)
    # print(score)
    # def record_audio_to_file(filename, duration=3, channels=2, rate=44100, frames_per_buffer=1024):
#     """Record user's input audio and save it to the specified file."""
#     FORMAT = pyaudio.paInt16
#     p = pyaudio.PyAudio()

#     # Open stream
#     stream = p.open(format=FORMAT, channels=channels, rate=rate, frames_per_buffer=frames_per_buffer, input=True)
#     print("Start recording...")

#     frames = []
#     # Record for the given duration
#     for _ in range(0, int(rate / frames_per_buffer * duration)):
#         data = stream.read(frames_per_buffer)
#         frames.append(data)

#     print("Recording stopped")

#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     # Save the recorded data as a WAV file
#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(channels)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(rate)
#     wf.writeframes(b''.join(frames))
#     wf.close()

#     print(f"Audio recorded and saved to {filename}")
    # with wave.open('C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\user_input.wav') as fw:
    #     nchannels=fw.getnchannels()#通道數
    #     sampwidth=fw.getsampwidth()#帶寬
    #     framerate=fw.getframerate()#音框率
    #     nframes =fw.getnframes()#音框數量
    #     strData = fw.readframes(nframes)#把所有音框轉到字串
    #     waveData = np.fromstring(strData, dtype=np.int16)#將音框字串用16為位元整數到波的資料中，為音框總數，亦即音量強度曲線的長度
    #     waveData = waveData*1.0/max(abs(waveData))  # 標準化
    #     fw.close()
        # frames_A=Frame(audio_pre_A,sr_A)
    # frames_B=Frame(audio_pre_B,sr_B)
    # amdf_A=amdfPlot(frames_A,sr_B,0,nchannels=1,sampwidth=2)
    # amdf_B=amdfPlot(frames_B,sr_B,0,nchannels=1,sampwidth=2)

    # main_period_A = calculate_neighborhood_size(amdf_A)
    # main_period_B = calculate_neighborhood_size(amdf_B)
    # # 設定閾值
    # threshold_A = np.mean(amdf_A)  # 一個可能的閾值選擇方法
    # threshold_B = np.mean(amdf_B)  # 一個可能的閾值選擇方法

    # # 高剪切處理
    # clipped_amdf_A = high_clipping(amdf_A, threshold_A)
    # clipped_amdf_B = high_clipping(amdf_B, threshold_B)

    # # 找局部最小值
    # main_period_A = calculate_neighborhood_size(amdf_A)
    # main_period_B = calculate_neighborhood_size(amdf_B)
    # local_minima_A = find_local_minima(clipped_amdf_A,main_period_A)
    # local_minima_B = find_local_minima(clipped_amdf_B,main_period_B)
    # local_minima_indices_A = np.where(local_minima_A)[0]
    # local_minima_indices_B = np.where(local_minima_B)[0]

    # # 計算頻率
    # frequency_A = calculate_frequency(local_minima_indices_A, sr_A)
    # frequency_B = calculate_frequency(local_minima_indices_B, sr_B)
    # print("Estimated Frequency A : {:.2f} Hz".format(frequency_A))
    # print("Estimated Frequency B : {:.2f} Hz".format(frequency_B))

    # 標準音檔 畫圖
    # plt.figure(figsize=(10, 6))
    # plt.plot(amdf_A, label='AMDF')
    # plt.plot(clipped_amdf_A, label='High Clipped AMDF')
    # # plt.scatter(local_minima_indices, clipped_amdf[local_minima_indices], color='red', label='Local Minima')
    # plt.legend()
    # plt.xlabel('Samples')
    # plt.ylabel('AMDF')
    # plt.title('AMDF with Local Minima')
    # plt.show()

    # # 測試音檔 畫圖
    # plt.figure(figsize=(10, 6))
    # plt.plot(amdf_B, label='AMDF')
    # plt.plot(clipped_amdf_B, label='High Clipped AMDF')
    # # plt.scatter(local_minima_indices, clipped_amdf[local_minima_indices], color='red', label='Local Minima')
    # plt.legend()
    # plt.xlabel('Samples')
    # plt.ylabel('AMDF')
    # plt.title('AMDF with Local Minima')
    # plt.show()
# def compute_similarity_score(distA, distB):
#     return np.mean(np.abs(distA - distA)) * 100

# # 正規化 MFCC 特徵
# def normalize_mfcc(mfccs):
#     return (mfccs - np.mean(mfccs)) / np.std(mfccs)
# def amdfPlot(waveData,sr,idx1=0,nchannels=1,sampwidth=2):
#     # nchannels=1#通道數
#     # sampwidth=2#帶寬就是16位元
#     # waveData=audio_pre_B*1.0/np.abs(audio_pre_B)
#     #waveData = frames_B[0] * 1.0 / np.max(np.abs(frames_B[0]))
#     #waveData=audio_pre_B#音訊的訊號值，類型是ndarray
#     print(waveData[0])
#     # plot the wave
#     frameSize = 256#音框大小，若直接觀察音訊的波形，只要聲音穩定，我們並不難直接看到基本週期的存在，以一個 3 秒的音叉聲音來說，我們可以取一個 256 點的音框，將此音框畫出來後，就可以很明顯地看到基本週期
#     hop_length=86#重疊部分
#     framerate=sr/(frameSize-hop_length)#音框率
#     time = np.arange(0, len(waveData)) * (1.0 / framerate)
#     #idx1 = 0#音框頭
#     idx2 = idx1+frameSize
#     index1 = idx1*1.0 / framerate
#     index2 = idx2*1.0 / framerate#邊界設定
#     amdf=AMDF(waveData[idx1:idx2])
#     pl.subplot(311)
#     pl.plot(time, waveData,'b')
#     pl.plot([index1,index1],[-1,1],'r')
#     pl.plot([index2,index2],[-1,1],'r')
#     pl.xlabel("time (seconds)")
#     pl.ylabel("Amplitude")

#     pl.subplot(312)
#     pl.plot(np.arange(frameSize),waveData[idx1:idx2],'r')
#     pl.xlabel("index in 1 frame")
#     pl.ylabel("Amplitude")

#     pl.subplot(313)
#     pl.plot(np.arange(frameSize),amdf,'g')
#     pl.xlabel("index in 1 frame")
#     pl.ylabel("AMDF")
#     pl.show()
#     return amdf
# def calculate_neighborhood_size(amdf):
#     peaks = np.diff(np.sign(np.diff(amdf))).nonzero()[0] + 1  # local max
#     if len(peaks) > 0:
#         main_period = peaks[0]
#         return main_period
#     return 20  # default value if no peaks found
# def high_clipping(amdf, threshold):
#     return np.where(amdf > threshold, threshold, amdf)

# def find_local_minima(amdf,main_period):
#     neighborhood_size = main_period  # 可調整
#     local_max = maximum_filter1d(amdf, size=neighborhood_size, mode='constant')
#     local_minima = (amdf == local_max)
#     return local_minima

# def calculate_frequency(local_minima_indices, sr):
#     periods = np.diff(local_minima_indices)
#     if len(periods) > 0:
#         period = np.mean(periods)
#         frequency = sr / period
#     else:
#         frequency = 0
#     return frequency