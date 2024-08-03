import numpy as np
import matplotlib.pyplot as plt
import librosa

# 讀取音頻文件
wave_file = 'C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\F.wav'
y, sr = librosa.load(wave_file, sr=None)
y = y - np.mean(y)  # zero-mean subtraction

# 設置參數
frame_size = 256
hop_length = 128

# 分幀
frames = librosa.util.frame(y, frame_length=frame_size, hop_length=hop_length)

# 計算音量
volume = np.sum(np.abs(frames), axis=0)

# 計算零交叉率（ZCR）
zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_size, hop_length=hop_length)[0]

# 計算音量閾值
volume_th1 = np.max(volume) * 0.1
volume_th2 = np.median(volume) * 0.1
volume_th3 = np.min(volume) * 10
volume_th4 = volume[0] * 5

# 計算ZCR閾值
zcr_th1 = np.max(zcr) * 0.1
zcr_th2 = np.median(zcr) * 0.1
zcr_th3 = np.min(zcr) * 10
zcr_th4 = zcr[0] * 5

# Debug: 打印閾值
print(f"Volume thresholds: {volume_th1}, {volume_th2}, {volume_th3}, {volume_th4}")
print(f"ZCR thresholds: {zcr_th1}, {zcr_th2}, {zcr_th3}, {zcr_th4}")

# 找到超過音量閾值的索引
index1 = np.where(volume > volume_th1)
index2 = np.where(volume > volume_th2)
index3 = np.where(volume > volume_th3)
index4 = np.where(volume > volume_th4)

# 找到超過ZCR閾值的索引
zcr_index1 = np.where(zcr > zcr_th1)
zcr_index2 = np.where(zcr > zcr_th2)
zcr_index3 = np.where(zcr > zcr_th3)
zcr_index4 = np.where(zcr > zcr_th4)

# Debug: 打印索引
print(f"ZCR indices for threshold 4: {zcr_index4}")

def frame_to_sample_index(indices, hop_length):
    return indices * hop_length

# 將幀索引轉換為樣本索引（音量）
end_point1 = frame_to_sample_index(index1[0], hop_length)
end_point2 = frame_to_sample_index(index2[0], hop_length)
end_point3 = frame_to_sample_index(index3[0], hop_length)
end_point4 = frame_to_sample_index(index4[0], hop_length)

# 將幀索引轉換為樣本索引（ZCR）
zcr_end_point1 = frame_to_sample_index(zcr_index1[0], hop_length)
zcr_end_point2 = frame_to_sample_index(zcr_index2[0], hop_length)
zcr_end_point3 = frame_to_sample_index(zcr_index3[0], hop_length)
zcr_end_point4 = frame_to_sample_index(zcr_index4[0], hop_length)

# 檢查 zcr_end_point4 是否為空
if len(zcr_end_point4) == 0:
    print("No ZCR endpoints found for threshold 4.")
    zcr_end_point4 = [0, len(y) - 1]  # Fallback to the entire range

# 繪圖
time = np.arange(len(y)) / sr
frame_time = frame_to_sample_index(np.arange(len(volume)), hop_length) / sr
zcr_frame_time = frame_to_sample_index(np.arange(len(zcr)), hop_length) / sr

plt.figure(figsize=(12, 10))

# 波形圖
plt.subplot(4, 1, 1)
plt.plot(time, y)
plt.ylabel('Amplitude')
plt.title('Waveform and EP (method=volZcr)')
plt.axvline(x=time[end_point1[0]], color='m')
plt.axvline(x=time[end_point2[0]], color='g')
plt.axvline(x=time[end_point3[0]], color='k')
plt.axvline(x=time[end_point4[0]], color='r')
plt.axvline(x=time[end_point1[-1]], color='m')
plt.axvline(x=time[end_point2[-1]], color='g')
plt.axvline(x=time[end_point3[-1]], color='k')
plt.axvline(x=time[end_point4[-1]], color='r')
plt.legend(['Waveform', 'Boundaries by threshold 1', 'Boundaries by threshold 2', 'Boundaries by threshold 3', 'Boundaries by threshold 4'])

# 音量圖
plt.subplot(4, 1, 2)
plt.plot(frame_time, volume, '.-')
plt.ylabel('Volume')
plt.axhline(y=volume_th1, color='m')
plt.axhline(y=volume_th2, color='g')
plt.axhline(y=volume_th3, color='k')
plt.axhline(y=volume_th4, color='r')
plt.legend(['Volume', 'Threshold 1', 'Threshold 2', 'Threshold 3', 'Threshold 4'])

# ZCR 圖
plt.subplot(4, 1, 3)
plt.plot(zcr_frame_time, zcr, '.-')
plt.ylabel('ZCR')
plt.axhline(y=zcr_th1, color='m')
plt.axhline(y=zcr_th2, color='g')
plt.axhline(y=zcr_th3, color='k')
plt.axhline(y=zcr_th4, color='r')
plt.legend(['ZCR', 'Threshold 1', 'Threshold 2', 'Threshold 3', 'Threshold 4'])

# EPD 之後的波形圖
plt.subplot(4, 1, 4)
plt.plot(time, y)
plt.ylabel('Amplitude')
plt.title('Waveform after EPD')
plt.axvline(x=time[zcr_end_point1[0]], color='m')
plt.axvline(x=time[zcr_end_point2[0]], color='g')
plt.axvline(x=time[zcr_end_point3[0]], color='k')
plt.axvline(x=time[zcr_end_point4[0]], color='r')
plt.axvline(x=time[zcr_end_point1[-1]], color='m')
plt.axvline(x=time[zcr_end_point2[-1]], color='g')
plt.axvline(x=time[zcr_end_point3[-1]], color='k')
plt.axvline(x=time[zcr_end_point4[-1]], color='r')
plt.legend(['Waveform', 'Boundaries by threshold 1', 'Boundaries by threshold 2', 'Boundaries by threshold 3', 'Boundaries by threshold 4'])

plt.tight_layout()
plt.show()