import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 讀取音檔
file_path = 'C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\A.wav'  # 替換為你的音檔路徑
y, sr = librosa.load(file_path)

# 計算短時傅立葉變換（STFT）
frame_size = 256
overlap = frame_size // 2
S = librosa.stft(y, n_fft=frame_size, hop_length=overlap)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# 視覺化頻譜圖
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='jet')
plt.colorbar(format='%+2.0f dB')
plt.title('頻譜圖')
plt.xlabel('時間 (秒)')
plt.ylabel('頻率 (Hz)')
plt.show()

# 計算能量分布
def compute_energy_distribution(S, sr, num_bands=4):
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
    band_limits = np.linspace(0, frequencies.max(), num_bands+1)
    energy_distribution = np.zeros(num_bands)

    for i in range(num_bands):
        band_mask = (frequencies >= band_limits[i]) & (frequencies < band_limits[i+1])
        band_energy = np.sum(np.abs(S[band_mask, :])**2, axis=0)
        energy_distribution[i] = np.mean(band_energy)

    return energy_distribution, band_limits

energy_distribution, band_limits = compute_energy_distribution(S, sr)

# 視覺化能量分布
band_centers = (band_limits[:-1] + band_limits[1:]) / 2

plt.figure(figsize=(10, 6))
plt.bar(band_centers, energy_distribution, width=np.diff(band_limits), align='center', edgecolor='black')
plt.xlabel('頻率 (Hz)')
plt.ylabel('能量')
plt.title('頻段能量分布')
plt.show()

# 打印能量分布
for i in range(len(band_centers)):
    print(f'頻段 {band_limits[i]:.2f} Hz - {band_limits[i+1]:.2f} Hz: 平均能量 = {energy_distribution[i]:.2f}')