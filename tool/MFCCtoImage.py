import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 讀取音頻文件
audio_path = 'C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\A.wav'
y, sr = librosa.load(audio_path, duration=3)

# 計算MFCC特徵
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 繪製MFCC特徵圖
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

# 保存為圖片文件
plt.savefig('mfcc_features.png')
plt.show()