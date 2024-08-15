import os
import librosa
import numpy as np
import soundfile as sf

# 設定參數
target_length = 1  # 目標長度為 1 秒
sample_rate = 44100  # 設定取樣率為 44100 Hz
trim_long_files = True  # 布林開關，True 表示裁剪長度大於 1 秒的音檔
MfccCheck = False
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定義資料夾路徑
folder_path = os.path.join(current_dir,'3')

# 檢查資料夾中的所有 wav 檔案
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(folder_path, filename)
        
        # 載入音檔
        y, sr = librosa.load(file_path, sr=sample_rate)

        # 取得音檔長度（秒）
        duration = librosa.get_duration(y=y, sr=sr)
        
        if MfccCheck:
            # 計算 MFCC 並列出長度
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            print(f'{filename} - MFCC Length: {mfcc.shape[1]}')
        
        # 處理小於 1 秒的音檔
        if duration < target_length:
            # 計算所需補充的樣本數
            padding_length = target_length * sr - len(y)
            
            # 補靜音
            y_padded = np.pad(y, (0, int(padding_length)), 'constant')
            
            # 儲存新的音檔
            sf.write(file_path, y_padded, sr)
            
            # 計算 MFCC 並列出長度
            mfcc = librosa.feature.mfcc(y=y_padded, sr=sr)
            print(f'{filename} - MFCC Length: {mfcc.shape[1]}')
        
        elif duration > target_length:
            if trim_long_files:
                # 裁剪至 1 秒
                y_trimmed = y[:int(target_length * sr)]
                
                # 儲存裁剪後的音檔
                sf.write(file_path, y_trimmed, sr)
                
                # 計算 MFCC 並列出長度
                mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr)
                print(f'{filename} (Trimmed) - MFCC Length: {mfcc.shape[1]}')
            else:
                print(f'{filename} is longer than 1 second and will not be processed.')