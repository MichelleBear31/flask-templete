import pickle
import numpy as np
import cv2
import math
import os

# 取得當前目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 設置 .pkl 檔案的具體路徑
file_path = os.path.join(current_dir,'PKL','2.pkl')  # 假設 .pkl 文件名為 audio_A.pkl

# 讀取 .pkl 檔案
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 將資料序列化為 bytes
data_bytes = pickle.dumps(data)

# 將 bytes 轉換為 numpy array
data_array = np.frombuffer(data_bytes, dtype=np.uint8)

# 計算合適的圖片大小
size = int(math.ceil(math.sqrt(len(data_array))))
image_array = np.zeros((size, size), dtype=np.uint8)

# 將資料填充到圖片像素中
image_array.flat[:len(data_array)] = data_array

# 將 numpy array 轉換為 OpenCV 的圖片格式
image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

# 儲存圖片
image_path = os.path.join(current_dir, "PKL2image.png")
cv2.imwrite(image_path, image)

print(f"Data has been converted to an image and saved to {image_path}.")