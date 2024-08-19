import pickle
import matplotlib.pyplot as plt

# 載入訓練歷史
history_path = 'LSTM_history.pkl'
with open(history_path, 'rb') as f:
    history = pickle.load(f)
# 繪製損失（Loss）圖表
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 繪製準確率（Accuracy）圖表
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from tensorflow.keras.models import load_model

# 加载模型
model = load_model('LSTM_model.keras')

# 输出模型的结构
model.summary()

# 或者，逐层打印模型的层名称和类型
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")