import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from tensorflow.keras.optimizers import RMSprop
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import GPUtil

# 选择第一个可用的 GPU（自动选择）
available_gpus = GPUtil.getGPUs()
if available_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    selected_gpu = available_gpus[0]
    print(f"Using GPU ID: 0, Name: {selected_gpu.name}, Load: {selected_gpu.load*100:.1f}%, Memory Free: {selected_gpu.memoryFree}MB / {selected_gpu.memoryTotal}MB")
else:
    print("No GPU available. Running on CPU.")

# 讀取資料
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'wave_fotoV2')
model_weights_path = 'cnnwavefotoV2_sift.weights.h5'
# 測試單張音檔的圖片
test_image_path = os.path.join(current_dir, 'static', 'audio', 'Fototest', 'Fototest3.png')
# 定義讀取影像的函數
def read_images_from_folder(folder, img_size=(100, 100)):
    imgs, labels = [], []
    for label in range(1, 4):
        label_folder = os.path.join(folder, str(label))
        for filename in os.listdir(label_folder):
            if filename.endswith('.png'):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#轉灰色
                imgs.append(img)
                labels.append(label - 1)
    return np.array(imgs), np.array(labels)
# 定義讀取單張圖片的函數
def read_single_image(image_path, img_size=(100, 100)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.array(img)
# 定義 SIFT 特徵提取器和 FLANN 匹配器
sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
# 定義 RootSIFT 特徵提取函數
def extract_rootsift_features(image):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors)
        return keypoints, descriptors.flatten()  # 直接保留完整特徵並扁平化
    return keypoints, np.zeros((128,))
# 提取特徵並填充
def extract_features_with_padding(images, expected_dim=128):
    sift_features = []
    for img in images:
        _, feature = extract_rootsift_features(img.squeeze(-1))
        if feature.shape[0] < expected_dim:
            feature = np.pad(feature, (0, expected_dim - feature.shape[0]), 'constant')
        elif feature.shape[0] > expected_dim:
            feature = feature[:expected_dim]
        sift_features.append(feature)
    return np.array(sift_features)
# 構建卷積神經網絡模型
def build_model():
    input_img = Input(shape=(100, 100, 1))
    cnn = Conv2D(128, kernel_size=(3, 3), activation='selu')(input_img)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
    cnn = Conv2D(256, kernel_size=(3, 3), activation='selu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
    cnn = Conv2D(512, kernel_size=(3, 3), activation='selu')(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
    cnn = Flatten()(cnn)

    sift_features_input = Input(shape=(128,))
    sift_flat = Dense(512, activation='selu')(sift_features_input)

    merged = Concatenate()([cnn, sift_flat])
    fc = Dense(1024, activation='selu')(merged)
    fc = Dropout(0.5)(fc)
    output = Dense(3, activation='softmax')(fc)

    model = tf.keras.models.Model(inputs=[input_img, sift_features_input], outputs=output)
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model
def compute_gradcam(model, img_array, sift_feature_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, sift_feature_array])
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

def plot_gradcam(gradcam, image):
    # 调整 gradcam 大小以匹配图像大小
    gradcam = cv2.resize(gradcam, (image.shape[1], image.shape[0]))

    # 将 gradcam 和 image 都转换为 float64 类型
    gradcam = gradcam.astype(np.float64)
    image = image.astype(np.float64)

    # 将 gradcam 归一化到 0-255 范围
    if np.max(gradcam) != 0:  # 防止除以0的错误
        gradcam = gradcam * 255.0 / np.max(gradcam)

    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Grad-CAM shape: {gradcam.shape}, dtype: {gradcam.dtype}")

    # 将 gradcam 与原始图像叠加
    superimposed_img = cv2.addWeighted(image, 0.6, gradcam, 0.4, 0)

    # 将结果转换为 uint8 类型
    superimposed_img = np.uint8(superimposed_img)

    return superimposed_img
def display_and_save_gradcam(model, test_image, test_sift_feature, last_conv_layer_name="conv2d_2", output_path="gradcam_output.png"):
    gradcam = compute_gradcam(model, test_image, test_sift_feature, last_conv_layer_name)
    test_image_gray = test_image.squeeze(0).squeeze(-1)  # 确保图像是二维的 (100, 100)
    gradcam_image = plot_gradcam(gradcam, test_image_gray)
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1,2,1)
    # plt.title("Original Image")
    # plt.imshow(test_image_gray, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.title("Grad-CAM")
    # plt.imshow(gradcam_image, cmap='gray')
    # 保存图像
    # plt.savefig(output_path)
    # plt.show()
def visualize_layer_activations(model, test_image, test_sift_feature):
    layer_outputs = [layer.output for layer in model.layers[:6]]  # 选择前几层
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict([test_image, test_sift_feature])

    for i, activation in enumerate(activations):
        plt.figure(figsize=(15, 15))
        plt.imshow(activation[0, :, :, 0], cmap='viridis')  # 显示第一张激活图
        plt.title(f"Activation Layer {i+1}")
        plt.show()
def plot_pca_features(features, labels, original_image):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.figure(figsize=(30, 30))
    plt.imshow(original_image, cmap='gray', extent=[reduced_features[:, 0].min(), reduced_features[:, 0].max(),
                                                    reduced_features[:, 1].min(), reduced_features[:, 1].max()])
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title('PCA of Features with Original Image Overlay')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# 使用t-SNE进行特征可视化
def plot_tsne_features(features, labels, original_image):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 20))
    plt.imshow(original_image, cmap='gray', extent=[reduced_features[:, 0].min(), reduced_features[:, 0].max(),
                                                    reduced_features[:, 1].min(), reduced_features[:, 1].max()])
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title('t-SNE of Features with Original Image Overlay')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

# 使用Grad-CAM圖形輸出可視化模型判斷類別過程
def display_gradcam(model, test_image, test_sift_feature, last_conv_layer_name="conv2d_2"):
    gradcam = compute_gradcam(model, test_image, test_sift_feature, last_conv_layer_name)
    test_image_gray = test_image.squeeze(0).squeeze(-1)  # 确保图像是二维的 (100, 100)
    gradcam_image = plot_gradcam(gradcam, test_image_gray)

    plt.figure(figsize=(12, 12))
    plt.imshow(test_image_gray, cmap='gray')
    plt.imshow(gradcam_image, cmap='jet', alpha=0.5)  # 使用 alpha 参数进行叠加
    plt.title("Grad-CAM with Original Image Overlay")
    plt.show()
# 在测试图像上应用PCA、t-SNE、Grad-CAM，并显示结果
def visualize_all(features, labels, model, test_image, test_sift_feature):
    test_image_gray = test_image.squeeze(0).squeeze(-1)  # 确保图像是二维的 (100, 100)
    
    # PCA可视化
    plot_pca_features(features, labels, test_image_gray)
    
    # t-SNE可视化
    plot_tsne_features(features, labels, test_image_gray)
    
    # Grad-CAM可视化
    display_gradcam(model, test_image, test_sift_feature, last_conv_layer_name="conv2d_2")

    #可视化中间层的激活图
    visualize_layer_activations(model, test_image.reshape((1, 100, 100, 1)), test_sift_feature)
train_model = input("Do you want to retrain the model? (y/n): ").strip().lower()

# 加载图像数据
images, labels = read_images_from_folder(data_path)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=42)

# 重新塑形影像數據以適應卷積神經網絡的輸入格式
train_images = train_images.reshape((train_images.shape[0], 100, 100, 1))
test_images = test_images.reshape((test_images.shape[0], 100, 100, 1))

# 提取並收集所有训练图像的 SIFT 描述符
train_sift_features = extract_features_with_padding(train_images)
test_sift_features = extract_features_with_padding(test_images)

# 構建或加载模型
model = build_model()
if train_model == 'y':
    # 訓練模型
    train_history = model.fit([train_images, train_sift_features], train_labels, epochs=130, batch_size=16, validation_data=([test_images, test_sift_features], test_labels))

    # 保存模型權重
    model.save_weights(model_weights_path)
    print(f"Model weights saved to {model_weights_path}")
else:
    # 加载已保存的權重
    model.load_weights(model_weights_path)
    print(f"Model weights loaded from {model_weights_path}")

# 評估模型
test_loss, test_acc = model.evaluate([test_images, test_sift_features], test_labels)
print(f"Test accuracy: {test_acc:.2f}")
test_image = read_single_image(test_image_path)

# 提取和处理单张测试图片的 SIFT 特征
_, test_sift_feature = extract_rootsift_features(test_image)

# 确保特征的长度为 128
expected_dim = 128
if test_sift_feature.shape[0] < expected_dim:
    test_sift_feature = np.pad(test_sift_feature, (0, expected_dim - test_sift_feature.shape[0]), 'constant')
elif test_sift_feature.shape[0] > expected_dim:
    test_sift_feature = test_sift_feature[:expected_dim]

# 调整形状为 (1, 128)
test_sift_feature = test_sift_feature.reshape(1, -1)

# 预测结果
predictions = model.predict([test_image.reshape((1, 100, 100, 1)), test_sift_feature])
predicted_class = np.argmax(predictions)

# 输出预测结果
print(f"Predicted folder (class): {predicted_class + 1}")  # +1 使结果与资料夹号对应
print(f"Prediction confidence: {np.max(predictions) * 100:.2f}%")

# 进行PCA和t-SNE可视化
# plot_pca_features(train_sift_features, train_labels,test_image)
# plot_tsne_features(train_sift_features, train_labels,test_image)

# 生成并保存 Grad-CAM 圖形輸出可視化模型判斷類別過程
display_and_save_gradcam(model, test_image.reshape((1, 100, 100, 1)), test_sift_feature, last_conv_layer_name="conv2d_2", output_path="gradcam_output.png")
visualize_all(train_sift_features, train_labels, model, test_image.reshape((1, 100, 100, 1)), test_sift_feature)