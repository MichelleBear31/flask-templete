# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# import matplotlib.pyplot as plt
# import os

# # Define the current directory and image folder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# image_folder = os.path.join(current_dir, 'VGG_MobileNet_mfcc_images', '1')

# # Get the list of image filenames
# image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

# # Check if there are at least 16 images
# if len(image_filenames) < 16:
#     print("Not enough images. Please make sure there are at least 16 images in the folder.")
# else:
#     # Load 16 images
#     images = [cv2.imread(os.path.join(image_folder, f)) for f in image_filenames[:16]]

#     # Loop through the images and compare each pair
#     for i in range(0, len(images), 2):
#         image1 = images[i]
#         image2 = images[i + 1]

#         # Compute the absolute pixel difference between the two images
#         pixel_diff = cv2.absdiff(image1, image2)

#         # Convert images to grayscale
#         gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#         gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#         # Compute the Structural Similarity Index (SSIM) between the two grayscale images
#         ssim_score, _ = ssim(gray1, gray2, full=True)

#         # Plot the images and the pixel difference
#         plt.figure(figsize=(12, 4))
        
#         plt.subplot(1, 3, 1)
#         plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
#         plt.title(f'Image {i + 1}')

#         plt.subplot(1, 3, 2)
#         plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
#         plt.title(f'Image {i + 2}')

#         plt.subplot(1, 3, 3)
#         plt.imshow(pixel_diff, cmap='gray')
#         plt.title(f'Pixel Difference\nSSIM Score: {ssim_score:.2f}')

#         # Display the plot
#         plt.show()



import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the images
image1_path = os.path.join(current_dir, 'VGG_MobileNet_mfcc_images', '1', '1.png')
image2_path = os.path.join(current_dir, 'VGG_MobileNet_mfcc_images', '2', '2.png')

# Load the images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Compute the absolute pixel difference between the two images
pixel_diff = cv2.absdiff(image1, image2)

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute the Structural Similarity Index (SSIM) between the two grayscale images
ssim_score, _ = ssim(gray1, gray2, full=True)

# Plot the images and the pixel difference
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Image 1')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Image 2')

plt.subplot(1, 3, 3)
plt.imshow(pixel_diff, cmap='gray')
plt.title(f'Pixel Difference\nSSIM Score: {ssim_score:.2f}')

# Display the plot
plt.show()

# 這個程式單純比對RGB顏色是否正確，沒辦法進行紋路比對