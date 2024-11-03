import cv2
import matplotlib.pyplot as plt
from skimage import filters, util
from skimage.morphology import disk
from skimage.transform import rescale


image = cv2.imread('data/lab_1/8.jpg', cv2.IMREAD_GRAYSCALE)
image = rescale(image, 0.15)


def add_random_noise(image):
    noise = util.random_noise(image, mode='speckle', mean=0.1)
    return noise


def high_pass_filter(image):
    edges = filters.sobel(image)
    return edges


def low_pass_filter(image, median):
    low_pass = filters.median(image, disk(median))
    return low_pass


def gaussian_filter(image, sigma):
    return filters.gaussian(image, sigma=sigma)


# Отримання зображень
noisy_image = add_random_noise(image)
high_pass_image = high_pass_filter(noisy_image)
low_pass_image = low_pass_filter(noisy_image, 3)
low_pass_image_11 = low_pass_filter(noisy_image, 11)
gaussian_image = gaussian_filter(noisy_image, 3)
gaussian_image_6 = gaussian_filter(noisy_image, 6)

fig, axs = plt.subplots(2, 3, figsize=(20, 10))

axs[0, 0].imshow(image)
axs[0, 0].axis('off')
axs[0, 0].set_title("Оригінальне зображення")

axs[0, 1].imshow(noisy_image)
axs[0, 1].axis('off')
axs[0, 1].set_title("Випадкове шумове зображення")

axs[0, 2].imshow(high_pass_image)
axs[0, 2].axis('off')
axs[0, 2].set_title("Високочастотна фільтрація")

axs[1, 0].imshow(low_pass_image)
axs[1, 0].axis('off')
axs[1, 0].set_title("Низькочастотна фільтрація (3)")

axs[1, 1].imshow(low_pass_image_11)
axs[1, 1].axis('off')
axs[1, 1].set_title("Низькочастотна фільтрація (11)")

axs[1, 2].imshow(gaussian_image)
axs[1, 2].axis('off')
axs[1, 2].set_title("Гауссовий фільтр (sigma=3)")

axs[1, 2].imshow(gaussian_image_6)
axs[1, 2].axis('off')
axs[1, 2].set_title("Гауссовий фільтр (sigma=6)")

plt.tight_layout()
plt.show()
