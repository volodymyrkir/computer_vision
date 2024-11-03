import os

import cv2
import matplotlib.pyplot as plt

IMAGE_FOLDER = 'data/'


def collect_images(image_dir) -> list:
    """Returns list of tuples containing RGB, BGR, CMY and HSV images each."""
    images_tuples = []
    for filename in os.listdir(image_dir):

        # BGR (вже маємо вихідне зображення в BGR)
        bgr_image = cv2.imread(os.path.join(image_dir, filename))

        # RGB (OpenCV за замовчуванням завантажує зображення в BGR, тому треба перемкнути його на RGB)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # CMY - отримується шляхом інверсії кольорів RGB
        # CMY — це інверсія RGB, тому нормалізуємо до 1, щоб працювати з кольорами від 0 до 1
        cmy_image = 1 - rgb_image / 255.0
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        images_tuples.append((rgb_image, bgr_image, cmy_image, hsv_image))

    return images_tuples


def display_images_grid(images_tuples):
    """Displays all tuples of images (RGB, BGR, CMY, HSV) in a single grid."""
    num_images = len(images_tuples)
    fig, axs = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))

    for i, (rgb_image, bgr_image, cmy_image, hsv_image) in enumerate(images_tuples):
        axs[i, 0].imshow(rgb_image)
        axs[i, 0].set_title("RGB")

        axs[i, 1].imshow(bgr_image)
        axs[i, 1].set_title("BGR")

        axs[i, 2].imshow(cmy_image)
        axs[i, 2].set_title("CMY")

        axs[i, 3].imshow(hsv_image)
        axs[i, 3].set_title("HSV")

        for ax in axs[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images = collect_images(IMAGE_FOLDER)
    display_images_grid(images)
