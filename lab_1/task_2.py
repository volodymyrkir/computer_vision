import cv2
import matplotlib.pyplot as plt


def display_hist_plots(images_list):
    """Displays histograms for images."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for idx, img in enumerate(images_list):
        axs[idx, 0].imshow(img)
        axs[idx, 0].axis('off')
        axs[idx, 0].set_title(f"Image {idx + 1}")

        axs[idx, 1].hist(img.ravel(), bins=256, color='gray', alpha=0.7)
        axs[idx, 1].set_title(f"Histogram of Image {idx + 1}")
        axs[idx, 1].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images = list(map(
        lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
        (cv2.imread('data/lab_1/6.jpg'), cv2.imread('data/lab_1/7.jpg'))
    ))
    display_hist_plots(images)
