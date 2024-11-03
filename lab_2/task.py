import cv2
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops


PATCH_SIZE = 30

image = cv2.imread('data/image1.jpg', cv2.IMREAD_GRAYSCALE)

surface_coordinates = [
    (3000, 2000),
    (3063, 2701),
    (4232, 2035),
    (4333, 230),
    (2897, 1725)
]

water_coordinates = [
    (3766, 798),
    (3833, 2973),
    (4230, 3304),
    (2767, 1247),
    (3594, 1601)
]


def collect_patches(input_image, coordinates: list[tuple[int, int]], patch_size: int = PATCH_SIZE) -> list:
    """Collect patches based on their coordinates and patch size"""
    return [
        input_image[x: x + patch_size, y: y + patch_size]
        for x, y in coordinates
    ]


def display_patches(patches: list, name: str, num: int = 1) -> None:
    for i, patch in enumerate(patches):
        ax = fig.add_subplot(3, len(patches), len(patches) * num + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax.set_xlabel(f"{name} {i + 1}")


surface_patches = collect_patches(image, surface_coordinates, PATCH_SIZE)

water_patches = collect_patches(image, water_coordinates, PATCH_SIZE)

xs = []
ys = []
for patch in [*surface_patches, *water_patches]:
    glcm = graycomatrix(
        patch, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
    )
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(graycoprops(glcm, 'homogeneity')[0, 0])

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)

for x, y in surface_coordinates:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'ro')
for x, y in water_coordinates:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bo')
ax.set_xlabel('Base image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[: len(surface_patches)], ys[: len(surface_patches)], 'rs', label='Surface')
ax.plot(xs[len(surface_patches):], ys[len(surface_patches):], 'bs', label='Water')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM homogeneity')
ax.legend()

display_patches(surface_patches, 'Surface')
display_patches(water_patches, 'Water', num=2)


fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()