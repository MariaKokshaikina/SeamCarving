import numpy as np
from scipy import signal

IMPORTANCE_COEF = 100000

def color_to_gray(img):
    return (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0


def energy_gradient(img, importance_map):
    height = img.shape[0]
    width = img.shape[1]
    energy = np.empty((height, width))
    for i in range(height):
        for j in range(width):
            L = img[i, (j - 1) % width]
            R = img[i, (j + 1) % width]
            U = img[(i - 1) % height, j]
            D = img[(i + 1) % height, j]

            dx_sq = np.sum((R - L) ** 2)
            dy_sq = np.sum((D - U) ** 2)
            energy[i, j] = np.sqrt(dx_sq + dy_sq) + importance_map[i, j] * IMPORTANCE_COEF
    return energy


def gradient_magnitude_sobel_operator(img, importance_map):
    image = color_to_gray(img)
    filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    new_image_x = signal.convolve2d(image, filter)[1:-1,1:-1]
    new_image_y = signal.convolve2d(image, np.flip(filter.T, axis=0))[1:-1,1:-1]
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    height = image.shape[0]
    width = image.shape[1]
    energy = np.empty((height, width))
    for i in range(height):
        for j in range(width):
            gradient_magnitude[i, j] += importance_map[i, j] * IMPORTANCE_COEF


    return gradient_magnitude


def energy_function_forward(img,  importance_map):
    return None
