import numpy as np
from scipy import signal
from skimage import filters, color
import math

IMPORTANCE_COEF = 100000


def color_to_gray(img):
    return (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0


def energy_gradient_for_i_j(img, i, j, importance_map):
    height = img.shape[0]
    width = img.shape[1]
    L = img[i, (j - 1) % width]
    R = img[i, (j + 1) % width]
    U = img[(i - 1) % height, j]
    D = img[(i + 1) % height, j]

    dx_sq = np.sum((R - L) ** 2)
    dy_sq = np.sum((D - U) ** 2)
    return np.sqrt(dx_sq + dy_sq) + importance_map[i, j] * IMPORTANCE_COEF


def energy_gradient(img, importance_map, mask, old_energy):
    height = img.shape[0]
    width = img.shape[1]
    if(old_energy is None):
        energy = np.empty((height, width))
    else:
        energy = old_energy
    if(not mask is None):
        indices = np.where(mask[:,:,0] == False)
        for i in range(len(indices[1])):
            for j in range(max(0,indices[1][i]-2), min(indices[1][i]+2, width)):
                energy[i,j] = energy_gradient_for_i_j(img, i, j, importance_map)
    else:
        for i in range(height):
            for j in range(width):
                energy[i,j] = energy_gradient_for_i_j(img, i, j, importance_map)
    return energy


def get_value_by_index_or_zero(array, i, j):
    if(i >= array.shape[0] or j >= array.shape[1] or i < 0 or j < 0):
        return 0
    else:
        return array[i,j]


def sobel_operator_for_i_j(img, importance_map, i, j, filter):
    result_x = 0
    result_y = 0
    r, c = filter.shape
    for ii in range(r):
        for jj in range(c):
            result_x += get_value_by_index_or_zero(img, i + ii - r + 1, j + jj - c +1) * filter[ii, jj]
            result_y += get_value_by_index_or_zero(img, i + ii - r + 1, j + jj - c +1) * filter.T[ii, jj]

    return math.sqrt(result_x**2 + result_y**2)+importance_map[i,j]*IMPORTANCE_COEF


def gradient_magnitude_sobel_operator(img, importance_map, mask, old_energy):
    image = color_to_gray(img)
    filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    width = img.shape[1]
    if old_energy is None:
        new_image_x = signal.convolve2d(image, filter)[1:-1, 1:-1]
        new_image_y = signal.convolve2d(image, np.flip(filter.T, axis=0))[1:-1, 1:-1]
        gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    else:
        energy = old_energy
        indices = np.where(mask[:, :, 0] == False)
        for i in range(len(indices[1])):
            for j in range(max(0, indices[1][i] - 2), min(indices[1][i] + 2, width)):
                energy[i, j] = sobel_operator_for_i_j(image, importance_map, i, j, filter)
        gradient_magnitude = energy  # np.sqrt(np.square(energy_x) + np.square(energy_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    return gradient_magnitude

def backward_energy(img,  importance_map,  mask, old_energy):
    return filters.sobel(color.rgb2gray(img))


def energy_function_forward(img,  importance_map,  mask, old_energy):
    return None
