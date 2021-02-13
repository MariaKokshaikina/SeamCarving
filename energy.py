import numpy as np
from scipy import signal
from skimage import filters, color
from numba import jit


IMPORTANCE_COEF = 100000

#@jit
def color_to_gray(img):
    return (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0

#@jit
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


def energy_gradient_for_i_j_proxy(args):
    return energy_gradient_for_i_j(*args), args


def energy_gradient(img, importance_map, mask, old_energy, pool):
    height = img.shape[0]
    width = img.shape[1]
    if old_energy is None:
        energy = np.empty((height, width))
    else:
        energy = old_energy
    if mask is not None:
        indices = np.where(mask == False)
        a = [
            (img, i, j, importance_map)
            for i in range(len(indices[1]))
            for j in range(max(0, indices[1][i] - 2), min(indices[1][i] + 2, width))
        ]
        result = pool.map(energy_gradient_for_i_j_proxy, a)
        for res, args in result:
            i, j = args[1], args[2]
            energy[i, j] = res
        #for i in range(len(indices[1])):
        #    for j in range(max(0, indices[1][i] - 2), min(indices[1][i] + 2, width)):
        #        energy[i, j] = energy_gradient_for_i_j(img, i, j, importance_map)
    else:
        a = [
           (img, i, j, importance_map)
            for i in range(height)
            for j in range(width)
        ]
        result = pool.map(energy_gradient_for_i_j_proxy, a)
        for res, args in result:
            i, j = args[1], args[2]
            energy[i, j] = res
    return energy


def get_value_by_index_or_zero(array, i, j):
    if i >= array.shape[0] or j >= array.shape[1] or i < 0 or j < 0:
        return 0
    else:
        return array[i, j]


def sobel_operator_for_i_j(img, importance_map, i, j):
    gx = (get_value_by_index_or_zero(img,i - 1,j - 1) + 2 * get_value_by_index_or_zero(img,i,j - 1) + get_value_by_index_or_zero(img,i + 1,j - 1)) - (
                get_value_by_index_or_zero(img,i - 1,j + 1) + 2 * get_value_by_index_or_zero(img,i,j + 1) + get_value_by_index_or_zero(img,i + 1,j + 1))
    gy = (get_value_by_index_or_zero(img,i - 1,j - 1) + 2 * get_value_by_index_or_zero(img,i - 1,j) + get_value_by_index_or_zero(img,i - 1,j + 1)) - (
                get_value_by_index_or_zero(img,i + 1,j - 1) + 2 * get_value_by_index_or_zero(img,i + 1,j) + get_value_by_index_or_zero(img,i + 1,j + 1))
    return np.sqrt(gx ** 2 + gy ** 2) + importance_map[i, j] * IMPORTANCE_COEF


def gradient_magnitude_sobel_operator(img, importance_map, mask, old_energy, pool):
    image = color_to_gray(img)
    filter_ = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    width = img.shape[1]
    if old_energy is None:
        new_image_x = signal.convolve2d(image, filter_)[1:-1, 1:-1]
        new_image_y = signal.convolve2d(image, np.flip(filter_.T, axis=0))[1:-1, 1:-1]
        gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    else:
        energy = old_energy
        indices = np.where(mask == False)
        for i in range(len(indices[1])):
            for j in range(max(0, indices[1][i] - 2), min(indices[1][i] + 2, width)):
                energy[i, j] = sobel_operator_for_i_j(image, importance_map, i, j)
        gradient_magnitude = energy  # np.sqrt(np.square(energy_x) + np.square(energy_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    return gradient_magnitude


def backward_energy(img, importance_map, mask, old_energy, pool):
    return filters.sobel(color.rgb2gray(img))

#@jit
def energy_function_forward(img, importance_map, mask, old_energy, pool):
    cache = {}  # todo: check if it's not useless

    def D(x0, y0, x1, y1):
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        elif x0 == x1 and y0 > y1:
            y0, y1 = y1, y0
        key = (x0, y0, x1, y1)
        if key not in cache:
            val = np.sum(np.power(img[y0 % height, x0 % width] - img[y1 % height, x1 % width], 2))
            cache[key] = val
        return cache[key]

    def get_forward_energy_for_x_y(x0, y0):
        return np.array([
            D(x0, y0, x1, y1) + imp_cost
            for (x1, y1) in [
                (x0 - 1, y0 + 1),  # 0
                (x0, y0 + 1),  # 1
                (x0 + 1, y0),  # 2
                (x0 + 1, y0 + 1),  # 3
                (x0 + 2, y0),  # 4
            ]
        ])

    height = img.shape[0]
    width = img.shape[1]
    if old_energy is None:
        energy = np.empty((height, width, 5))
    else:
        energy = old_energy
    if mask is not None:
        indices = np.where(mask == False)
        for y0 in range(len(indices[1])):
            for x0 in range(max(0, indices[1][y0] - 2), min(indices[1][y0] + 2, width)):  # todo: check
                imp_cost = importance_map[y0, x0] * IMPORTANCE_COEF
                energy[y0, x0] = get_forward_energy_for_x_y(x0, y0)
    else:
        for y0 in range(height):
            for x0 in range(width):
                imp_cost = importance_map[y0, x0] * IMPORTANCE_COEF
                energy[y0, x0] = get_forward_energy_for_x_y(x0, y0)

    return energy
