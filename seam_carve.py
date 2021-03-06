import numpy as np
from PIL import Image
import time
import os
import imageio
import imutils
from multiprocessing import Pool

from utils import get_config

config = get_config()

GIFS_DIR = config['gifs_dir']
IMAGES_DIR = config['output_dir']

POOL_PROCESSES = config['pool_processes']


def get_importance_map_from_borders(img, upper, bottom, left, right):
    importance_map = np.zeros(np.asarray(img)[:, :, 0].shape)
    old_height = np.array(img).shape[1]
    old_width = np.array(img).shape[0]
    object_border_upper = round(old_height * upper)
    object_border_bottom = round(old_height * bottom)
    object_border_left = round(old_width * left)
    object_border_right = round(old_width * right)
    importance_map[object_border_upper:object_border_bottom, object_border_left:object_border_right] = 1
    return importance_map


def rotate_image(image, angle):

    rotated_img = imutils.rotate_bound(image, angle)
    return rotated_img


class SeamCarve:

    def __init__(self,
                 image_path,
                 new_size,
                 importance_map,
                 energy_function,
                 seam_map_function,
                 carve_function,
                 name_suffix,
                 save_gif=False):

        self.initial_image = Image.open(image_path)
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]

        self.original_height = np.asarray(self.initial_image).shape[0]
        self.new_height = new_size[0]

        self.original_width = np.asarray(self.initial_image).shape[1]
        self.new_width = new_size[1]

        if importance_map is None:
            importance_map = np.zeros((self.original_height, self.original_width))
        self.importance_map = importance_map

        self.energy_function = energy_function
        self.seam_map_function = seam_map_function
        self.carve_function = carve_function

        self.name_suffix = name_suffix

        self.time_by_step = []
        self.final_image = None

        self.save_gif = save_gif
        self.images_for_gif = []
        self.pool = Pool(processes=POOL_PROCESSES)

    def run(self):

        img = np.asarray(self.initial_image)

        if self.save_gif:
            gif0 = np.empty((np.max([self.new_height, self.original_height]),
                             np.max([self.new_width, self.original_width]), 3), dtype=np.uint8)
            gif0.fill(255)
            gif0[:self.original_height, :self.original_width] = img
            self.images_for_gif.append(gif0)

        width_diff = self.original_width - self.new_width
        height_diff = self.original_height - self.new_height

        # todo: think about alternate resizing width and height

        if width_diff != 0:
            img = self.resize_img_width(img, width_diff, rotated=False)

        if height_diff != 0:
            rotated_img = rotate_image(img, 90)
            self.importance_map = rotate_image(self.importance_map, 90)
            img = self.resize_img_width(rotated_img, height_diff, rotated=True)
            img = rotate_image(img, -90)

        self.final_image = Image.fromarray(img.astype(np.uint8))

    def run_with_multiple_seam_drop(self):
        img = np.asarray(self.initial_image)

        gif0 = np.empty((np.max([self.new_height, self.original_height]),
                         np.max([self.new_width, self.original_width]), 3), dtype=np.uint8)
        gif0.fill(255)
        gif0[:self.original_height, :self.original_width] = img
        self.images_for_gif.append(gif0)

        width_diff = self.original_width - self.new_width
        height_diff = self.original_height - self.new_height

        # todo: think about alternate resizing width and height

        if width_diff != 0:
            img = self.resize_img_width(img, width_diff, rotated=False)

        if height_diff != 0:
            rotated_img = rotate_image(img, 90)
            self.importance_map = self.importance_map.T
            img = self.resize_img_width(rotated_img, height_diff, rotated=True)
            img = rotate_image(img, -90)

        self.final_image = Image.fromarray(img.astype(np.uint8))

    def resize_img_width_with_multiple_seam_drop(self, img, width_diff, rotated=False):

        energy = None
        mask = None
        importance_map = self.importance_map

        if width_diff > 0:
            seams_removed = 0

            while(seams_removed < width_diff):

                img_gif = img.copy()

                try:
                    start_time = time.clock()
                except AttributeError:
                    start_time = time.time()

                energy_map = self.energy_function(img, self.importance_map, mask, energy, self.pool)
                for i in range(seams_removed // 10):
                    map_, backtrack = self.seam_map_function(img, energy_map)
                    mask = self.carve_function(img, map_, backtrack)
                    img, energy, importance_map = self.drop_seam(img, mask, energy, importance_map)

                    try:
                        self.time_by_step.append(time.clock() - start_time)
                    except AttributeError:
                        self.time_by_step.append(time.time() - start_time)

                    img_gif[np.stack([mask] * 3, axis=2)[:, :, 0] == False] = (255, 0, 0)

                    if rotated:
                        self.images_for_gif.append(rotate_image(img_gif, -90))
                    else:
                        self.images_for_gif.append(img_gif)

            self.importance_map = importance_map

        elif width_diff < 0:

            img_tmp = img.copy()
            masks = []

            for i in range(np.abs(width_diff)):

                mask, energy = self.get_seam_to_drop(img_tmp, mask, energy)
                masks.append(mask)
                img_tmp, energy, importance_map = self.drop_seam(img_tmp, mask, energy, importance_map)

            while len(masks) > 0:

                current_mask = masks.pop(0)

                img_gif = img.copy()
                img_gif[np.stack([current_mask] * 3, axis=2)[:, :, 0] == False] = (255, 0, 0)

                img = self.add_seam_to_img(img, current_mask)
                self.importance_map = self.update_importance_map(current_mask)

                if rotated:
                    self.images_for_gif.append(rotate_image(img_gif, -90))
                else:
                    self.images_for_gif.append(img_gif)

                masks = self.update_mask(current_mask, masks)

        return img.astype(np.uint8)

    def find_low_energy_window(self, energy, window_size):
        min_window_starts_from = 0
        min_energy = 10000000000
        for i in range(0, energy.shape[1] - window_size):
            curr_energy = np.sum(energy[:,i:i+window_size])
            if curr_energy < min_energy:
                min_energy = curr_energy
                min_window_starts_from = i
        return min_window_starts_from


    def get_seam_to_drop(self, img, prev_mask, prev_energy):

        energy_map = self.energy_function(img, self.importance_map, prev_mask, prev_energy, self.pool)
        map_, backtrack = self.seam_map_function(img, energy_map)
        mask = self.carve_function(img, map_, backtrack)

        return mask, energy_map

    def drop_seam(self, img, mask, energy, importance_map):

        r, c, _ = img.shape

        try:
            energy = energy[mask].reshape((r, c - 1))
        except (ValueError, TypeError):
            energy = energy[mask].reshape((r, c - 1, 5))  # for forward energy, todo: come up with smth better

        importance_map = importance_map[mask].reshape((r, c - 1))

        mask = np.stack([mask] * 3, axis=2)
        img_new = img[mask].reshape((r, c - 1, 3))

        return img_new, energy, importance_map

    def resize_img_width(self, img, width_diff, rotated=False):

        energy = None
        mask = None
        importance_map = self.importance_map

        if width_diff > 0:

            for i in range(width_diff):

                if self.save_gif:
                    img_gif = img.copy()

                try:
                    start_time = time.clock()
                except AttributeError:
                    start_time = time.time()

                mask, energy = self.get_seam_to_drop(img, mask, energy)
                img, energy, importance_map = self.drop_seam(img, mask, energy, importance_map)

                try:
                    self.time_by_step.append(time.clock() - start_time)
                except AttributeError:
                    self.time_by_step.append(time.time() - start_time)

                if self.save_gif:
                    img_gif[np.stack([mask] * 3, axis=2)[:, :, 0] == False] = (255, 0, 0)

                    if rotated:
                        self.images_for_gif.append(rotate_image(img_gif, -90))
                    else:
                        self.images_for_gif.append(img_gif)

            self.importance_map = importance_map

        elif width_diff < 0:

            img_tmp = img.copy()
            masks = []

            for i in range(np.abs(width_diff)):

                mask, energy = self.get_seam_to_drop(img_tmp, mask, energy)
                masks.append(mask)
                img_tmp, energy, importance_map = self.drop_seam(img_tmp, mask, energy, importance_map)

            while len(masks) > 0:

                current_mask = masks.pop(0)

                if self.save_gif:
                    img_gif = img.copy()
                    img_gif[np.stack([current_mask] * 3, axis=2)[:, :, 0] == False] = (255, 0, 0)

                img = self.add_seam_to_img(img, current_mask)
                self.importance_map = self.update_importance_map(current_mask)

                if self.save_gif:
                    if rotated:
                        self.images_for_gif.append(rotate_image(img_gif, -90))
                    else:
                        self.images_for_gif.append(img_gif)

                masks = self.update_mask(current_mask, masks)

        return img.astype(np.uint8)

    def update_importance_map(self, mask):

        r, c = self.importance_map.shape
        new_map = np.zeros((r, c + 1))

        for row in range(r):
            col = np.where(mask[row] == False)[0][0]
            if col == 0:
                new_map[row, col] = self.importance_map[row, col]
                new_map[row, col + 1] = int(np.mean(self.importance_map[row, col: col + 2]))
                new_map[row, col + 1:] = self.importance_map[row, col:]
            else:
                new_map[row, : col] = self.importance_map[row, : col]
                new_map[row, col] = int(np.mean(self.importance_map[row, col - 1: col + 1]))
                new_map[row, col + 1:] = self.importance_map[row, col:]

        return new_map

    def add_seam_to_img(self, img, mask):

        initial_img = img.copy()
        r, c, g = initial_img.shape
        new_img = np.zeros((r, c + 1, g))

        for row in range(r):
            col = np.where(mask[row] == False)[0][0]
            for rgb in range(g):
                if col == 0:
                    new_img[row, col, rgb] = initial_img[row, col, rgb]
                    new_img[row, col + 1, rgb] = int(np.mean(initial_img[row, col: col + 2, rgb]))
                    new_img[row, col + 1:, rgb] = initial_img[row, col:, rgb]
                else:
                    new_img[row, : col, rgb] = initial_img[row, : col, rgb]
                    new_img[row, col, rgb] = int(np.mean(initial_img[row, col - 1: col + 1, rgb]))
                    new_img[row, col + 1:, rgb] = initial_img[row, col:, rgb]

        return new_img

    def update_mask(self, current_mask, masks_left):

        new_masks = []

        curr_mask_col = np.where(current_mask == False)[1]

        for next_mask in masks_left:
            r, c = next_mask.shape
            next_mask_col = np.where(next_mask == False)[1]
            next_mask_col[np.where(next_mask_col >= curr_mask_col)] += 2

            new_mask = np.ones((r, c + 2), dtype=bool)
            x, y = np.transpose([(i, int(j)) for i, j in enumerate(next_mask_col)])
            new_mask[x, y] = False

            new_masks.append(new_mask)

        return new_masks

    def get_new_img_filename(self):
        return f'{IMAGES_DIR}/{self.image_name}_w{self.original_width}to{self.new_width}_h{self.original_height}to{self.new_height}_{self.name_suffix}.jpg'

    def get_gif_filename(self):
        return f'{GIFS_DIR}/{self.image_name}_w{self.original_width}to{self.new_width}_h{self.original_height}to{self.new_height}_{self.name_suffix}.gif'

    def save_result(self, gif_fps=10):

        if not os.path.exists(IMAGES_DIR):
            os.mkdir(IMAGES_DIR)

        img_filename = self.get_new_img_filename()
        self.final_image.save(img_filename)

        if self.save_gif:
            if not os.path.exists(GIFS_DIR):
                os.mkdir(GIFS_DIR)

            gif_filename = self.get_gif_filename()
            imageio.mimwrite(gif_filename, map(lambda x: x.astype(np.uint8), self.images_for_gif), fps=gif_fps)
        else:
            gif_filename = None

        return img_filename, gif_filename

    def show_gif(self):
        import base64
        from IPython import display
        with open(f'{GIFS_DIR}/{self.image_name}'
                  f'_w{self.original_width}to{self.new_width}'
                  f'_h{self.original_height}to{self.new_height}'
                  f'_{self.name_suffix}.gif', 'rb') as fd:
            b64 = base64.b64encode(fd.read()).decode('ascii')
        return display.HTML(f'<img src="data:image/gif;base64,{b64}" width="300" />')
