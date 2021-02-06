import numpy as np
from tqdm.notebook import tqdm
from PIL import Image
import time
import os
import imageio


GIFS_DIR = 'gifs'
IMAGES_DIR = 'images'


class SeamCarve:
    
    def __init__(self,
                 image_path,
                 new_width,
                 importance_map,
                 energy_function,
                 seam_map_function,
                 carve_function):
        
        self.initial_image = Image.open(image_path)
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        self.original_width = np.asarray(self.initial_image).shape[1]
        self.new_width = new_width
        
        self.importance_map = importance_map
        self.energy_function = energy_function
        self.seam_map_function = seam_map_function
        self.carve_function = carve_function
        
        self.time_by_step = []
        self.final_image = None
        
        self.images_for_gif = []


    def run(self):
        
        img = np.asarray(self.initial_image)
        
        width_diff = self.original_width - self.new_width
        
        if width_diff >= 0:
        
            for i in tqdm(range(self.new_width, self.original_width)):
                
                img_gif = img.copy()
                r,c,_ = img.shape

                start_time = time.clock()
                energy = self.energy_function(img, self.importance_map)
                map_, backtrack = self.seam_map_function(img, energy)
                mask = self.carve_function(img, map_, backtrack)
                mask = np.stack([mask] * 3, axis=2)
                
                img = img[mask].reshape((r, c - 1, 3))
                self.time_by_step.append(time.clock() - start_time)

                img_gif[mask[:,:,0]==False] = (255,0,0)
                self.images_for_gif.extend([img_gif, img])

            self.final_image = Image.fromarray(img)                
        
        elif width_diff < 0:
            
            img_tmp = img.copy()
            masks = []
            
            gif0 = np.empty((img_tmp.shape[0], np.abs(width_diff), 3), dtype=np.uint8)
            gif0.fill(255)
            gif0 = np.concatenate((img_tmp, gif0), axis=1)
            self.images_for_gif.append(gif0)
            
            
            for i in tqdm(range(2 * self.original_width - self.new_width, self.original_width)):

                r,c,_ = img_tmp.shape
                
                energy = self.energy_function(img_tmp, self.importance_map)
                map_, backtrack = self.seam_map_function(img_tmp, energy)
                mask = self.carve_function(img_tmp, map_, backtrack)
                masks.append(mask)

                mask = np.stack([mask] * 3, axis=2)
                img_tmp = img_tmp[mask].reshape((r, c - 1, 3))

            while len(masks) > 0:

                current_mask = masks.pop(0)

                img_gif = img.copy()
                img_gif[np.stack([current_mask] * 3, axis=2)[:,:,0]==False] = (255,0,0)

                img = self.add_seam_to_img(img, current_mask)
                self.images_for_gif.extend([img_gif, img])

                masks = self.update_mask(current_mask, masks)
            
            self.final_image = Image.fromarray(img)
    
    
    def add_seam_to_img(self, img, mask):
    
        initial_img = img.copy()
        r,c,g = initial_img.shape
        new_img = np.zeros((r, c+1, g))

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

        return new_img.astype(np.uint8)
    
    
    def update_mask(self, current_mask, masks_left):
    
        new_masks = []

        curr_mask_col = np.where(current_mask == False)[1]

        for next_mask in masks_left:

            r, c = next_mask.shape
            next_mask_col = np.where(next_mask == False)[1]
            next_mask_col[np.where(next_mask_col >= curr_mask_col)] += 2

            new_mask = np.ones((r, c+2), dtype=bool)
            x, y = np.transpose([(i, int(j)) for i, j in enumerate(next_mask_col)])
            new_mask[x,y] = False

            new_masks.append(new_mask)

        return new_masks
    
    
    def save_result(self, name_suffix, gif_fps=10):
        
        if not os.path.exists(IMAGES_DIR):
                os.mkdir(IMAGES_DIR)
                
        self.final_image.save(
            f'{IMAGES_DIR}/{self.image_name}_'
            f'w{self.original_width}to{self.new_width}'
            f'_{name_suffix}.jpg')
        
        if not os.path.exists(GIFS_DIR):
            os.mkdir(GIFS_DIR)  
            
        imageio.mimwrite(f'{GIFS_DIR}/{self.image_name}'
                         f'_w{self.original_width}to{self.new_width}'
                         f'_{name_suffix}.gif', 
                         self.images_for_gif, fps=gif_fps)
        