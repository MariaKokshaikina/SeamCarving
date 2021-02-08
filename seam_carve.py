import numpy as np
from tqdm.notebook import tqdm
from PIL import Image
import time
import os
import imageio
import imutils


GIFS_DIR = 'gifs'
IMAGES_DIR = 'images'

def get_importance_map_from_borders(img, upper, bottom, left, right):
    importance_map = np.zeros(np.asarray(img)[:,:,0].shape)
    old_height = np.array(img).shape[1]
    old_width = np.array(img).shape[0]
    object_border_upper = round(old_height * upper)
    object_border_bottom = round(old_height * bottom)
    object_border_left = round(old_width * left)
    object_border_right = round(old_width * right)
    importance_map[object_border_upper:object_border_bottom, object_border_left:object_border_right] = 1
    return importance_map


class SeamCarve:

    def __init__(self,
                 image_path,
                 new_size,
                 importance_map,
                 energy_function,
                 seam_map_function,
                 carve_function,
                 name_suffix):
        
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
        
        self.images_for_gif = []


    def run(self):
        
        img = np.asarray(self.initial_image)
        initial_importance_map = self.importance_map
        
        width_diff = self.original_width - self.new_width
        height_diff = self.original_height - self.new_height
        
        # todo: think about alternate resizing width and height
        
        if width_diff != 0:
            img = self.resize_img_width(img, width_diff, rotated=False)
            
        if height_diff != 0:
            rotated_img = self.rotate_image(img, 90)
            self.importance_map = self.importance_map.T
            img = self.resize_img_width(rotated_img, height_diff, rotated=True)
            img = self.rotate_image(img, -90)
            
        self.final_image = Image.fromarray(img.astype(np.uint8))
        
        
    def rotate_image(self, image, angle):
        
        rotated_img = imutils.rotate_bound(image, angle)
        return rotated_img
    
            
    def get_seam_to_drop(self, img, mask, energy, importance_map):
        
        r,c,_ = img.shape
        
        energy = self.energy_function(img, self.importance_map, mask, energy)
        map_, backtrack = self.seam_map_function(img, energy)
        mask = self.carve_function(img, map_, backtrack)
        
        try:
            energy = energy[mask].reshape((r, c - 1))
        except ValueError:
            energy = energy[mask].reshape((r, c - 1, 5)) # for forward energy
        importance_map = importance_map[mask].reshape((r, c - 1))
        
        return mask, energy, importance_map
    
    
    def resize_img_width(self, img, width_diff, rotated=False):

        energy = None
        mask = None
        importance_map = self.importance_map
        
        if width_diff > 0:

            for i in tqdm(range(width_diff)):

                img_gif = img.copy()
                r,c,_ = img.shape

                start_time = time.clock()
                mask, energy, importance_map = self.get_seam_to_drop(img, mask, energy, importance_map)
                mask = np.stack([mask] * 3, axis=2)
                
                img = img[mask].reshape((r, c - 1, 3))
                self.time_by_step.append(time.clock() - start_time)

                img_gif[mask[:,:,0]==False] = (255,0,0)
                if rotated:
                    self.images_for_gif.extend([self.rotate_image(img_gif, -90)])#, 
#                                                 self.rotate_image(img, -90)])
                else:
                    self.images_for_gif.extend([img_gif])#, img])
            
            self.importance_map = importance_map

        elif width_diff < 0:

            img_tmp = img.copy()
            masks = []
            
            gif0 = np.empty((img_tmp.shape[0], np.abs(width_diff), 3), dtype=np.uint8)
            gif0.fill(255)
            gif0 = np.concatenate((img_tmp, gif0), axis=1)
            if rotated:
                gif0 = self.rotate_image(gif0, -90)
            self.images_for_gif.append(gif0)
            
            for i in tqdm(range(np.abs(width_diff))):

                r,c,_ = img_tmp.shape

                mask, energy, importance_map = self.get_seam_to_drop(img_tmp, mask, energy, importance_map)
                masks.append(mask)
                
                mask = np.stack([mask] * 3, axis=2)
                img_tmp = img_tmp[mask].reshape((r, c - 1, 3))

            while len(masks) > 0:

                current_mask = masks.pop(0)

                img_gif = img.copy()
                img_gif[np.stack([current_mask] * 3, axis=2)[:,:,0]==False] = (255,0,0)
    
                img = self.add_seam_to_img(img, current_mask)
                self.importance_map = self.update_importance_map(current_mask)
            
                if rotated:
                    self.images_for_gif.extend([self.rotate_image(img_gif, -90)])#, 
#                                                 self.rotate_image(img, -90)])
                else:
                    self.images_for_gif.extend([img_gif])#, img])

                masks = self.update_mask(current_mask, masks)

        return img.astype(np.uint8)

    
    def update_importance_map(self, mask):
        
        r,c = self.importance_map.shape
        new_map = np.zeros((r, c+1))

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

        return new_img

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


    def save_result(self, gif_fps=10):

        if not os.path.exists(IMAGES_DIR):
                os.mkdir(IMAGES_DIR)
                
        self.final_image.save(
            f'{IMAGES_DIR}/{self.image_name}'
            f'_w{self.original_width}to{self.new_width}'
            f'_h{self.original_height}to{self.new_height}'
            f'_{self.name_suffix}.jpg')
        
        if not os.path.exists(GIFS_DIR):
            os.mkdir(GIFS_DIR)  
            
        imageio.mimwrite(f'{GIFS_DIR}/{self.image_name}'
                         f'_w{self.original_width}to{self.new_width}'
                         f'_h{self.original_height}to{self.new_height}'
                         f'_{self.name_suffix}.gif',
                         self.images_for_gif, fps=gif_fps)

    def show_gif(self):
        import base64
        from IPython import display
        with open(f'{GIFS_DIR}/{self.image_name}'
                  f'_w{self.original_width}to{self.new_width}'
                  f'_h{self.original_height}to{self.new_height}'
                  f'_{self.name_suffix}.gif', 'rb') as fd:
            b64 = base64.b64encode(fd.read()).decode('ascii')
        return display.HTML(f'<img src="data:image/gif;base64,{b64}" width="300" />')
