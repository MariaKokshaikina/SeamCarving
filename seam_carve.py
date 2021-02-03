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
                 energy_function,
                 seam_map_function,
                 carve_function):
        
        self.initial_image = Image.open(image_path)
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        self.original_width = np.asarray(self.initial_image).shape[1]
        self.new_width = new_width
        
        self.energy_function = energy_function
        self.seam_map_function = seam_map_function
        self.carve_function = carve_function
        
        self.time_by_step = []
        self.final_image = None
        
        self.images_for_gif = []


    def run(self):
        
        img = np.asarray(self.initial_image)
        
        for i in tqdm(range(self.new_width, self.original_width)):
            
            r,c,_ = img.shape
            img_gif = img.copy()
            
            start_time = time.clock()
            energy = self.energy_function(img) 
            map_, backtrack = self.seam_map_function(img, energy)
            mask = self.carve_function(img, map_, backtrack)
            
            img = img[mask].reshape((r, c - 1, 3))
            self.time_by_step.append(time.clock() - start_time)
            
            img_gif[mask[:,:,0]==False] = (255,0,0)
            self.images_for_gif.extend([img_gif, img])

        self.final_image = Image.fromarray(img)                
            
        
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
        