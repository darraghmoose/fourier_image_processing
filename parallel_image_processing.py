import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from tqdm import tqdm
from PIL import Image
import os
from multiprocessing import Pool

"""
Folder Structure 

#Input images
|   Wildfire present
|   Wildfires NOT present

#Main 
|   Components
|   |   Wildfire present
|   |   Wildfires NOT present
|
|   Transforms
|   |   Wildfire present
|   |   Wildfires NOT present
"""

#colours and colour maps
colours = ['red', 'green', 'blue']
colour_maps = ['Reds', 'Greens', 'Blues']

#declare directory where code will run and images are saved
root_dir = '/home/darraghmoose/University/uni_level_3/phy_3008/group_project' #this folder should have subfolder of images to be process named "/input_images"
#create directory to save outputs to
main_dir = os.path.join(root_dir, 'wildfires_main') 


def process_image(jobs):
    "In order to use pool map for parallel processing the function needs to only have one input. This is a tuple of the 4 terms below"
    input_path, name, components_dir, transforms_dir = jobs

    #open image from directory given as input and ensure its saved as rgb
    colour_image = Image.open(f"{input_path}").convert('RGB')
    #compress the image
    colour_image.thumbnail((1024, 1024))
    
    #split colour image into RGB components and save to list [red, green, blue]
    colour_bands = colour_image.split()

    #perform fourier transform on all 3 image colour bands
    for i in range(3):
        #save colour component to appropriate directory (not using plt.imsave as its slow)
        colour_bands[i].save(os.path.join(components_dir, f'{name}_{colours[i]}_components.png'))

        #perform fft on colour band and shift the zero frequency to the centre. Now higher frequencies radiate outwards
        fourier_input = np.asarray(colour_bands[i], dtype=np.float32)
        fshift = fftshift(fft2(fourier_input)) #type:ignore  -  This comments gets rid of false error due to type misunderstanding 

        #converting spectrum to log scale makes details easier to visualise
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        #save image of transformed component
        cmap = plt.get_cmap(colour_maps[i])     #type: ignore
        rgba = cmap(magnitude_spectrum / 255)   #convert to rgb image with alpha channel (rgba)
        picture = (rgba[:, :, :3] * 255).astype(np.uint8)   #drop alpha channel and save in np.uint8 form which PIL can handle
        Image.fromarray(picture).save(os.path.join(transforms_dir, f'{name}_{colours[i]}_transformed.png'), compress_level=1) #light compression and very fast
    

def create_jobs_list():
    jobs = []
    #populate jobs path as (input_path, name, components_dir, transforms_dir)
    for flag in ['wildfire_present', 'wildfire_absent']:
        #components and transforms directories
        components_dir = os.path.join(main_dir, 'components', f'{flag}_components') #declares where we want to save the image components to
        transforms_dir = os.path.join(main_dir, 'transforms', f'{flag}_transforms')
        os.makedirs(components_dir, exist_ok=True)
        os.makedirs(transforms_dir, exist_ok=True)

        #find directory containing all input images
        input_dir = os.path.join(root_dir, 'input_images', flag)
        #add all images in input directories to jobs list
        for name in os.listdir(input_dir):
            jobs.append((os.path.join(input_dir, name), name, components_dir, transforms_dir))
    return jobs


#Run the code in parallel
if __name__ == '__main__':
    jobs = create_jobs_list()
    with Pool(processes = 4) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, jobs), total=len(jobs)):
            pass