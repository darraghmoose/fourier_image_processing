import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import PIL
from PIL import Image
import os
from os import listdir

"""
PARALLELISATION:
define all functions
create main loop to call functions which will actually execute the code
"""

#fixed parameters for the code

#colours and colour maps
colours = ['red', 'green', 'blue']
colour_maps = ['Reds', 'Greens', 'Blues']

#declare directory where code will run and images are saved
root_dir = '/home/darraghmoose/University/uni_level_3/phy_3008/group_project' #this folder should have subfolder of images to be process named "/input_images"

#create directory to save outputs to
main_dir = os.path.join(root_dir, 'wildfires_main') 





def process_image(inputs):
    "In order to use pool map for parallel processing the function needs to only have one input. This is a tuple of the 4 terms below"
    input_path, name, components_dir, transforms_dir = inputs

    #open image from directory given as input and ensure its saved as rgb
    colour_image = Image.open(f"{input_path}").convert('RGB')
    
    #split colour image into RGB components and save to list [red, green, blue]
    colour_bands = colour_image.split()

    #perform fourier transform on all 3 image colour bands
    for i in range(3):
        #save colour component to appropriate directory (not using plt.imsave as its slow)
        colour_bands[i].save(os.path.join(components_dir, f'{name}_{colours[i]}_components.png'))

        #perform fft on colour band and shift the zero frequency to the centre. Now higher frequencies radiate outwards
        fourier_input = np.asarray(colour_bands[i])
        fshift = fftshift(fft2(fourier_input)) #type:ignore  -  This comments gets rid of false error due to type misunderstanding 

        #converting spectrum to log scale makes details easier to visualise
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        #save FT of R/G/B component of the image
        plt.imsave(os.path.join(transforms_dir, f'{name}_{colours[i]}_transformed.png'), magnitude_spectrum, cmap = colour_maps[i], vmin=0, vmax=255)
    









if __name__ == '__main__':
    print('hey')