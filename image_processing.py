import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import PIL
from PIL import Image
import os
from os import listdir

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

TODO
Include image compression step to speed up process
compute multiple images in parallel
"""

#colours and colour maps
colours = ['red', 'green', 'blue']
colour_maps = ['Reds', 'Greens', 'Blues']

#declare directory where code will run and images are saved
root_dir = '/home/darraghmoose/University/uni_level_3/phy_3008/group_project' #this folder should have subfolder of images to be process named "/input_images"


#create directory to save outputs to
main_dir = os.path.join(root_dir, 'wildfires_main') #main folder where all data is to be saved
os.makedirs(main_dir, exist_ok=True)

#components and transforms directories
components_dir = os.path.join(main_dir, 'components') #declares where we want to save the iamge components to
transforms_dir = os.path.join(main_dir, 'transforms')
os.makedirs(components_dir, exist_ok=True)
os.makedirs(transforms_dir, exist_ok=True)

#directories for wildifire present & wildfire absent
fire_present_dir_components = os.path.join(components_dir, 'wildfire_present_components') 
fire_present_dir_transforms = os.path.join(transforms_dir, 'wildfire_present_transforms')
os.makedirs(fire_present_dir_components, exist_ok=True) 
os.makedirs(fire_present_dir_transforms, exist_ok=True)

#directories for wildifire present & wildfire absent
fire_absent_dir_components = os.path.join(components_dir, 'wildfire_absent_components')
fire_absent_dir_transforms = os.path.join(transforms_dir, 'wildfire_absent_transforms')
os.makedirs(fire_absent_dir_components, exist_ok=True) 
os.makedirs(fire_absent_dir_transforms, exist_ok=True)

#process all images containing wildfires
present_input_dir = root_dir + "/input_images/wildfire_present"
for image in os.listdir(present_input_dir):
    print(image)

    colour_image = Image.open(f"{present_input_dir}/{image}").convert('RGB')

    #split colour image into RGB components and save to list [red, green, blue]
    images_list = list(Image.Image.split(colour_image))

    #perform fourier transform on all 3 image colour bands
    for i in range(3):
        #perform fft on colour band
        f = fft2(images_list[i])

        #this shifts the zero frequency to the centre and higher frequencies radiate outwards
        fshift = fftshift(np.asanyarray(f)) #np function explicitly marks as array, not image file 

        #converting spectrum to log scale makes details easier to visualise
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        #save R/G/B component of image 
        plt.imsave(os.path.join(fire_present_dir_components, f'{image}_{colours[i]}_components.png'), np.asanyarray(images_list[i]), cmap = 'gray', vmin=0, vmax=255) #np.asanyarray() converts the image explicitly into an array. matplotlib does this at runtime but the ide sees this as an error, np function fixes this

        #save FT of R/G/B component of the image
        plt.imsave(os.path.join(fire_present_dir_transforms, f'{image}_{colours[i]}_transformed.png'), magnitude_spectrum, cmap = colour_maps[i], vmin=0, vmax=255)


#process all images containing NO wildfires
absent_input_dir = root_dir +  "/input_images/wildfire_absent"
for image in os.listdir(absent_input_dir):
    print(image)

    colour_image = Image.open(f"{absent_input_dir}/{image}")

    #get image dimensions
    image_height = colour_image.height
    image_width = colour_image.width

    #compress image for lower runtime
    #colour_image.resize((image_width, image_height), Image.NEAREST)


    #split colour image into RGB components and save to list
    images_list = list(Image.Image.split(colour_image))

    #perform fourier transform on all 3 image colour bands
    for i in range(3):
        #perform fft on colour band
        f = fft2(images_list[i])

        #this shifts the zero frequency to the centre and higher frequencies radiate outwards
        fshift = fftshift(np.asanyarray(f))

        #converting spectrum to log scale makes details easier to visualise
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        #save R/G/B component of image as png
        plt.imsave(os.path.join(fire_absent_dir_components, f'{image}_{colours[i]}_components.png'), np.asanyarray(images_list[i]), cmap = 'gray', vmin=0, vmax=255)

        #save FT of R/G/B component of the image as png
        plt.imsave(os.path.join(fire_absent_dir_transforms, f'{image}_{colours[i]}_transformed.png'), magnitude_spectrum, cmap = colour_maps[i], vmin=0, vmax=255) 
