import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from tqdm import tqdm
from PIL import Image
import os
from multiprocessing import Pool

#colours and colour maps
colours = ['red', 'green', 'blue']
colour_maps = ['Reds', 'Greens', 'Blues']

#declare directory where code will run and images are saved
root_dir = '/home/darraghmoose/University/uni_level_3/phy_3008/group_project' #this folder should have subfolder of images to be process named "/input_images"
#create directory to save outputs to
main_dir = os.path.join(root_dir, 'wildfires_main') 


def process_image(jobs):
    "In order to use pool map for parallel processing, the function needs to only have one input. This is a tuple of the 4 terms below"
    input_path, name, outputs_dir = jobs

    #open image from directory given as input and ensure its saved as rgb
    colour_image = Image.open(f"{input_path}").convert('RGB')
    #compress the image
    colour_image.thumbnail((1024, 1024))
    
    #split colour image into RGB components and save to list [red, green, blue]
    colour_bands = colour_image.split()

    #emply list to store transformed colour components in
    transformed_images = []
    
    #perform fourier transform on all 3 image colour bands
    for colour in range(3):
        #perform fft on colour band and shift the zero frequency to the centre. Now higher frequencies radiate outwards
        fourier_input = np.asarray(colour_bands[colour], dtype=np.float32)
        fshift = fftshift(fft2(fourier_input)) #type:ignore  -  This comments gets rid of false error flag  

        #converting spectrum to log scale makes details easier to visualise
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        #scale between 0-255 and convert to 8-bit so PIL can procces image
        scaled = np.clip(magnitude_spectrum, 0, 255).astype(np.uint8)

        #save as a PIL image
        transformed_images.append(Image.fromarray(scaled, mode='L'))

    #merge rgb components into one image
    merged_image = Image.merge('RGB', transformed_images)

    #save merged image to diractory of all transforms
    merged_image.save(os.path.join(outputs_dir, f'{name}_transformed.png'))


#create job list of all images to be processed 
def create_jobs_list():
    jobs =[]
    #populate jobs path
    for flag in ['wildfire', 'no_wildfire']:    #enter names of image directories here
        #create output directory
        output_dir = os.path.join(main_dir, f'{flag}_transformed_images')
        os.makedirs(output_dir, exist_ok=True)

        #find directory containing all input images
        input_dir = os.path.join(root_dir, 'input_images', flag)

        #add all images in input directories to jobs list
        for name in os.listdir(input_dir):
            jobs.append((os.path.join(input_dir, name), name, output_dir))

    return jobs


#Run the code in parallel
if __name__ == '__main__':
    jobs = create_jobs_list()
    with Pool(processes = 4) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, jobs), total=len(jobs),ncols=100, ascii=" ░▒█", desc="Processing images", colour="#00bb82"):
            pass

#expected ~8 minutes for 30,000 350x350 images (1.1GB)
#processing speed ~73MB/s