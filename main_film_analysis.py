from skimage import io, transform, draw, filter
import numpy as np
import libconf
import matplotlib.pyplot as plt
from class_film_analysis import film_analysis

"""
The script works out the survival function from the raw images of the experiment.
The first image is displayed, after the threshold, the rotation and the cropping.
In addition, the measuring lines are plotted on the first image to check by eye
the validity of the analysis.
Then, the code analyses all the images before saving the survival_function and
the corresponding time in a text file.
"""

# Data configuration
with open('./config_file.cfg') as cfg:
	config = libconf.load(cfg)

exp_id = config['exp_id'] # Experiment name/ID
fileroot = config['fileroot'] # Data location without image number
extension = config['extension'] # Image format
n_image = config['n_image'] # Number of images to analyse
incr_image = config['incr_image'] # Image increment
time_conv = config['time_conv'] # Time conversion
time_unit = config['time_unit'] # Time unit
err = config['err'] # Horizontal error on the detection lines
n_tube = config['n_tube'] # Number of tubes in the experiment
w_min = config['w_min'] # Min width of the peaks
w_max = config['w_max'] # Max width of the peaks
v_gap = config['v_gap'] # Mumber of pixels between the 3 detection lines
h_position = config['h_position'] # Horizontal position of the tubes
rotate_angle = config['rotate_angle'] # Angle to get the image horizontal
top = config['top'] # Start pixel left
bottom = config['bottom'] # End pixel right
left = config['left'] # Start pixel bottom
right = config['right'] # End pixel top

# Object calling
film_ana = film_analysis(fileroot, extension, h_position, err, n_tube, w_min, w_max, v_gap)

# Arrays creation
survival_function = []
time = []

# Display first the image after rotation, cropping, filtering and with the position of the measuring lines
filename = film_ana.open_file_4_digits(0)
image = io.imread(filename)
image = transform.rotate(image, rotate_angle, resize=True)
image = image[top:bottom, left:right]
thresh = filter.threshold_otsu(image)
binary_img = image > thresh
for i in range(n_tube):
	binary_img[draw.line(0, h_position[i], image.shape[0]-1, h_position[i])] = 1
	binary_img[draw.line(0, h_position[i]+v_gap, image.shape[0]-1, h_position[i]+v_gap)] = 1
	binary_img[draw.line(0, h_position[i]-v_gap, image.shape[0]-1, h_position[i]-v_gap)] = 1
io.imshow(binary_img)
io.show()

# Image analysis, loop over all images
for i in range(n_image):
	filename = film_ana.open_file_4_digits(i * incr_image) # Converts the image number into file name
	image = io.imread(filename) # Opens the image
	image = transform.rotate(image, rotate_angle, resize=True) # Rotates the image
	image = image[top:bottom, left:right] # Crops the image
	thresh = filter.threshold_otsu(image) # Applies an Otsu threshold
	binary_img = image > thresh # Converts into a black and white image
	n_tot = film_ana.image_analysis(binary_img) # Counts the number of films on the image
	survival_function.append(n_tot) # Fills the survival_function array with the current number of films
	time.append(i*time_conv) # Fills the time array with the current time
	if (100 * i / n_image) % 5 == 0:
		print(100 * i / n_image, '%')

# Survival function cleaning
survival_function = film_ana.survival_cleaning(survival_function)

# Data saving
np.savetxt('./' + exp_id  + '_data.txt', np.transpose([time, survival_function]))

# Survival function plotting
plt.plot(time, survival_function, 'or-')
plt.xlabel('Time [' + time_unit + ']')
plt.ylabel('Number of films')
plt.show()
