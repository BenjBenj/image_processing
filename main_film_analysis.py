from skimage import io, transform, draw, filters, morphology
import numpy as np
import libconf
import matplotlib.pyplot as plt
from class_film_analysis import film_analysis

"""
The script works out the survival function from the raw images of the experiment.
The first image is displayed, after the threshold, the rotation, the cropping and
the filtering. In addition, the measuring lines are plotted on the first image
to check by eye the validity of the analysis.
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
length_conv = config['length_conv'] # Pixel conversion
time_unit = config['time_unit'] # Time unit
err = config['err'] # Horizontal error on the detection lines
n_tube = config['n_tube'] # Number of tubes in the experiment
w_min = config['w_min'] # Min width of the peaks
w_max = config['w_max'] # Max width of the peaks
v_gap = config['v_gap'] # Mumber of pixels between the 3 detection lines
h_position = config['h_position'] # Horizontal position of the tubes
rotate_angle = config['rotate_angle'] # Angle to get the image horizontal
top = config['top'] # Start pixel top (REFERENCE FOR THE Y POSITION)
bottom = config['bottom'] # End pixel bottom
left = config['left'] # Start pixel left
right = config['right'] # End pixel right
low_boundary_fft_filter = config['low_boundary_fft_filter'] # Bandpass filter: low boundary
high_boundary_fft_filter = config['high_boundary_fft_filter'] # Bandpass filter: high boundary
threshold_value = config['threshold_value'] # Threshold black/white limit

# Object calling
film_ana = film_analysis(fileroot, extension, h_position, err, n_tube, w_min, w_max, v_gap, time_conv, \
low_boundary_fft_filter, high_boundary_fft_filter, threshold_value)

# Arrays creation
survival_function = []
time = []
all_film_position = [] # Raw data containing the positions of the films for each image.

# Displays the first image after rotation, cropping, filtering and draws the position of the measuring lines
# Also displays the initial number of films
last_first_img = [int((n_image * incr_image) - 1 ), 0]
n_last_first = np.zeros(2)

for i in range(2):
	filename = film_ana.open_file_4_digits(last_first_img[i])
	image = io.imread(filename)
	image = transform.rotate(image, rotate_angle, resize=True)
	image = image[top:bottom, left:right]

	### image = film_ana.bandpass_filter(image) # Filter (frequency)
	### image = film_ana.threshold(image) # Binary threshold

	thresh = filters.threshold_yen(image)
	binary_img = image > thresh
	binary_img = morphology.binary_erosion(binary_img, morphology.square(1))

	n_last_first[i], peak_position_i = film_ana.image_analysis(binary_img, last_first_img[i])

print('Initial number of films: ', int(n_last_first[1] - n_last_first[0]))

for i in range(n_tube):
	binary_img[draw.line(0, h_position[i], image.shape[0]-1, h_position[i])] = 1
	binary_img[draw.line(0, h_position[i]+v_gap, image.shape[0]-1, h_position[i]+v_gap)] = 1
	binary_img[draw.line(0, h_position[i]-v_gap, image.shape[0]-1, h_position[i]-v_gap)] = 1
io.imshow(binary_img)
io.show()

# Image analysis, loop over all images
print('\nProgress:\n---------\n', 0, '%')
for i in range(n_image):
	filename = film_ana.open_file_4_digits(i * incr_image) # Converts the image number into file name
	image = io.imread(filename) # Opens the image
	image = transform.rotate(image, rotate_angle, resize=True) # Rotates the image
	image = image[top:bottom, left:right] # Crops the image

	### image = film_ana.bandpass_filter(image) # Filter (frequency)
	### image = film_ana.threshold(image) # Binary threshold

	thresh = filters.threshold_yen(image) # Threshold
	binary_img = image > thresh
	binary_img = morphology.binary_erosion(binary_img, morphology.square(1)) # Removes small particles

	n_tot, peak_position_i = film_ana.image_analysis(binary_img, i) # Counts the number of films on the image
	survival_function.append(n_tot) # Fills the survival_function array with the current number of films
	time.append(i * time_conv) # Fills the time array with the current time

	#all_film_position.append(peak_position_i)
	if i != 0:
		all_film_position = np.concatenate([all_film_position, peak_position_i], axis=0)
	else:
		all_film_position = peak_position_i

	if (100 * i / n_image) % 5 == 0:
		print(100 * (i+1) / n_image, '%')

	#print(n_tot)
	#io.imshow(binary_img)
	#io.show()

# Survival function cleaning
survival_function = film_ana.survival_cleaning(survival_function)

# Data saving
np.savetxt('./' + exp_id  + '_survival_data.txt', np.transpose([time, survival_function]))
np.savetxt('./' + exp_id  + '_position_data.txt', np.transpose([all_film_position[:,0], all_film_position[:,1], all_film_position[:,2] * length_conv]))

# Survival function plotting
plt.plot(time, survival_function, 'or-')
plt.xlabel('Time [' + time_unit + ']')
plt.ylabel('Number of films')
plt.show()
