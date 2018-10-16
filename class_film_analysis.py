import numpy as np
from scipy import signal

class film_analysis:


	def __init__(self, fileroot, extension, h_position, err, n_tube, w_min, w_max, v_gap, time_conv, \
		low_boundary_fft_filter, high_boundary_fft_filter, threshold_value):
		"""
		Input: fileroot (string), extension (string), h_position (array), err (int),
		n_tube (int), w_min (int), w_max (int), v_gap (int).
		-----
		Output: None.
		-----
		Comment: Constructor. Takes as self parameters the fileroot of the data
		(without the name), the extension (e.g. '.png', '.BMP', etc.), the horizontal position,
		h_position, of the tubes. It also takes the relative error between the 3 measuring
		lines (+/- err), the number of tube, n_tube, the range of the peak to
		detect (between w_min and w_max) and the gap between two consecutive measuring lines.
		-----
		"""
		self.fileroot = fileroot
		self.extension = extension
		self.h_position =h_position
		self.err = err
		self.n_tube = n_tube
		self.w_min = w_min
		self.w_max = w_max
		self.v_gap = v_gap
		self.time_conv = time_conv
		self.low_boundary_fft_filter = low_boundary_fft_filter
		self.high_boundary_fft_filter = high_boundary_fft_filter
		self.threshold_value = threshold_value

	def open_file_4_digits(self, img_nb):
		"""
		Input: img_ng (int).
		-----
		Output: filename (string).
		-----
		Comment: Takes the index of the image to analyse and return the actual name:
		"self.fileroot + str(img_nb) + self.extension". The index is four digits format,
		(from 0000 to 9999).
		-----
		"""
		filename = self.fileroot + str('%04d' % img_nb) + self.extension
		return filename


	def bandpass_filter(self, image):
		"""
		Input: image (matrix), low_boundary_fft_filter (float between 0 and 1),
		high_boundary_fft_filter (float between 0 and 1).
		-----
		Output: filtered_image (matrix).
		-----
		Comment: Works out the fast Fourier transform of image, normalizes the result,
		filters the frequencies between low_boundary_fft_filter and high_boundary_fft_filter,
		turns the result back to the original scale, works out the inverse fast Fourier transform
		and returns the filtered image (filtered_image).
		-----
		"""
		im_fft = np.fft.rfft2(image)
		im_fft_normalized = (im_fft - im_fft.min()) / (im_fft.max() - im_fft.min())
		for i in range(len(im_fft_normalized[:,0])):
			for j in range(len(im_fft_normalized[0,:])):
				if (self.low_boundary_fft_filter > im_fft_normalized[i,j]) or (im_fft_normalized[i,j] > self.high_boundary_fft_filter):
					im_fft_normalized[i]=0
		im_fft_unnormalized = im_fft_normalized * (im_fft.max() - im_fft.min()) + im_fft.min()
		filtered_image = np.fft.irfft2(im_fft_unnormalized)
		return filtered_image


	def threshold(self, image):
		"""
		Input: image (matrix), threshold_value (float between 0 and 1).
		-----
		Output: binary (matrix).
		-----
		Comment: Normalizes the grey scale, works out the threshold according to the wanted
		value (threshold_value) and returns the binarized image (binary).
		-----
		"""
		image = ((image - image.min()) / (image.max() - image.min()))
		binary = np.copy(image)*0
		for i in range(len(image[:,0])):
			for j in range(len(image[0,:])):
				if image[i,j] < self.threshold_value:
					binary[i,j] = 0
				else:
					binary[i,j] = 1
		return binary


	def tube_analysis(self, peak_0, peak_left, peak_right):
		"""
		Input: peak_0 (array), peak_left (array), peak_right (array).
		-----
		Output: n_film (int), peak_position (array).
		-----
		Comment: Takes the positions of the films for the 3 measuring lines
		(peak_0, peak_left, peak_right), returns the number of films in the
		tube (n_film) and their positions (peak_position).
		A film is counted if the 3 measuring lines give the same values within
		the tolerence of self.err.
		-----
		"""
		peak_position = []
		n_film = 0
		for j in range(len(peak_0)):
			for k in range(len(peak_left)):
				for l in range(len(peak_right)):
					if (abs(peak_0[j] - peak_left[k]) <= self.err) and (abs(peak_left[k] - peak_right[l]) <= self.err):
						peak_position.append((peak_0[j] + peak_left[k] + peak_right[l])/3)
		n_film += len(peak_position)
		return n_film, peak_position


	def image_analysis(self, binary, image_nb):
		"""
		Input: binary (matrix), image_nb (int).
		-----
		Output: n_tot (int), all_tube_peak_position (array).
		-----
		Comment: Takes the full thresholded image, binary, loop over all tubes using
		the self.tube_analysis function and returns the number of films in the image,
		their positions and the time ([T, X, Y]).
		-----
		"""
		n_tot = 0
		all_tube_peak_position = []
		binary = binary + 1e-16
		for i in range(self.n_tube):
			peak_0 = signal.find_peaks_cwt(binary[:,int(self.h_position[i])], np.arange(self.w_min, self.w_max))
			peak_left = signal.find_peaks_cwt(binary[:,int(self.h_position[i])+self.v_gap], np.arange(self.w_min, self.w_max))
			peak_right = signal.find_peaks_cwt(binary[:,int(self.h_position[i])-self.v_gap], np.arange(self.w_min, self.w_max))
			n_film, peak_position = self.tube_analysis(peak_0, peak_left, peak_right)
			n_tot += n_film
			#all_tube_peak_position.append(peak_position)
			slice_time = np.zeros(len(peak_position)) + self.time_conv * image_nb
			x_position = np.zeros(len(peak_position)) + self.h_position[i]
			peak_position = np.column_stack([slice_time, x_position, peak_position])
			if i != 0:
				all_tube_peak_position = np.concatenate([all_tube_peak_position, peak_position], axis=0)
			else:
				all_tube_peak_position = peak_position
		return n_tot, all_tube_peak_position


	def survival_cleaning(self, survival_function):
		"""
		Input: survival_function (array).
		-----
		Output: np.array(survival_function) - min(survival_function) (numy array).
		-----
		Comment: Cleans the survival_function, removes the static detected elements
		and the films which appear during the recording.
		-----
		"""
		for i in range(len(survival_function)-1):
			if survival_function[i+1] > survival_function[i]:
				survival_function[i+1] = survival_function[i]
		return np.array(survival_function) - min(survival_function)
