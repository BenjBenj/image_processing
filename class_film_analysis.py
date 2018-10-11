import numpy as np
from scipy import signal

class film_analysis:


	def __init__(self, fileroot, extension, h_position, err, n_tube, w_min, w_max, v_gap):
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


	def tube_analysis(self, peak_0, peak_left, peak_right):
		"""
		Input: peak_0 (array), peak_left (array), peak_right (array).
		-----
		Output: n_film (int).
		-----
		Comment: Takes the positions of the films for the 3 measuring lines
		(peak_0, peak_left, peak_right) and returns the number of film in the tube.
		A film is counted if the 3 measuring lines give the same values within
		the tolerence of self.err.
		-----
		"""
		real_peak = []
		n_film = 0
		for j in range(len(peak_0)):
			for k in range(len(peak_left)):
				for l in range(len(peak_right)):
					if (abs(peak_0[j] - peak_left[k]) <= self.err) and (abs(peak_left[k] - peak_right[l]) <= self.err):
						real_peak.append((peak_0[j] + peak_left[k] + peak_right[l])/3)
		n_film += len(real_peak)
		return n_film


	def image_analysis(self, binary):
		"""
		Input: binary (matrix).
		-----
		Output: n_tot (int).
		-----
		Comment: Takes the full thresholded image, binary, loop over all tubes using
		the self.tube_analysis function and returns the number of films in the image.
		-----
		"""
		n_tot = 0
		for i in range(self.n_tube):
			peak_0 = signal.find_peaks_cwt(binary[:,int(self.h_position[i])], np.arange(self.w_min, self.w_max))
			peak_left = signal.find_peaks_cwt(binary[:,int(self.h_position[i])+self.v_gap], np.arange(self.w_min, self.w_max))
			peak_right = signal.find_peaks_cwt(binary[:,int(self.h_position[i])-self.v_gap], np.arange(self.w_min, self.w_max))
			n_film = self.tube_analysis(peak_0, peak_left, peak_right)
			n_tot += n_film
		return n_tot


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
