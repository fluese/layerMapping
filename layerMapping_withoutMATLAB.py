'''
Mapping of quantitative T1 values from MP2RAGE data onto surface
=======================================

This is a pipeline processing MP2RAGE data by performing the following steps:

01. Setup
02. Resampling to 500 µm (only for hires option)
03. Imhomogeneity correction and skull stripping
04. Registration of whole brain to slab data (optionally)
05. Regisration of additional to structural data (optionally)
06. Weighted image combination of whole brain and slab data (optionally)
07. Atlas-guided tissue classification using MGDM
08. Region extraction (left hemisphere) 
09. Crop volume (left hemisphere)
10. CRUISE cortical reconstruction (left hemisphere)
11. Extract layers across cortical sheet and map on surface (left hemisphere)
12. Region extraction (right hemisphere)
13. Crop volume (right hemisphere)
14. CRUISE cortical reconstruction (right hemisphere)
15. Extract layers across cortical sheet and map on surface (right hemisphere)

Version 0.97 (27.07.2022)
'''

############################################################################
# Contact information
# -------------------
# Dr. Falk Luesebrink
#
# falk dot luesebrink at ovgu dot de
# github.com/fluese
#

############################################################################
# Acknowledgement
# -------------------
# This work was supported by CRC 1436 "Neural Resources of Cognition" of
# the German Research Foundation (DFG) under project ID 425899996.

############################################################################
# THINGS TO DO
# -------------------
# 1. Specify files, paths, and flag from outside this script
# 2. Flag to write all data to disk or final results only
# 3. Add logging feature
#

############################################################################
# Installation instructions
# -------------------
# Things needed to be installed:
# 1. Nighres (https://nighres.readthedocs.io/en/latest/installation.html)
# 2. antspy (https://github.com/ANTsX/ANTsPy)
#

############################################################################
# Usage instructions
# -------------------
# This script requires whole brain MP2RAGE data organized according to BIDS.
# You can make use of a high resolution MP2RAGE slab additionally. The slab
# will be merged into the whole brain volume automatically and used for
# further processing. 
#
# In the subsection "set parameters" of the setup section below, you need to
# specify the folder to the BIDS directory and the label of the subject you
# want to process.
#
# Furthermore, you can specify a single volume which will be mapped on the
# surface additionally. This is meant for mapping another contrast  on the
# surface, e.g. QSM data or fMRI results. The resulting transformation can
# be applied either to another volume or to all compressed NIfTI files of
# an entire directory.
#
# To enable logging of the output, one can send the screen output into a
# file. This can be done, e.g. by using following command:
# python3 -u surfaceMapping.py |& tee -a /tmp/luesebrink/surfaceMapping.log
#
# A few flags have been set up which allow to reprocess data at various
# stages of the pipeline (e.g. entirely, from the segmentation onwards,
# mapping of additional data, or per hemisphere).
#

############################################################################
# 1. Setup
# -------------------
# First, we import nighres, the os and nibabel module. Set the in- and output
# directory and define the file names. In the final release this will be
# changed accordingly.

############################################################################
# 1.1. Import python libraries
# -------------------
import nighres
import os
import ants
import nibabel as nb
import glob
import numpy as np
from nilearn.image import mean_img
from nilearn.image import crop_img
from nibabel.processing import conform
from time import localtime, strftime
from typing import Tuple

############################################################################
# 1.2. Set parameters
# -------------------
# Define BIDS path
BIDS_path = '/tmp/luesebrink/sensemap/'

# Define subject following BIDS
sub = 'aae'

# Process with an additional high resolution MP2RAGE slab. If 'True' the 
# first run must be the lower resolution whole brain MP2RAGE volume and
# the second run must be the higher resolution MP2RAGE slab volume.
hires = True

# Map specific volume onto the surface. This could be the BOLD of a task
# fMRI time series (preferably the mean across the time series) or the
# magnitude data of a QSM volume. This volume will be registered to the
# T1map of the MP2RAGE volume.
# Requries an absolute path to a NIfTI file. Results will be written to
# <BIDS_path>/derivatives/sub-<label>/. If the path points to a 
# non-existing file, the according option will be omitted.
map_data = ''
#map_data = BIDS_path + 'sub-wtl/anat/sub-wtl_part-mag_SWI.nii.gz'
#map_data = BIDS_path + 'derivatives/sub-wtl/func/sub-wtl_task-rest_bold_mean.nii.gz'
#map_data = BIDS_path + 'derivatives/sub-wtl/func/sub-wtl_task-pRF_bold_mean.nii.gz'

# Transform data in the same space as 'map_data' which is then mapped onto
# the surface. Could for example be the statistical maps of SPM from fMRI or QSM
# data.
# Requries an absolute path to a NIfTI file or a directory containing compressed
# NIfTI files. The transformation is then applied to all files within the
# directory. Results will be written to <BIDS_path>/derivatives/sub-<label>/.
# If the path points to a non-existing file or directory, the according
# option will be omitted.
transform_data = ''
#transform_data = BIDS_path + 'derivatives/sub-wtl/QSM/sub-wtl_Chimap.nii.gz'
#transform_data = BIDS_path + 'derivatives/sub-wtl/resting_state/sub-wtl_task-rest_bold_ecm_rlc.nii.gz'
#transform_data = BIDS_path + 'derivatives/sub-wtl/pRF_statisticalMaps/'
#transform_data = BIDS_path + 'derivatives/sub-wtl/pRF_model/'

# Flag to overwrite all existing data
reprocess = True

# Flag to start reprocessing from segmentation onwards
reprocess_segmentation = False

# Flag to reprocess left or right hemisphere only
reprocess_leftHemisphere = False
reprocess_rightHemisphere = False

# Flag to reprocess additional mapping (and transformation) data. This flag
# is especially useful if you want to map more information onto the surface
# without re-running the entire pipeline. The basename of the output will be
# based on the file name of the input data. According to BIDS the subject
# label will be added as prefix to all output data.
reprocess_map_data = False

# Define path from where data is to be copied into "BIDS_path". Could either
# be to create a backup of the data before processing or transfer to a
# compute server.
# If the path does not exist, this option will be omitted and the data from
# "BIDS_path" will be used instead. Does not work for network drives.
copy_data_from = 'gerd:/media/luesebrink/bmmr_data/data/all/young/ben/'

# Define path to atlas. Here, we use custom priors which seem to work well
# with 7T MP2RAGE data collected in Magdeburg, Germany. Otherwise, please 
# choose the recent priors that come along with nighres (currently 3.0.9)
# instead of using the default atlas (currently 3.0.3). You can find that
# text file in the python package of nighres in the atlas folder.
atlas = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'atlas', 'brain-atlas-quant-3.0.9_customPriors_old.txt')

# Functions
def _min_max_scaling(image: np.ndarray, scaling_range: Tuple[int, int] = (10, 100)) -> Tuple[np.ndarray, float, float]:
	image_scaled = np.copy(image).astype(np.float32)
	min_ = (scaling_range[0] - np.min(image)).astype(np.float32)
	scale_ = ((scaling_range[1] - scaling_range[0]) / (np.max(image) - np.min(image))).astype(np.float32)
	image_scaled *= scale_
	image_scaled += min_
	print(np.min(image_scaled))
	return image_scaled, min_, scale_

def _invert_min_max_scaling(image_scaled: np.ndarray, scale_: float, min_: float) -> np.ndarray:
	image_scaled -= min_
	image_scaled /= scale_
	return image_scaled


############################################################################
# 1.2.1. Display stuff:
# -------------------
print('*****************************************************')
print('* Processing pipeline started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
print('Working directory: ' + BIDS_path)
print('Subject to be processed: ' + sub)

############################################################################
# 1.3. Define file names (and optionally create backup or copy data to
# compute server)
# -------------------
# Set paths
in_dir = BIDS_path + 'sub-' + sub + '/anat/'
out_dir = BIDS_path + 'derivatives/sub-' + sub + '/'
os.system('mkdir -p ' + in_dir)
os.system('mkdir -p ' + out_dir)

# Define file names
if hires == True:
	T1map = os.path.join(in_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')
	T1map_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_T1map.nii.gz')
	T1w = os.path.join(in_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')
	T1w_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_T1w.nii.gz')
	INV2 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-2_MP2RAGE.nii.gz')
	INV2_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_inv-2_MP2RAGE.nii.gz')
else:
	T1map = os.path.join(in_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')
	T1w = os.path.join(in_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')
	INV2 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-2_MP2RAGE.nii.gz')

# Copy data if path exists
if os.path.isdir(copy_data_from):
	print('')
	print('*****************************************************')
	print('* Data transfer to working directory.')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	if os.path.isfile(T1w)  and reprocess != True:
		print('Files exists already. Skipping data transfer.')
	else:
		os.system('scp -r ' + copy_data_from + ' ' + BIDS_path)
else:
	print('')
	print('*****************************************************')
	print('* Data transfer to working directory.')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	print('No path found to copy data from found. Continuing without copying data!')

# Check if data exists.
print('')
print('*****************************************************')
print('* Checking if files exists.')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
if os.path.isfile(T1map) and os.path.isfile(T1w):
	print('Files exists. Good!')
else:
	print('No files found. Exiting!')
	exit()

if hires == True:
	if os.path.isfile(T1map_slab) and os.path.isfile(T1w_slab):
		print('High resolution files exists. Good!')
	else:
		print('No high resolution files found. Continuing without hires option!')
		hires = False

# Check if file for mapping of additional data exists.
print('')
print('*****************************************************')
print('* Checking for additional data to be mapped.')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
if os.path.isfile(map_data):
	map_file_onto_surface = True
	print('File found for mapping of additional data onto surface. Good!')

	# Change name string for output of map data. This will first remove the path
	# of given the file for an appropriate naming of the output data.
	file_name = os.path.basename(map_data)
	index_of_dot = file_name.index('.')
	map_data_output = file_name[:index_of_dot]
	# This will remove the sub-<label> string if the data is already BIDS
	# formatted. It'll be added during creation of the output data later
	# again.
	if 'sub-' + sub in map_data_output:
		map_data_output = map_data_output[8:]
else:
	map_file_onto_surface = False
	map_data_output = ''
	print('Could not find file for surface mapping. Omitting flag!')

# Check if file for applying the transformation of the additional data exists.
if os.path.isfile(transform_data) == False and os.path.isdir(transform_data) == False:
	map_transform_file_onto_surface = False
	transform_data_output = ''
	print('Could not find file or path for applying transform of additional data. Omitting flag!')
elif os.path.isdir(transform_data) or os.path.isfile(transform_data):
	map_transform_file_onto_surface = True
	print('File or path found for applying transform of additional data. Good!')

	# Change name string for output of transformed data. This will remove the 
	# path of the input given for an appropriate naming of the output data.
	# In case a path is given as input a list with all compressed NIfTI files
	# will be created. Then the path from that list will be stripped to create
	# a list for appropriate naming of the output data.
	if os.path.isdir(transform_data):
		transform_data = glob.glob(transform_data + '*.nii.gz')
		transform_data_output = []
		if isinstance(transform_data, list):
			for tmp in transform_data:
				file_name = os.path.basename(tmp)
				index_of_dot = file_name.index('.')
				tmp = file_name[:index_of_dot]
				# This will remove the sub-<label> string if the data is already BIDS
				# formatted. It'll be added during creation of the output data later
				# again.
				if 'sub-' + sub in tmp:
					tmp = tmp[8:]
				transform_data_output.append(tmp)
	else:
		file_name = os.path.basename(transform_data)
		index_of_dot = file_name.index('.')
		transform_data_output = file_name[:index_of_dot]
		# This will remove the sub-<label> string if the data is already BIDS
		# formatted. It'll be added during creation of the output data later
		# again.	
		if 'sub-' + sub in transform_data_output:
			transform_data_output = transform_data_output[8:]

# For nameing and checking of processing files.
if hires == True:
	merged = '_merged_run-01+02'
	resampled = '_resampled'
else:
	merged = '_run-01'
	resampled = ''

############################################################################
# 3. Resampling to higher resolution data
# -------------------
# Resample image to an isotropic resolution of 500 µm.
#
# Should be changed to the resolution of the high resolution data at some
# point and not expect 500 µm data.
if hires == True:
	print('')
	print('*****************************************************')
	print('* Resampling MP2RAGE to an isotropic resolution of 500 µm')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	# Check for file and resample T1w and T1map using 4th order b spline interpolation.
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled.nii.gz')): #and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		input_img = nb.load(T1w)
		resampled_img = conform(input_img, out_shape=(336,448,448), voxel_size=(0.5,0.5,0.5), order=4)
		nb.save(resampled_img, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_resampled.nii.gz'))

		input_img = nb.load(T1map)
		resampled_img = conform(input_img, out_shape=(336,448,448), voxel_size=(0.5,0.5,0.5), order=4)
		nb.save(resampled_img, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled.nii.gz'))

		input_img = nb.load(INV2)
		resampled_img = conform(input_img, out_shape=(336,448,448), voxel_size=(0.5,0.5,0.5), order=4)
		nb.save(resampled_img, os.path.join(out_dir, 'sub-' + sub + '_run-01_INV2_resampled.nii.gz'))
		print('Done.')

	# Update file names
	T1map = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled.nii.gz')
	T1w = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_resampled.nii.gz')
	INV2 = os.path.join(out_dir, 'sub-' + sub + '_run-01_INV2_resampled.nii.gz')
else:
	# Copy and update file name
	os.system('cp ' + T1map + ' ' + out_dir + 'sub-' + sub + '_run-01_T1map.nii.gz')
	T1map = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')

############################################################################
# 4. Inhomogeneity correction and skull stripping
# ----------------
# Here, we perform inhomogeneity correction using ANTs N4 algorithm. After
# bias field correction a brainmask is applied.
print('')
print('*****************************************************')
print('* Inhomogeneity correction of whole brain data')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_' + resampled + 'biasCorrected_masked.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')

	# Update file names
	T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_masked.nii.gz')
	T1map_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected.nii.gz')
	T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected_masked.nii.gz')
	T1w_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected.nii.gz')
	T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected_masked.nii.gz')
else:
	# Create mask from INV2, T1w & T1map for better inhomogeneity correction with N4
#	mask = nighres.brain.mp2rage_skullstripping(
#		second_inversion=INV2,
#		t1_weighted=T1w,
#		#t1_map=T1map,
#		save_data=True,
#		overwrite=reprocess,
#		file_name='sub-' + sub + '_run-01_T1w' + resampled,
#		output_dir=out_dir)['brain_mask']

	mask = nighres.brain.intensity_based_skullstripping(
		main_image=INV2,
		extra_image=T1map,
		save_data=True,
		overwrite=False,
		file_name='sub-' + sub + '_run-01_T1map' + resampled,
		output_dir=out_dir)['brain_mask']

	# Estimate dura region
	dura = nighres.brain.mp2rage_dura_estimation(
		second_inversion=INV2,
		skullstrip_mask=mask,
		background_distance=5.0,
		output_type='dura_region',
		save_data=True,
		overwrite=False,
		output_dir=out_dir,
		file_name='sub-' + sub + '_run-01_T1map' + resampled)['result']
		
	# Binarize estimated dura region by threshold and invert mask
	dura_binary = ants.utils.threshold_image(
		ants.image_read(dura),
		low_thresh = 0.5,
		high_thresh = 1,
		inval = 0,
		outval = 1)
	
	# Remove estimated dura region from brain mask
	mask = ants.mask_image(ants.image_read(mask), dura_binary)
	
	# Truncate intensities
	T1w = ants.iMath(ants.image_read(T1w), 'TruncateIntensity', 0.01, 0.99)
	T1map = ants.iMath(ants.image_read(T1map), 'TruncateIntensity', 0.01, 0.99)
	
	# Rescale intesitiy for stability of N4
	#T1w_rescaler = ants.contrib.RescaleIntensity(T1w.min()+100,T1w.max()+100)
	#T1w = T1w_rescaler.transform(T1w)
	#T1map_rescaler = ants.contrib.RescaleIntensity(T1map.min()+100,T1map.max()+100)
	#T1map = T1map_rescaler.transform(T1map)

	# Run N4 for bias field correction
	T1w_biasCorrected = ants.n4_bias_field_correction(
			T1w,
			mask = mask,
			shrink_factor = 2,
			convergence = {'iters':[50,50,50,50],'tol':1e-05},
			spline_param = 160,
			verbose = True)

	T1map_biasCorrected = ants.n4_bias_field_correction(
			T1map,
			mask = mask,
			shrink_factor = 2,
			convergence = {'iters':[50,50,50,50],'tol':1e-05},
			spline_param = 160,
			verbose = True)
	
	# Rescale to original intensity range
	#T1w_rescaler = ants.contrib.RescaleIntensity(T1w.min()-100,T1w.max()-100)
	#T1w_biasCorrected = T1w_rescaler.transform(T1w_biasCorrected)
	#T1map_rescaler = ants.contrib.RescaleIntensity(T1map.min()-100,T1map.max()-100)
	#T1map_biasCorrected = T1map_rescaler.transform(T1map_biasCorrected)
	
	# Masking of images
	T1map_biasCorrected_masked = ants.mask_image(T1map_biasCorrected, mask)
	T1w_biasCorrected_masked = ants.mask_image(T1w_biasCorrected, mask)
	T1map_masked = ants.mask_image(T1map, mask)
	
	# # Write bias corrected images and masked images to disk
	ants.image_write(mask, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_brainmask.nii.gz'))
	ants.image_write(T1w_biasCorrected, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected.nii.gz'))
	ants.image_write(T1map_biasCorrected, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected.nii.gz'))
	ants.image_write(T1map_biasCorrected_masked, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected_masked.nii.gz'))
	ants.image_write(T1map_masked, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_masked.nii.gz'))
	ants.image_write(T1w_biasCorrected_masked, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected_masked.nii.gz'))
	
	# Update file names
	T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_masked.nii.gz')
	T1map_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected.nii.gz')
	T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected_masked.nii.gz')
	T1w_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected.nii.gz')
	T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected_masked.nii.gz')

if hires == True:
	print('')
	print('*****************************************************')
	print('* Inhomogeneity correction of slab data')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')

		# Update file names
		T1map_slab_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_masked.nii.gz')
		T1map_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz')
		T1map_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_masked.nii.gz')
		T1w_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected.nii.gz')
		T1w_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_masked.nii.gz')
	else:
		# Create mask from T1w for better inhomogeneity correction with N4
#		mask = nighres.brain.mp2rage_skullstripping(
#			second_inversion=INV2_slab,
#			t1_weighted=T1w_slab,
#			t1_map=T1map_slab,
#			save_data=True,
#			overwrite=reprocess,
#			file_name='sub-' + sub + '_run-02_T1w',
#			output_dir=out_dir)['brain_mask']

		mask = nighres.brain.intensity_based_skullstripping(
			main_image=INV2_slab,
			extra_image=T1map_slab,
			save_data=True,
			overwrite=reprocess,
			file_name='sub-' + sub + '_run-02_T1map',
			output_dir=out_dir)['brain_mask']

		# Estimate dura region
		dura = nighres.brain.mp2rage_dura_estimation(
			second_inversion=INV2_slab,
			skullstrip_mask=mask,
			background_distance=5.0,
			output_type='dura_region',
			save_data=True,
			overwrite=reprocess,
			file_name='sub-' + sub + '_run-02_T1map',
			output_dir=out_dir)['result']

		# Binarize estimated dura region by threshold and invert mask
		dura_binary = ants.utils.threshold_image(
			ants.image_read(dura),
			low_thresh = 0.8,
			high_thresh = 1,
			inval = 0,
			outval = 1)
		
		# Remove estimated dura region from brain mask
		mask = ants.mask_image(ants.image_read(mask), dura_binary)

		# Run N4 for bias field correction
		T1w_slab_biasCorrected = ants.n4_bias_field_correction(
				ants.image_read(T1w_slab),
				mask = mask,
				shrink_factor = 2,
				convergence = {'iters':[50,50,50,50],'tol':1e-05},
				spline_param = 160,
				verbose = True)

		T1map_slab_biasCorrected = ants.n4_bias_field_correction(
				ants.image_read(T1map_slab),
				mask = mask,
				shrink_factor = 2,
				convergence = {'iters':[50,50,50,50],'tol':1e-05},
				spline_param = 160,
				verbose = True)

		# Masking of images
		T1map_slab_biasCorrected_masked = ants.mask_image(T1map_slab_biasCorrected, mask)
		T1w_slab_biasCorrected_masked = ants.mask_image(T1w_slab_biasCorrected, mask)
		T1map_slab_masked = ants.mask_image(ants.image_read(T1map_slab), mask)
		
		# # Write bias corrected images and masked images to disk
		ants.image_write(mask, os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_brainmask.nii.gz'))
		ants.image_write(T1w_slab_biasCorrected, os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected.nii.gz'))
		ants.image_write(T1map_slab_biasCorrected, os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz'))
		ants.image_write(T1map_slab_biasCorrected_masked, os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_masked.nii.gz'))
		ants.image_write(T1map_slab_masked, os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_masked.nii.gz'))
		ants.image_write(T1w_slab_biasCorrected_masked, os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_masked.nii.gz'))

		# Update file names
		T1map_slab_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_masked.nii.gz')
		T1map_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz')
		T1map_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_masked.nii.gz')
		T1w_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected.nii.gz')
		T1w_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_masked.nii.gz')
		
############################################################################
# 5. Register data to upsampled 500 µm data
# -------------------
# Here, we register the T1map whole brain data to the high resolution
# slab using ANTs with an adapted script. This gives better registration
# results than registering the slab to the whole brain image. Furthermore,
# we use masked data to limit the registration for even better registration
# results. Afterwards we apply the transformation of this registration to
# the other contrasts by using antsApplyTransformation.
#
# Changes of the script include initial moving transform (from origin to
# contrast), number of iterations, precision of float instead double as
# well as BSpline interpolation instead of linear interpolation for
# sharper respresentation of the resulting volume.
if hires == True:
	print('')
	print('*****************************************************')
	print('* Register native 500 µm to upsampled 700 µm MP2RAGE data')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir + 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_' + sub + '_run-01_T1w_resampled_biasCorrected_masked.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		# Register whole brain data to high resolution slab
		registeredImage = ants.registration(
				fixed = ants.image_read(T1map_slab_biasCorrected_masked),
				moving = ants.image_read(T1map_biasCorrected_masked),
				type_of_transform = 'SyNRA',
				syn_metric = 'CC',
				syn_sampling = 4,
				reg_iterations = (200, 100, 30 ,15),
				verbose = True,
				#outprefix = out_dir + 'sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked_registered_to_sub-' + sub + '_run-02_T1map_biasCorrected_masked_',
				)
		print('')
		print('*****************************************************')
		print('* Apply transformations')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')
		# Apply transformation to several files.
		T1map_slab_reg = ants.apply_transforms(
				fixed = ants.image_read(T1map_biasCorrected),
				moving = ants.image_read(T1map_slab),
		                transformlist = registeredImage['invtransforms'],
				interpolator = 'bSpline',
				verbose = True,
				)

		ants.image_write(T1map_slab_reg, out_dir + 'sub-' + sub + '_run-02_T1map_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')

		T1map_slab_biasCorrected_reg = ants.apply_transforms(
				fixed = ants.image_read(T1map_biasCorrected),
				moving = ants.image_read(T1map_slab_biasCorrected),
		                transformlist = registeredImage['invtransforms'],
				interpolator = 'bSpline',
				verbose = True,
				)

		ants.image_write(T1map_slab_biasCorrected_reg, out_dir + 'sub-' + sub + '_run-02_T1map_biasCorrected_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz') 

		T1w_slab_biasCorrected_reg = ants.apply_transforms(
				fixed = ants.image_read(T1w_biasCorrected),
				moving = ants.image_read(T1w_slab_biasCorrected),
		                transformlist = registeredImage['invtransforms'],
				interpolator = 'bSpline',
				verbose = True,
				)

		ants.image_write(T1w_slab_biasCorrected_reg, out_dir + 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_' + sub + '_run-01_T1w_resampled_biasCorrected_masked.nii.gz') 

############################################################################
# 6.1 Register additional data to (upsampled) T1map
# -------------------
if reprocess_map_data:
	reprocess = True

if map_file_onto_surface == True:
	if hires == True:
		print('')
		print('*****************************************************')
		print('* Register additional data to upsampled T1map')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')
		reg1 = 'sub-' + sub + '_' + map_data_output + '_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected'
	else:
		print('')
		print('*****************************************************')
		print('* Register additional data to T1map')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')
		reg1 = 'sub-' + sub + '_' + map_data_output + '_registered_to_sub-' + sub + '_run-01_T1map_biasCorrected'

	if map_file_onto_surface:
		if os.path.isfile(os.path.join(out_dir, reg1 + '.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
		else:
			# Here, we mask the data to be registered to improve the registration
			map_data_mask = ants.image_read(map_data)
			map_data_mask = ants.get_mask(map_data_mask)
			map_data_masked = ants.mask_image(ants.image_read(map_data),map_data_mask)
			ants.image_write(map_data_masked, out_dir + 'sub-' + sub + '_' + map_data_output + '_masked.nii.gz')

			# Register additional data non-linearly to T1map using mutual information as similarity metric
			registeredImage = ants.registration(
					fixed = map_data_masked,
					moving = ants.image_read(T1map_biasCorrected_masked),
					type_of_transform = 'SyNRA',
					aff_random_sampling_rate = 1,
					grad_step = 0.1,
					reg_iterations = (200, 100, 50 , 30),
					verbose = True,
					# outprefix = out_dir + reg1 + '_',
					)

			# Apply transformation
			warpedImage = ants.apply_transforms(
					fixed = ants.image_read(T1map_biasCorrected),
					moving = ants.image_read(map_data),
				        transformlist = registeredImage['invtransforms'],
					interpolator = 'bSpline',
					verbose = True,
					)

			# Write file to disk
			ants.image_write(warpedImage, out_dir + reg1 + '.nii.gz') 

	# Update file name
	map_data = os.path.join(out_dir, reg1 + '.nii.gz')

############################################################################
# 6.2. Apply transformation to data to be mapped on the surface
# -------------------
# Define file name for output of transformation.
if isinstance(transform_data_output, list):
	reg2 = []
	for name in transform_data_output:
		if hires == True:
			reg2.append('sub-' + sub + '_' + name + '_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected.nii.gz')
		else:
			reg2.append('sub-' + sub + '_' + name + '_registered_to_' + sub + '_run-01_T1map_biasCorrected.nii.gz')

else:
	if hires == True:
		reg2 = 'sub-' + sub + '_' + transform_data_output + '_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected.nii.gz'
	else:
		reg2 = 'sub-' + sub + '_' + transform_data_output + '_registered_to_' + sub + '_run-01_T1map_biasCorrected.nii.gz'

if map_file_onto_surface and map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Apply transformation of registration to additional data')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	# Check if a list exists and if so, if the last file of list exists already.
	if isinstance(transform_data, list):
		if os.path.isfile(os.path.join(out_dir, reg2[-1])) and reprocess != True:
			print('File exists already. Skipping process.')
		else:
			# Iterate over length of list
			length = len(transform_data)
			for i in range(length):
				# Apply transformation
				warpedImage = ants.apply_transforms(
						fixed = ants.image_read(T1map_biasCorrected),
						moving = ants.image_read(transform_data[i]),
						transformlist = registeredImage['invtransforms'],
						interpolator = 'bSpline',
						verbose = True,
						)

				# Write file to disk
				ants.image_write(warpedImage, out_dir + reg2[i])

	else:
	# Check if output exists already.
		if os.path.isfile(os.path.join(out_dir, reg2)) and reprocess != True:
			print('File exists already. Skipping process.')
		else:
			# Apply transformation
			warpedImage = ants.apply_transforms(
					fixed = ants.image_read(T1map_biasCorrected),
					moving = ants.image_read(transform_data),
				        transformlist = registeredImage['invtransforms'],
					interpolator = 'bSpline',
					verbose = True,
					)

			# Write file to disk
			ants.image_write(warpedImage, out_dir + reg2) 
		
# Update file name
if isinstance(reg2, list):
	transform_data = []
	for tmp in reg2:
		transform_data.append(os.path.join(out_dir, tmp))		
else:
	transform_data = os.path.join(out_dir, reg2)

if reprocess_map_data:
	reprocess = False

############################################################################
# 7. Combination of native and upsampled data
# ----------------
# Here, we combine the slab and whole brain data by a weighted averaged
# using a custom MATLAB script. As the slab does not cover the entire
# temporal lobe, SNR is very low in the higher resolution MP2RAGE.
# Therefore, weighted averaging is applied in z-direction (superior to
# inferior) in the last third number of slices gradually increasing 
# the weighting with distance.
if hires == True:
	print('')
	print('*****************************************************')
	print('* Combination of native and upsampled data')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')

		# Update file names
		T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected_masked.nii.gz')
		T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_masked.nii.gz')
		T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected_masked.nii.gz')
	else:
		# Load data
		T1map_biasCorrected = ants.image_read(T1map_biasCorrected)
		T1map_slab_biasCorrected = ants.image_read(T1map_slab_biasCorrected)

		T1map = ants.image_read(T1map)
		T1map_slab = ants.image_read(T1map_slab)

		T1w_biasCorrected = ants.image_read(T1w_biasCorrected)
		T1w_slab_biasCorrected = ants.image_read(T1w_slab_biasCorrected)

		mask = ants.image_read(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_brainmask.nii.gz'))

		# Create volume filled with zeros
		clone = ants.image_clone(T1w_biasCorrected)
		weighted = ants.image_clone(T1w_biasCorrected)
		weighted[:,:,:] = 0

		# Get white matter segmentation from whole brain image to scale intensities
		segmentation=ants.atropos(a=T1w_biasCorrected, x=mask, m='[0.2,1x1x1]', c='[3,0]', i='kmeans[3]', v=1)['segmentation']
		segmentation=ants.threshold_image(segmentation, low_thresh=3, high_thresh=3, inval=1, outval=0, binary=False)
		segmentation=ants.iMath(segmentation, 'GetLargestComponent')
		segmentation=ants.morphology(segmentation, operation='dilate', radius=1, mtype='grayscale', shape='box')

		# Mask high resolution slab to get white matter segmentation
		segmentation_slab = ants.mask_image(slab,segmentation,binar=False)

		# Calulate ratio between whole brain white matter segmentation and high resolution slab data
		ratio = segmentation.mean() / segmentation_slab.mean()

		# Weight whole brain image by ratio to harmonize intensities between whole brain and slab data
		full = full / ratio

		# Get dimensions of image and reduce it by one to accomodate that array start at 0 and not 1...
		size = full.shape

		# Convert to numpy
		T1map_biasCorrected_image = T1map_biasCorrected.numpy()
		T1map_slab_biasCorrected_image = T1map_slab_biasCorrected.numpy()

		T1map_image = T1map.numpy()
		T1map_slab_image = T1map_slab.numpy()

		T1w_biasCorrected_image = T1w_biasCorrected.numpy()
		T1w_slab_biasCorrected_image = T1w_slab_biasCorrected.numpy()

		T1map_biasCorrected_weighted_image = weighted.numpy()
		T1map_weighted_image = weighted.numpy()
		T1w_biasCorrected_weighted_image = weighted.numpy()

		for x in range(size[0]-1):
			for y in range(size[1]-1):
				for z in range(size[2]-1):
					if T1map_biasCorrected_image[x,y,z] != 0 and T1map_slab_biasCorrected_image[x,y,z] == 0 and weighted_image[x,y,z] == 0:
						T1map_biasCorrected_weighted_image[x,y,z] = T1map_biasCorrected_image[x,y,z]
						T1map_weighted_image[x,y,z] = T1map_image[x,y,z]
						T1w_biasCorrected_weighted_image[x,y,z] = T1w_biasCorrected_image[x,y,z]
					elif T1map_biasCorrected_image[x,y,z] != 0 and T1map_slab_biasCorrected_image[x,y,z] != 0 and weighted_image[x,y,z] == 0:
						T1map_biasCorrected_weighted_image[x,y,z] = T1map_slab_biasCorrected_image[x,y,z]
						T1map_weighted_image[x,y,z] = T1map_slab_image[x,y,z]
						T1w_biasCorrected_weighted_image[x,y,z] = T1_slab_biasCorrected_image[x,y,z]
						if z >= round(size_z*2/3):
							T1map_biasCorrected_weighted_image[x,y,z] = T1map_slab_biasCorrected_image[x,y,z]
							T1map_weighted_image[x,y,z] = T1map_slab_image[x,y,z]
							T1w_biasCorrected_weighted_image[x,y,z] = T1w_slab_biasCorrected_image[x,y,z]
						else:
							counter = 0
							while T1map_biasCorrected_image[x,y,z-counter] != 0 and T1map_slab_biasCorrected_image[x,y,z-counter] != 0 and weighted_image[x,y,z] == 0:
								counter = counter + 1
								if z-counter < 1:
									counter = counter - 1
									break
							for weight_in_z in range(counter):
								T1map_biasCorrected_weighted_image[x,y,z-weight_in_z]=T1map_biasCorrected_image[x,y,z-weight_in_z]*weight_in_z/counter+T1map_slab_biasCorrected_image[x,y,z-weight_in_z]*(1-weight_in_z/counter)
								T1map_weighted_image[x,y,z-weight_in_z]=T1map_image[x,y,z-weight_in_z]*weight_in_z/counter+T1map_slab_image[x,y,z-weight_in_z]*(1-weight_in_z/counter)
								T1w_biasCorrected_weighted_image[x,y,z-weight_in_z]=T1w_biasCorrected_image[x,y,z-weight_in_z]*weight_in_z/counter+T1w_slab_biasCorrected_image[x,y,z-weight_in_z]*(1-weight_in_z/counter)
					if T1map_biasCorrected_weighted_image[x,y,z] < 0:
						T1map_biasCorrected_weighted_image = 0
						T1map_weighted_image = 0
						T1w_biasCorrected_weighted_image = 0

		# Copy numpy array into image
		T1map_biasCorrected = weighted.new_image_like(T1map_biasCorrected_weighted_image)
		T1map = weighted.new_image_like(T1map_weighted_image)
		T1w_biasCorrected = weighted.new_image_like(T1w_biasCorrected_weighted_image)

		# Mask image
		T1map_biasCorrected_masked = ants.mask_image(T1map_biasCorrected,mask)
		T1map_masked = ants.mask_image(T1map,mask)
		T1w_biasCorrected_masked = ants.mask_image(T1w_biasCorrected,mask)

		# Write to disk
		ants.image_write(T1map_biasCorrected, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected.nii.gz'))
		ants.image_write(T1map_biasCorrected_masked, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected_masked.nii.gz'))

		ants.image_write(T1map, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map.nii.gz'))
		ants.image_write(T1map_masked, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_masked.nii.gz'))

		ants.image_write(T1w_biasCorrected, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected.nii.gz'))
		ants.image_write(T1w_biasCorrected_masked, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected_masked.nii.gz'))

		# Update file names
		T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected_masked.nii.gz')
		T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_masked.nii.gz')
		T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected_masked.nii.gz')

#############################################################################
# 8. MGDM classification
# ---------------------
# Next, we use the masked data as input for tissue classification with the
# MGDM algorithm. MGDM works with a single contrast, but can  be improved
# with additional contrasts potentially. Here, we make use of the T1map 
# only.
#
# [Note: Using the T1w contrast additionally yields very strange 
# segmentation results for some subjects and needs further investigation.]
print('')
print('*****************************************************')
print('* Segmentation with MGDM')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if reprocess_segmentation:
	reprocess = True
	
mgdm_results = nighres.brain.mgdm_segmentation(
                contrast_image1=T1map_biasCorrected_masked,
                contrast_type1='T1map7T',
				#contrast_image2=T1w_biasCorrected_masked,
				#contrast_type2='Mp2rage7T',
                n_steps=10,
                max_iterations=1000,
                atlas_file=atlas,
                normalize_qmaps=False,
				#adjust_intensity_priors=True,
                save_data=True,
                overwrite=reprocess,
                output_dir=out_dir,
                file_name='sub-' + sub + merged + '_T1map')

###########################################################################
# 9. Region Extraction (left hemisphere)
# ------------------
# Here, we pull from the MGDM output the needed regions for cortical
# reconstruction: the GM cortex ('region'), the underlying WM (with filled
# subcortex and ventricles, 'inside') and the surrounding CSF (with masked
# regions, 'background')
print('')
print('*****************************************************')
print('* Region extraction of left hemisphere')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if reprocess_leftHemisphere:
	reprocess = True

cortex = nighres.brain.extract_brain_region(segmentation=mgdm_results['segmentation'],
    	levelset_boundary=mgdm_results['distance'],
        maximum_membership=mgdm_results['memberships'],
        maximum_label=mgdm_results['labels'],
        atlas_file=atlas,
        extracted_region='left_cerebrum',
        save_data=True,
        overwrite=reprocess,
        file_name='sub-' + sub + merged + '_leftHemisphere',
        output_dir=out_dir)

#############################################################################
# 10. Crop volume to left hemisphere
# --------------------------------
# Here, we crop the volume to the left hemisphere based on the output of
# MGDM. This will reduce the memory demand tremendously as well as
# processing time without sacrificing any accuracy.
print('')
print('*****************************************************')
print('* Crop volume to left hemisphere')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_T1map_cropped.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	inside_mask = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xmask-lcrwm_cropped.nii.gz'))
	inside_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xproba-lcrwm_cropped.nii.gz'))
	region_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xproba-lcrgm_cropped.nii.gz'))
	background_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xproba-lcrbg_cropped.nii.gz'))
	T1map_masked_leftHemisphere = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_T1map_cropped.nii.gz'))
else:
	# Load grey matter image, binarize image, and get information for cropping
	img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xmask-lcrgm.nii.gz'))
	tmp = img.get_fdata()
	tmp[tmp<0] = 0
	tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
	crop,coord = crop_img(tmp, pad=4, return_offset=True)

	# Apply cropping
	tmp1 = nb.load(cortex['inside_mask'])
	tmp2 = tmp1.get_fdata()
	inside_mask = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_mask, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xmask-lcrwm_cropped.nii.gz'))

	# Apply cropping
	tmp1 = nb.load(cortex['inside_proba'])
	tmp2 = tmp1.get_fdata()
	inside_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xproba-lcrwm_cropped.nii.gz'))

	# Apply cropping
	tmp1 = nb.load(cortex['region_proba'])
	tmp2 = tmp1.get_fdata()
	region_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(region_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xproba-lcrgm_cropped.nii.gz'))

	# Apply cropping
	tmp1 = nb.load(cortex['background_proba'])
	tmp2 = tmp1.get_fdata()
	background_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(background_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xproba-lcrbg_cropped.nii.gz'))

	tmp1 = nb.load(T1map_masked)
	tmp2 = tmp1.get_fdata()
	T1map_masked_leftHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(T1map_masked_leftHemisphere, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_T1map_cropped.nii.gz'))

	del img
	del tmp
	del tmp1
	del tmp2
	print('Done.')

############################################################################
# 10.1. Crop additional data to left hemispehre
# -------------------
if reprocess_map_data:
	reprocess = True

if map_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Crop additional data to left hemisphere')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_leftHemisphere_cropped.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		map_data_leftHemisphere = os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_leftHemisphere_cropped.nii.gz')
	else:
		# Load grey matter image, binarize image, and get information for cropping
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xmask-lcrgm.nii.gz'))
		tmp = img.get_fdata()
		tmp[tmp<0] = 0
		tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
		crop,coord = crop_img(tmp, pad=4, return_offset=True)

		# Apply cropping
		tmp1 = nb.load(map_data)
		tmp2 = tmp1.get_fdata()
		map_data_leftHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
		nb.save(map_data_leftHemisphere, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_leftHemisphere_cropped.nii.gz'))

		del img
		del tmp
		del tmp1
		del tmp2
		print('Done.')

############################################################################
# 10.2. Crop additionally transformed data to left hemispehre
# -------------------
if map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Crop transformed data to left hemisphere')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	# Check if a list of multiple files exists to be transformed
	if isinstance(transform_data_output,list):
		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[-1] + '_leftHemisphere_cropped.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			transform_data_leftHemisphere = []
			for tmp in transform_data_output:
				transform_data_leftHemisphere.append(os.path.join(out_dir, 'sub-' + sub + '_' + tmp + '_leftHemisphere_cropped.nii.gz'))
		else:
			# Load grey matter image, binarize image, and get information for cropping
			img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xmask-lcrgm.nii.gz'))
			tmp = img.get_fdata()
			tmp[tmp<0] = 0
			tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
			crop,coord = crop_img(tmp, pad=4, return_offset=True)

			# Apply cropping
			length = len(transform_data)
			transform_data_leftHemisphere = []
			for i in range(length):
				tmp1 = nb.load(transform_data[i])
				tmp2 = tmp1.get_fdata()
				tmp3 = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
				nb.save(tmp3, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_leftHemisphere_cropped.nii.gz'))
				transform_data_leftHemisphere.append(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_leftHemisphere_cropped.nii.gz'))

			del img
			del tmp
			del tmp1
			del tmp2
			del tmp3
			print('Done.')

	else:
		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_leftHemisphere_cropped.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			transform_data_leftHemisphere = os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_leftHemisphere_cropped.nii.gz')
		else:
			# Load grey matter image, binarize image, and get information for cropping
			img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_xmask-lcrgm.nii.gz'))
			tmp = img.get_fdata()
			tmp[tmp<0] = 0
			tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
			crop,coord = crop_img(tmp, pad=4, return_offset=True)

			# Apply cropping
			tmp1 = nb.load(transform_data)
			tmp2 = tmp1.get_fdata()
			transform_data_leftHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
			nb.save(transform_data_leftHemisphere, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_leftHemisphere_cropped.nii.gz'))

			del img
			del tmp
			del tmp1
			del tmp2
			print('Done.')

if reprocess_map_data:
	reprocess = False

#############################################################################
# 11. CRUISE cortical reconstruction (left hemisphere)
# --------------------------------
# the WM inside mask as a (topologically spherical) starting point to grow a
# refined GM/WM boundary and CSF/GM boundary
print('')
print('*****************************************************')
print('* Cortical reconstruction with CRUISE')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

cruise = nighres.cortex.cruise_cortex_extraction(
    init_image=inside_mask,
    wm_image=inside_proba,
	gm_image=region_proba,
	csf_image=background_proba,
	vd_image=None,
    data_weight=0.8,
    regularization_weight=0.2,
	max_iterations=1000,
    normalize_probabilities=True,
    save_data=True,
    overwrite=reprocess,
    file_name='sub-' + sub + merged + '_leftHemisphere',
    output_dir=out_dir)

###########################################################################
# 12. Extract layers across cortical sheet and map on surface
###########################################################################
# 12.1 Volumetric layering for depth measures (left hemisphere)
# ---------------------
# Finally, we use the GM/WM boundary (GWB) and CSF/GM boundary (CGB) from
# CRUISE to compute cortical depth with a volume-preserving technique.
print('')
print('*****************************************************')
print('* Extract layers across the cortical sheet (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

layers = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise['gwb'],
                        outer_levelset=cruise['cgb'],
                        n_layers=20,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers-allLayers',
                        output_dir=out_dir)['boundaries']
layers = nb.load(layers)

###########################################################################
# 12.2. Extract middle layer for surface generation and mapping (left hemisphere)
# ---------------------
# Here, we extract the levelset of the middle layer which will be used to
# map the information on.
print('')
print('*****************************************************')
print('* Extracting middle cortical layer (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-middleLayer.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	extractedLayers=nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-middleLayer.nii.gz'))
else:
	tmp = layers.get_fdata()[:,:,:,10:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-middleLayer.nii.gz'))
	del tmp

###########################################################################
# 12.3 Cortical surface generation and inflation of middle layer (left hemisphere)
# ---------------------
# Here, we create a surface from the levelset of the middle layer and
# inflate it afterwards for better visualisation.
print('')
print('*****************************************************')
print('* Generate surface of middle cortical layer and inflate it (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

corticalSurface = nighres.surface.levelset_to_mesh(
                        levelset_image=extractedLayers,
			connectivity='26/6',
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

inflatedSurface = nighres.surface.surface_inflation(
                        surface_mesh=corticalSurface['result'],
			max_iter=10000,
			max_curv=8.0,
			#regularization=1.0,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

###########################################################################
# 12.4. Profile sampling of all layers (left hemisphere)
# ---------------------
# Here, we sample the T1 values of the T1map onto all layers.
print('')
print('*****************************************************')
print('* Profile sampling across cortical sheet (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

profile = nighres.laminar.profile_sampling(
                        profile_surface_image=layers,
                        intensity_image=T1map_masked_leftHemisphere,
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers')['result']
profile = nb.load(profile)

###########################################################################
# 12.5. Extract all cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 3 to 18, removing layers close to the white
# matter and cerebrospinal fluid to avoid partial volume effects.
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting all cortical layers (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
else:
	tmp = profile.get_fdata()[:,:,:,3:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-allLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers-allLayers.vtk')


###########################################################################
# 12.6. Extract deep cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 3 to 6, to cover the "deep" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting deep cortical layers (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-deepLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:7]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-deepLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-deepLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers-deepLayers.vtk')

###########################################################################
# 12.7. Extract inner middle cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 7 to 10, to cover the "inner middle" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting inner middle cortical layers (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-innerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,7:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-innerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-innerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers-innerLayers.vtk')

###########################################################################
# 12.8. Extract outer middle cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 11 to 14, to cover the "outer middle" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting outer middle cortical layers (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-outerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,11:15]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-outerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-outerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers-outerLayers.vtk')

###########################################################################
# 12.9. Extract superficial cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 15 to 18, to cover the "superficial" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting superficial cortical layers (left hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-superficialLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,15:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-superficialLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-superficialLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_leftHemisphere_extractedLayers-superficialLayers.vtk')

###########################################################################
# 12.10. Profile sampling of additional data of all layers and mapping it onto
# the surface (left hemisphere)
# ---------------------
# Here, we sample the information of the additional data onto all layers.
# And we extract the layers 1 to 18, removing layers close to CSF to avoid
# partial volume effects. Afterwards the information is mapped on the
# original and inflated surface of the middle layer.
if reprocess_map_data:
	reprocess = True

if map_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Profile sampling of additional data (left hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	profile = nighres.laminar.profile_sampling(
		                profile_surface_image=layers,
		                intensity_image=map_data_leftHemisphere,
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_' + map_data_output + '_leftHemisphere_extractedLayers')['result']
	profile = nb.load(profile)

	print('')
	print('*****************************************************')
	print('* Extracting all cortical layers of additional data (left hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	else:
		tmp = profile.get_fdata()[:,:,:,1:19]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_leftHemisphere_extractedLayers-allLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		                intensity_image=meanProfile,
		                surface_mesh=corticalSurface['result'],
		                inflated_mesh=inflatedSurface['result'],
		                mapping_method='closest_point',
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_' + map_data_output + '_leftHemisphere_extractedLayers-allLayers.vtk')

###########################################################################
# 12.11. Profile sampling of transformed data of all layers and mapping it onto
# the surface (left hemisphere)
# ---------------------
# Here, we sample the information of the transformed data onto all layers.
# And we extract the layers 1 to 18, removing layers close to CSF to avoid
# partial volume effects. Afterwards the information is mapped on the original
# and inflated surface of the middle layer.
if map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Profile sampling of transformed data (left hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')	

	if isinstance(transform_data_output,list):
		length = len(transform_data_output)
		for i in range(length):
			print('')
			print('*****************************************************')
			print('* Profile sampling and extracting all cortical layers of additional data (left hemisphere)')
			print('*****************************************************')

			profile = nighres.laminar.profile_sampling(
						profile_surface_image=layers,
						intensity_image=transform_data_leftHemisphere[i],
						save_data=True,
						overwrite=reprocess,
						output_dir=out_dir,
						file_name='sub-' + sub + '_' + transform_data_output[i] + '_leftHemisphere_extractedLayers')['result']
			profile = nb.load(profile)

			if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
				print('File exists already. Skipping process.')
				meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
			else:
				tmp = profile.get_fdata()[:,:,:,1:19]
				extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
				nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_leftHemisphere_extractedLayers-allLayers.nii.gz'))
				meanProfile = mean_img(extractedLayers)
				nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
				del tmp

			nighres.surface.surface_mesh_mapping(
						intensity_image=meanProfile,
						surface_mesh=corticalSurface['result'],
						inflated_mesh=inflatedSurface['result'],
						mapping_method='closest_point',
						save_data=True,
						overwrite=reprocess,
						output_dir=out_dir,
						file_name='sub-' + sub + '_' + transform_data_output[i] + '_leftHemisphere_extractedLayers-allLayers.vtk')
	else:
		profile = nighres.laminar.profile_sampling(
				        profile_surface_image=layers,
				        intensity_image=transform_data_leftHemisphere,
				        save_data=True,
				        overwrite=reprocess,
				        output_dir=out_dir,
				        file_name='sub-' + sub + '_' + transform_data_output + '_leftHemisphere_extractedLayers')['result']
		profile = nb.load(profile)

		print('')
		print('*****************************************************')
		print('* Extracting all cortical layers of additional data (left hemisphere)')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')

		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
		else:
			tmp = profile.get_fdata()[:,:,:,1:19]
			extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
			nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_leftHemisphere_extractedLayers-allLayers.nii.gz'))
			meanProfile = mean_img(extractedLayers)
			nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
			del tmp

		nighres.surface.surface_mesh_mapping(
				        intensity_image=meanProfile,
				        surface_mesh=corticalSurface['result'],
				        inflated_mesh=inflatedSurface['result'],
				        mapping_method='closest_point',
				        save_data=True,
				        overwrite=reprocess,
				        output_dir=out_dir,
				        file_name='sub-' + sub + '_' + transform_data_output + '_leftHemisphere_extractedLayers-allLayers.vtk')

# Set reprocess flag to false in case left hemisphere or mapping of
# additional data should be done only.
if reprocess_leftHemisphere or reprocess_map_data:
	reprocess = False

###########################################################################
# 13. Region Extraction (right hemisphere)
# ------------------
# Here, we pull from the MGDM output the needed regions for cortical
# reconstruction: the GM cortex ('region'), the underlying WM (with filled
# subcortex and ventricles, 'inside') and the surrounding CSF (with masked
# regions, 'background')
print('')
print('*****************************************************')
print('* Region extraction of right hemisphere')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if reprocess_rightHemisphere or reprocess_segmentation:
	reprocess = True

cortex = nighres.brain.extract_brain_region(segmentation=mgdm_results['segmentation'],
                                            levelset_boundary=mgdm_results['distance'],
                                            maximum_membership=mgdm_results['memberships'],
                                            maximum_label=mgdm_results['labels'],
			                    atlas_file=atlas,
                                            extracted_region='right_cerebrum',
                                            save_data=True,
                                            overwrite=reprocess,
                                            file_name='sub-' + sub + merged + '_rightHemisphere',
                                            output_dir=out_dir)

#############################################################################
# 14. Crop volume to right hemisphere
# --------------------------------
# Here, we crop the volume to the right hemisphere based on the output of
# MGDM. This will reduce the memory demand tremendously as well as
# processing time without sacrificing any accuracy.
print('')
print('*****************************************************')
print('* Crop volume to right hemisphere')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_T1map_cropped.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	inside_mask = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xmask-rcrwm_cropped.nii.gz'))
	inside_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xproba-rcrwm_cropped.nii.gz'))
	region_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xproba-rcrgm_cropped.nii.gz'))
	background_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xproba-rcrbg_cropped.nii.gz'))
	T1map_masked_rightHemisphere = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_T1map_cropped.nii.gz'))
else:
	img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xmask-rcrgm.nii.gz'))
	tmp = img.get_fdata()
	tmp[tmp<0] = 0
	tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
	crop,coord = crop_img(tmp, pad=4, return_offset=True)

	tmp1 = nb.load(cortex['inside_mask'])
	tmp2 = tmp1.get_fdata()
	inside_mask = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_mask, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xmask-rcrwm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['inside_proba'])
	tmp2 = tmp1.get_fdata()
	inside_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xproba-rcrwm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['region_proba'])
	tmp2 = tmp1.get_fdata()
	region_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(region_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xproba-rcrgm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['background_proba'])
	tmp2 = tmp1.get_fdata()
	background_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(background_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xproba-rcrbg_cropped.nii.gz'))

	tmp1 = nb.load(T1map_masked)
	tmp2 = tmp1.get_fdata()
	T1map_masked_rightHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(T1map_masked_rightHemisphere, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_T1map_cropped.nii.gz'))

	del img
	del tmp
	del tmp1
	del tmp2
	print('Done.')

############################################################################
# 14.1. Crop additional data to right hemispehre
# -------------------
if reprocess_map_data:
	reprocess = True

if map_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Crop additional data to right hemisphere')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_rightHemisphere_cropped.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		map_data_rightHemisphere = os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_rightHemisphere_cropped.nii.gz')
	else:
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xmask-rcrgm.nii.gz'))
		tmp = img.get_fdata()
		tmp[tmp<0] = 0
		tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
		crop,coord = crop_img(tmp, pad=4, return_offset=True)

		tmp1 = nb.load(map_data)
		tmp2 = tmp1.get_fdata()
		map_data_rightHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
		nb.save(map_data_rightHemisphere, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_rightHemisphere_cropped.nii.gz'))

		del img
		del tmp
		del tmp1
		del tmp2
		print('Done.')

############################################################################
# 14.2. Crop additionally transformed data to right hemispehre
# -------------------
if map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Crop transformed data to right hemisphere')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	if isinstance(transform_data_output,list):
		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[-1] + '_rightHemisphere_cropped.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			transform_data_rightHemisphere = []
			for tmp in transform_data_output:
				transform_data_rightHemisphere.append(os.path.join(out_dir, 'sub-' + sub + '_' + tmp + '_rightHemisphere_cropped.nii.gz'))
		else:
			# Load grey matter image, binarize image, and get information for cropping
			img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xmask-rcrgm.nii.gz'))
			tmp = img.get_fdata()
			tmp[tmp<0] = 0
			tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
			crop,coord = crop_img(tmp, pad=4, return_offset=True)

			# Apply cropping
			length = len(transform_data)
			transform_data_rightHemisphere = []
			for i in range(length):
				tmp1 = nb.load(transform_data[i])
				tmp2 = tmp1.get_fdata()
				tmp3 = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
				nb.save(tmp3, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_rightHemisphere_cropped.nii.gz'))
				transform_data_rightHemisphere.append(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_rightHemisphere_cropped.nii.gz'))

			del img
			del tmp
			del tmp1
			del tmp2
			del tmp3
			print('Done.')

	else:
		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_rightHemisphere_cropped.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			transform_data_rightHemisphere = os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_rightHemisphere_cropped.nii.gz')
		else:
			# Load grey matter image, binarize image, and get information for cropping
			img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_xmask-rcrgm.nii.gz'))
			tmp = img.get_fdata()
			tmp[tmp<0] = 0
			tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
			crop,coord = crop_img(tmp, pad=4, return_offset=True)

			# Apply cropping
			tmp1 = nb.load(transform_data)
			tmp2 = tmp1.get_fdata()
			transform_data_rightHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
			nb.save(transform_data_rightHemisphere, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_rightHemisphere_cropped.nii.gz'))

			del img
			del tmp
			del tmp1
			del tmp2
			print('Done.')

if reprocess_map_data:
	reprocess = False

#############################################################################
# 15. CRUISE cortical reconstruction (right hemisphere)
# --------------------------------
# the WM inside mask as a (topologically spherical) starting point to grow a
# refined GM/WM boundary and CSF/GM boundary
cruise = nighres.cortex.cruise_cortex_extraction(
                        init_image=inside_mask,
                        wm_image=inside_proba,
                        gm_image=region_proba,
                        csf_image=background_proba,
			vd_image=None,
                        data_weight=0.8,
                        regularization_weight=0.2,
			max_iterations=1000,
                        normalize_probabilities=True,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + merged + '_rightHemisphere',
                        output_dir=out_dir)

###########################################################################
# 16. Extract layers across cortical sheet and map on surface (right hemisphere)
###########################################################################
# 16.1 Volumetric layering for depth measures (right hemisphere)
# ---------------------
# Finally, we use the GM/WM boundary (GWB) and CSF/GM boundary (CGB) from
# CRUISE to compute cortical depth with a volume-preserving technique.
# ---------------------
layers = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise['gwb'],
                        outer_levelset=cruise['cgb'],
                        n_layers=20,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers-allLayers',
                        output_dir=out_dir)['boundaries']
layers = nb.load(layers)

###########################################################################
# 16.2. Extract middle layer for surface generation and mapping (right hemisphere)
# ---------------------
# Here, we extract the levelset of the middle layer which will be used to
# map the information on.
print('')
print('*****************************************************')
print('* Extracting middle cortical layer (right hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-middleLayer.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = layers.get_fdata()[:,:,:,10:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-middleLayer.nii.gz'))
	del tmp

###########################################################################
# 16.3 Cortical surface generation and inflation of middle layer (right hemisphere)
# ---------------------
# Here, we create a surface from the levelset of the middle layer and
# inflate it afterwards for better visualisation.
corticalSurface = nighres.surface.levelset_to_mesh(
                        levelset_image=extractedLayers,
			connectivity='26/6',
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

inflatedSurface = nighres.surface.surface_inflation(
                        surface_mesh=corticalSurface['result'],
			max_iter=10000,
			max_curv=8.0,
			#regularization=1.0,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

###########################################################################
# 16.4. Profile sampling of all layers (right hemisphere)
# ---------------------
# Here, we sample the T1 values of the T1map onto all layers.
profile = nighres.laminar.profile_sampling(
                        profile_surface_image=layers,
                        intensity_image=T1map_masked_rightHemisphere,
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers')['result']
profile = nb.load(profile)

###########################################################################
# 16.5. Extract all cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 3 to 18, removing layers close to the white
# matter and cerebrospinal fluid to avoid partial volume effects.
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting all cortical layers (right hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-allLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers-allLayers.vtk')


###########################################################################
# 16.6. Extract deep cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 3 to 6, to cover the "deep" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting deep cortical layers (right hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-deepLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:7]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-deepLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-deepLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers-deepLayers.vtk')

###########################################################################
# 16.7. Extract inner middle cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 7 to 10, to cover the "inner middle" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting inner middle cortical layers (right hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-innerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,7:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-innerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-innerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers-innerLayers.vtk')

###########################################################################
# 16.8. Extract outer middle cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 11 to 14, to cover the "outer middle" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting outer middle cortical layers (right hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-outerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,11:15]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-outerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-outerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers-outerLayers.vtk')

###########################################################################
# 16.9. Extract superficial cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 15 to 18, to cover the "superficial" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting superficial cortical layers (right hemisphere)')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-superficialLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,15:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-superficialLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-superficialLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + merged + '_rightHemisphere_extractedLayers-superficialLayers.vtk')

###########################################################################
# 16.10. Profile sampling of additional data of all layers and mapping it onto
# the surface (right hemisphere)
# ---------------------
# Here, we sample the T1 values of the additional data onto all layers. And
# we extract the layers 1 to 18, removing layers close to the cerebrospinal
# fluid to avoid partial volume effects. Afterwards the information is
# mapped on the original and inflated surface of the middle layer.
if reprocess_map_data:
	reprocess = True

if map_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Profile sampling of additional data (right hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	profile = nighres.laminar.profile_sampling(
		                profile_surface_image=layers,
		                intensity_image=map_data_rightHemisphere,
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_' + map_data_output + '_rightHemisphere_extractedLayers')['result']
	profile = nb.load(profile)

	print('')
	print('*****************************************************')
	print('* Extracting all cortical layers of additional data (right hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	else:
		tmp = profile.get_fdata()[:,:,:,1:19]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_rightHemisphere_extractedLayers-allLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		                intensity_image=meanProfile,
		                surface_mesh=corticalSurface['result'],
		                inflated_mesh=inflatedSurface['result'],
		                mapping_method='closest_point',
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_' + map_data_output + '_rightHemisphere_extractedLayers-allLayers.vtk')

###########################################################################
# 16.11. Profile sampling of transformed data of all layers and mapping it onto
# the surface (right hemisphere)
# ---------------------
# Here, we sample the T1 values of the transformed data onto all layers.
# And we extract the layers 1 to 18, removing layers close to the
# cerebrospinal fluid to avoid partial volume effects. Afterwards the
# information is mapped on the original and inflated surface of the middle
# layer.
if map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Profile sampling of transformed data (right hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	if isinstance(transform_data_output,list):
		length = len(transform_data_output)
		for i in range(length):
			print('')
			print('*****************************************************')
			print('* Profile sampling and extracting all cortical layers of additional data (right hemisphere)')
			print('*****************************************************')
			profile = nighres.laminar.profile_sampling(
						profile_surface_image=layers,
						intensity_image=transform_data_rightHemisphere[i],
						save_data=True,
						overwrite=reprocess,
						output_dir=out_dir,
						file_name='sub-' + sub + '_' + transform_data_output[i] + '_rightHemisphere_extractedLayers')['result']
			profile = nb.load(profile)

			if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
				print('File exists already. Skipping process.')
				meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
			else:
				tmp = profile.get_fdata()[:,:,:,1:19]
				extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
				nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_rightHemisphere_extractedLayers-allLayers.nii.gz'))
				meanProfile = mean_img(extractedLayers)
				nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
				del tmp

			nighres.surface.surface_mesh_mapping(
						intensity_image=meanProfile,
						surface_mesh=corticalSurface['result'],
						inflated_mesh=inflatedSurface['result'],
						mapping_method='closest_point',
						save_data=True,
						overwrite=reprocess,
						output_dir=out_dir,
						file_name='sub-' + sub + '_' + transform_data_output[i] + '_rightHemisphere_extractedLayers-allLayers.vtk')
	else:
		profile = nighres.laminar.profile_sampling(
				        profile_surface_image=layers,
				        intensity_image=transform_data_rightHemisphere,
				        save_data=True,
				        overwrite=reprocess,
				        output_dir=out_dir,
				        file_name='sub-' + sub + '_' + transform_data_output + '_rightHemisphere_extractedLayers')['result']
		profile = nb.load(profile)

		print('')
		print('*****************************************************')
		print('* Extracting all cortical layers of additional data (right hemisphere)')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')

		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
		else:
			tmp = profile.get_fdata()[:,:,:,1:19]
			extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
			nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_rightHemisphere_extractedLayers-allLayers.nii.gz'))
			meanProfile = mean_img(extractedLayers)
			nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
			del tmp

		nighres.surface.surface_mesh_mapping(
				        intensity_image=meanProfile,
				        surface_mesh=corticalSurface['result'],
				        inflated_mesh=inflatedSurface['result'],
				        mapping_method='closest_point',
				        save_data=True,
				        overwrite=reprocess,
				        output_dir=out_dir,
				        file_name='sub-' + sub + '_' + transform_data_output + '_rightHemisphere_extractedLayers-allLayers.vtk')

# Set reprocess flag to false in case left hemisphere or mapping of
# additional data should be done only.
if reprocess_rightHemisphere or reprocess_map_data:
	reprocess = False

print('')
print('*****************************************************')
print('* Processing finished successfully at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
