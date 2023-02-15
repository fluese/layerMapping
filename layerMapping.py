'''
Mapping of quantitative T1 values from MP2RAGE data onto surface
=======================================

This is a pipeline processing BIDS formatted MP2RAGE data by performing the following steps:

01. Setup
02. MP2RAGE background cleaning
03. Resampling to 500 µm (hires pipeline only)
04. Imhomogeneity correction
05. Skull stripping
06. Registration of whole brain to slab data (hires pipeline only)
07. Weighted image combination of whole brain and slab data (hires pipeline only)
08. FreeSurfer
09. Atlas-guided tissue classification using MGDM
10. Extract hemisphere
11. Crop volume to hemisphere
12. CRUISE cortical reconstruction
13. Extract layers across cortical sheet and map on surface
14. Process addtional data (if data is specified)
15. Process transform data (if data is specified)

Version 1 (<3 14.02.2023 <3) 
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
# 0. Largest Component for white semgentation?
# 1. Change os.system to subprocess.run for MATLAB invocations
# 2. Hard coded resolution for high resolution pipeline should be set automatically based on slab's resolution
# 2. Get rid of MATLAB:
#	   * MP2RAGE background cleaning [Needs to be re-written]
# 	   * SPM's bias correction method [Probably by using FreeSurfer results]
#	   * Combination of high resolution slab and low resolution whole brain data [Note: Function is included at the end. However, MATLAB function provides better (?) results. At least using the Python implementation CRUISE produced more error. Maybe due to using the FreeSurfer segmentation this is not an issue anymore]
# 3. Flag to write all data to disk or final results only
# 4. Add logging feature
#

############################################################################
# Installation instructions
# -------------------
# Things needed to be installed:
# 1. Nighres (tested with v1.4: https://nighres.readthedocs.io/en/latest/installation.html)
# 2. antspy (tested with v0.3.2: https://github.com/ANTsX/ANTsPy)
# 3. FreeSurfer (tested with v7.3.2: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)
# 4. MATLAB (tested with R2022a)
# 5. SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/download/)
# 6. MP2RAGE-related-scripts (https://github.com/JosePMarques/MP2RAGE-related-scripts)
# 7. Custom MATLAB scripts (weightedAverage, removeBackgroundnoise and biasCorrection)
#
# You need to change the path to the tissue probability model for the bias
# field correction method. This needs to be done in
# 
# ./biasCorrection/preproc_sensemap.m on line 19
#

############################################################################
# Usage instructions
# -------------------
# This script requires whole brain MP2RAGE data organized according to BIDS.
# The script expects the first and second inversion, the T1 weighted data as
# well as the T1 map. You can make use of a high resolution MP2RAGE slab
# additionally. The slab will be merged into the whole brain volume
# automatically and used for further processing. To make use of a high 
# resolution slab, the slab needs to be acquired as the second run in a
# session (or at least named as if it were acquired in the same session).
#
# In the subsection "set parameters" of the setup section below, you need to
# specify the folder to the BIDS directory and the label of the subject you
# want to process. This can be done in multiple ways. First, specify the 
# path and file in the script directly. Secondly, when the script is called
# the next two inputs are for the subject ID and path to the BIDS directory,
# respectively. Lastly, if all is left empty, the user will be asked to
# specify a subject ID and the full path to the BIDS directory at the 
# beginning of the script.
#
# Example use (Case 1):
# python3 layerMapping.py aaa /path/to/BIDS/data/
#
# Example use (Case 2):
# python3 layerMapping.py aaa
#
# Example use (Case 3):
# python3 layerMapping.py
#
# In case 1 the subject with the ID 'aaa' will be processed. The according
# data is expected in '/path/to/BIDS/data/'. Output will be generated in
# '/path/to/BIDS/data/derivatives/sub-aaa/'. In case 2, either the path
# to the data needs to be defined in the script itself in the variable
# 'BIDS_path' or if left empty the user is asked to specify the path
# to the data. In case 3, the same for the path is true for the subject
# ID. It can either be specified in the script itself in the 'sub'
# variable or if left empty, the user will be asked for an ID during
# processing.
#
# Furthermore, you can specify a single volume which will be mapped on the
# surface additionally. This is meant for mapping another contrast  on the
# surface, e.g. QSM data or fMRI results. The resulting transformation can
# be applied either to another volume or to all compressed NIfTI files of
# an entire directory. Currently, this can be done in the script only.
#
# To enable logging of the output, one can send the screen output into a
# file. This can be done, e.g. by using following command:
# python3 -u layerMapping.py aaa |& tee -a /tmp/luesebrink/sensemap/derivatives/sub-aaa/layerMapping.log
#
# A few flags have been set up which allow to reprocess data at various
# stages of the pipeline (e.g. entirely, from the segmentation onwards,
# mapping of additional data, or per hemisphere). In case you want to 
# map multiple files onto the surface additionally, this is especially 
# useful as you don't have to process all other data again. Simply
# specify a file under 'map_data' that shall be mapped onto the surface
# and change the 'reprocess_map_data' flag to True.
#

############################################################################
# 1. Setup
# -------------------
# First, we import nighres and all other modules. Then, we will define all
# required parameters.

############################################################################
# 1.1. Import python libraries
# -------------------
import nighres
import os
import sys
import ants
import nibabel as nb
import glob
import subprocess
from nilearn.image import mean_img
from nilearn.image import crop_img
from nibabel.processing import conform
from time import localtime, strftime

############################################################################
# 1.2. Set parameters
# -------------------
# Define subject to be processed according to BIDS nomenclature here. If
# left empty, the user will be asked to specify the subject's ID during
# processing. Otherwise, the first input after the script is meant to
# specify the subject ID.
# sub = 'wtl'
sub = ''

# Define BIDS path here. If left empty, the user will be asked to specify
# the path during processing. Otherwise, the second input after the script
# is meant to specify the BIDS_path. In a future release this will be 
# changed for easier handling.
BIDS_path = ''
#BIDS_path = '/tmp/luesebrink/sensemap/'

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
# the surface. Could for example be the statistical maps of SPM from fMRI or the
# Chi map of QSM data.
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

# Choose interpolation method for mapping of additional data. Choice of
# interpolator can be 'linear', 'nearstNeighbor, 'bSpline', 'genericLabel'.
# See https://antspy.readthedocs.io/en/latest/_modules/ants/registration/apply_transforms.html 
# for full list of supported interpolators.
interpolation_method = 'nearestNeighbor'

# Flag to overwrite all existing data
reprocess = False

# Flag to start reprocessing from FreeSurfer onwards
reprocess_FreeSurfer = False

# Flag to start reprocessing from segmentation onwards
reprocess_segmentation = False

# Flag to reprocess left or right hemisphere only
reprocess_leftHemisphere = False
reprocess_rightHemisphere = False

# Flag to reprocess additional mapping (and transformation) data. This flag
# is especially useful if you want to map additional information onto the
# surface without re-running the entire pipeline.
# This could be the case if you have mapped functional data on the surface
# already already and now you also want to map QSM data onto the surface. 
# In this scenario you would want to set this flag to true, change the path
# of 'map_data' and 'transform_data' to your QSM data and the resulting
# Chi map, respectively. All other reprocess flags should be set to 'False'
# to avoid unnessecary reprocessing.
# The basename of the output will be based on the file name of the input
# data. According to BIDS the subject label will be added as prefix to all
# output data.
reprocess_map_data = False

# Define path from where data is to be copied into "BIDS_path". Could either
# be used to create a backup of the data before processing or transfer to a
# compute server.
# If the path does not exist, this option will be omitted and data from
# "BIDS_path" will be used instead.
# Unfortunately, does not work for network drives. -> Workaround?
copy_data_from = ''

# Define path to atlas. Here, we use custom priors which seem to work well
# with 7T MP2RAGE data collected in Magdeburg, Germany. Otherwise, please 
# choose the recent priors that come along with nighres (currently 3.0.9)
# instead of using the default atlas (currently 3.0.3). You can find that
# text file in the python package of nighres in the atlas folder.
atlas = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'atlas', 'brain-atlas-quant-3.0.9_customPriors.txt')

############################################################################
# 1.3. Define file names (and optionally create backup or copy data to
# compute server)
# -------------------
if not sub:
	if len(sys.argv) > 1:
		sub = sys.argv[1]
	else:
		sub = input('Please enter subject ID: ')
	
if not BIDS_path:
	if len(sys.argv) > 2:
		BIDS_path = sys.argv[2]
	else:
		BIDS_path = input('Please enter full path to BIDS directory: ')

# Set paths and create directories
in_dir = BIDS_path + 'sub-' + sub + '/anat/'
out_dir = BIDS_path + 'derivatives/sub-' + sub + '/'
subprocess.run(["mkdir", "-p", in_dir])
subprocess.run(["mkdir", "-p", out_dir])

print('*****************************************************')
print('* Processing pipeline started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
print('Working directory: ' + BIDS_path)
print('Subject to be processed: ' + sub)
open(os.path.join(out_dir, 'is_running'), mode='a').close()

# Define file names
if hires == True:
	INV1 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-1_MP2RAGE.nii.gz')
	INV2 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-2_MP2RAGE.nii.gz')
	T1map = os.path.join(in_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')
	UNI = os.path.join(in_dir, 'sub-' + sub + '_run-01_UNIT1.nii.gz')

	INV1_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_inv-1_MP2RAGE.nii.gz')
	INV2_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_inv-2_MP2RAGE.nii.gz')
	T1map_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_T1map.nii.gz')
	UNI_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_UNIT1.nii.gz')
else:
	INV1 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-1_MP2RAGE.nii.gz')
	INV2 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-2_MP2RAGE.nii.gz')
	T1map = os.path.join(in_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')
	UNI = os.path.join(in_dir, 'sub-' + sub + '_run-01_UNIT1.nii.gz')

# Copy data if path exists
print('')
print('*****************************************************')
print('* Data transfer to working directory.')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
if os.path.isdir(copy_data_from):
	if os.path.isfile(UNI)  and reprocess != True:
		print('Files exists already. Skipping data transfer.')
	else:
		subprocess.run(["scp", "-r", copy_data_from, " ", BIDS_path])
else:
	print('No path found to copy data from found. Continuing without copying data!')

# Check if data exists.
print('')
print('*****************************************************')
print('* Checking if files exists.')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
if os.path.isfile(INV1) and os.path.isfile(INV2) and os.path.isfile(T1map) and os.path.isfile(UNI):
	print('Files exists. Good!')
else:
	print('No files found. Exiting!')
	exit()

if hires == True:
	if os.path.isfile(INV1_slab) and os.path.isfile(INV2_slab) and os.path.isfile(T1map_slab) and os.path.isfile(UNI_slab):
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
	print('Could not find file for surface mapping. Omitting flag!')

# Check if file for applying the transformation of the additional data exists.
if os.path.isfile(transform_data) == False and os.path.isdir(transform_data) == False:
	map_transform_file_onto_surface = False
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
# 2. MP2RAGE background cleaning
# -------------------
# This script creates MP2RAGE T1w images without the strong background noise in
# air regions as implemented by Marques [Taken from his Github repository] using
# the method of O'Brien.
#
# O'Brien, et al, 2014.
# Robust T1-Weighted Structural Brain Imaging and Morphometry at 7T Using MP2RAGE
# PLOS ONE 9, e99676. doi:10.1371/journal.pone.0099676
# http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0099676
# -------------------
print('')
print('*****************************************************')
print('* MP2RAGE background cleaning')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
# Check for file and start MATLAB to conduct background cleaning.
if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	reg = str(10)
	output = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')
	#os.system('matlab -nosplash -nodisplay -r \"removeBackgroundnoise(\'' + UNI + '\', \'' + INV1 + '\', \'' + INV2 + '\', \'' + output + '\', ' + reg + '); exit;\"')
	os.system('matlab -nosplash -nodisplay -r \"removeBackgroundnoise(\'' + UNI + '\', \'' + INV1 + '\', \'' + INV2 + '\', \'' + output + '\', ' + reg + '); exit;\"')


# Check for high resolution slab data and start MATLAB to conduct background cleaning.
if hires == True:
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		output = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w.nii.gz')
		os.system('matlab -nosplash -nodisplay -r \"removeBackgroundnoise(\'' + UNI_slab + '\', \'' + INV1_slab + '\', \'' + INV2_slab + '\', \'' + output + '\', ' + reg + '); exit;\"')

# Update file names
T1w = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')
T1w_slab = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w.nii.gz')

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
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		input_img = nb.load(T1w)
		resampled_img = conform(input_img, out_shape=(336,448,448), voxel_size=(0.5,0.5,0.5), order=4)
		nb.save(resampled_img, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_resampled.nii.gz'))

		input_img = nb.load(T1map)
		resampled_img = conform(input_img, out_shape=(336,448,448), voxel_size=(0.5,0.5,0.5), order=4)
		nb.save(resampled_img, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled.nii.gz'))
		print('Done.')

	# Update file names
	T1map = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled.nii.gz')
	T1w = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_resampled.nii.gz')
else:
	# Copy and update file name
	subprocess.run(["cp", T1map, os.path.join(out_dir, "sub-" + sub + "_run-01_T1map.nii.gz")])
	T1map = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')

############################################################################
# 4. Inhomogeneity correction 
# ----------------
# Here, we perform inhomogeneity correction using SPM's segment routine.
print('')
print('*****************************************************')
print('* Inhomogeneity correction of whole brain data')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	os.system('matlab -nosplash -nodisplay -r \"preproc_sensemap(\'' + T1w + '\'); exit;\"')
	os.system('matlab -nosplash -nodisplay -r \"preproc_sensemap(\'' + T1map + '\'); exit;\"')

if hires == True:
	print('')
	print('*****************************************************')
	print('* Inhomogeneity correction of slab data')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	# Copy data and update file name
	subprocess.run(["cp", T1map_slab, os.path.join(out_dir, "sub-" + sub + "_run-02_T1map.nii.gz")])
	T1map_slab = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map.nii.gz')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		os.system('matlab -nosplash -nodisplay -r \"preproc_sensemap(\'' + T1w_slab + '\'); exit;\"')
		os.system('matlab -nosplash -nodisplay -r \"preproc_sensemap(\'' + T1map_slab + '\'); exit;\"')

############################################################################
# 5. Skull stripping
# ----------------
# Here, we perform skull stripping of T1w data using FreeSurfer's
# mri_synthstrip routine. Afterwards, we apply the mask to the T1map
# data for consistency.
print('')
print('*****************************************************')
print('* Skull stripping of whole brain data')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
T1w_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected.nii.gz')
T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected_masked.nii.gz')
mask = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_brainmask.nii.gz')

if os.path.isfile(T1w_biasCorrected_masked) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	subprocess.run(["mri_synthstrip", "-i", T1w_biasCorrected, "-o", T1w_biasCorrected_masked, "-m", mask, "-b", "-1"])

if hires == True:
	print('')
	print('*****************************************************')
	print('* Skull stripping of high resolution slab data')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	T1w_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected.nii.gz')
	T1w_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_masked.nii.gz')
	mask_slab = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_brainmask.nii.gz')
	
	if os.path.isfile(T1w_slab_biasCorrected_masked) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		subprocess.run(["mri_synthstrip", "-i", T1w_slab_biasCorrected, "-o", T1w_slab_biasCorrected_masked, "-m", mask, "-b", "-1"])

print('')
print('*****************************************************')
print('* Masking T1map of whole brain data')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
# Update file names	
T1w_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected.nii.gz')
T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_biasCorrected_masked.nii.gz')
T1map_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected.nii.gz')
T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected_masked.nii.gz')
mask = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_brainmask.nii.gz')

# Check if volume exists
if os.path.isfile(T1map_biasCorrected_masked) and reprocess != True:
	print('File exists already. Skipping process.')
else:
# Mask image otherwise
	maskedImage = ants.mask_image(ants.image_read(T1map), ants.image_read(mask))
	ants.image_write(maskedImage, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map' + resampled + '_biasCorrected_masked.nii.gz'))
	print('Done.')

if hires == True:
	print('')
	print('*****************************************************')
	print('* Masking T1map of high resolution slab data')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	# Update file names	
	T1w_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected.nii.gz')
	T1w_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_masked.nii.gz')
	T1map_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz')
	T1map_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_masked.nii.gz')
	mask_slab = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_brainmask.nii.gz')

	# Check if volume exists
	if os.path.isfile(T1map_slab_biasCorrected_masked) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
	# Mask image otherwise
		maskedImage = ants.mask_image(ants.image_read(T1map_slab_biasCorrected), ants.image_read(mask_slab))
		ants.image_write(maskedImage, os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_masked.nii.gz'))
		print('Done.')

############################################################################
# 6. Register data to upsampled 500 µm data
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

	if os.path.isfile(os.path.join(out_dir + 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_sub-' + sub + '_run-01_T1w_resampled_biasCorrected_masked.nii.gz')) and reprocess != True:
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
		warpedImage = ants.apply_transforms(
			fixed = ants.image_read(T1map_biasCorrected),
			moving = ants.image_read(T1map_slab),
	        transformlist = registeredImage['invtransforms'],
			interpolator = 'bSpline',
			verbose = True,
			)

		ants.image_write(warpedImage, out_dir + 'sub-' + sub + '_run-02_T1map_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')

		warpedImage = ants.apply_transforms(
			fixed = ants.image_read(T1map_biasCorrected),
			moving = ants.image_read(T1map_slab_biasCorrected),
	        transformlist = registeredImage['invtransforms'],
			interpolator = 'bSpline',
			verbose = True,
			)

		ants.image_write(warpedImage, out_dir + 'sub-' + sub + '_run-02_T1map_biasCorrected_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz') 

		warpedImage = ants.apply_transforms(
			fixed = ants.image_read(T1w_biasCorrected),
			moving = ants.image_read(T1w_slab_biasCorrected),
	        transformlist = registeredImage['invtransforms'],
			interpolator = 'bSpline',
			verbose = True,
			)

		ants.image_write(warpedImage, out_dir + 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_sub-' + sub + '_run-01_T1w_resampled_biasCorrected_masked.nii.gz') 
		
		warpedImage = ants.apply_transforms(
			fixed = ants.image_read(T1map_biasCorrected),
			moving = ants.image_read(mask_slab),
	        transformlist = registeredImage['invtransforms'],
			interpolator = 'bSpline',
			verbose = True,
			)

		ants.image_write(warpedImage, out_dir + 'sub-' + sub + '_run-02_T1map_brainmask_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz') 
		
	# Update file names
	T1w_slab_biasCorrected_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_sub-' + sub + '_run-01_T1w_resampled_biasCorrected_masked.nii.gz')
	T1map_slab_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')
	T1map_slab_biasCorrected_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')
	mask_slab_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_brainmask_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz') 

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
	else:
		# Update file names
		T1w_WM = os.path.join(out_dir, 'c2sub-' + sub + '_run-01_T1w_resampled.nii.gz')
		T1map_WM = os.path.join(out_dir, 'c2sub-' + sub + '_run-01_T1map_resampled.nii.gz')

		# Combine data using custom MATLAB script
		os.system('matlab -nosplash -nodisplay -r \"weightedAverage(\'' + T1w_biasCorrected + '\', \'' + T1w_slab_biasCorrected_reg + '\', \'' + T1w_WM + '\', \'' + mask + '\', \'' + out_dir + 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected.nii.gz' + '\'); exit;\"')
		os.system('matlab -nosplash -nodisplay -r \"weightedAverage(\'' + T1map + '\', \'' + T1map_slab_reg + '\', \'' + T1map_WM + '\', \'' + mask + '\', \'' + out_dir + 'sub-' + sub + '_merged_run-01+02_T1map.nii.gz' + '\'); exit;\"')
		
		os.system('matlab -nosplash -nodisplay -r \"weightedAverage(\'' + T1map_biasCorrected + '\', \'' + T1map_slab_biasCorrected_reg + '\', \'' + T1map_WM + '\', \'' + mask + '\', \'' + out_dir + 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected.nii.gz' + '\'); exit;\"')

	# Update file names
	T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected_masked.nii.gz')
	T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_masked.nii.gz')
	T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected_masked.nii.gz')

###########################################################################
# 8.1. Run FreeSurfer to produce segmentation
# 	   [Could be exchanged by mri_synthseg potentially, less accurate, but much faster]
#	   [If the entire FreeSurfer pipeline is run, one could make use of more results,
#		e.g. bias field correction and white matter segmentation to get rid of MATLAB
#		dependencies].
# ------------------
# Here, we run FreeSurfer in order to produce an accurate voxel-base 
# segmentation of the cortex. This is a bit overkill, but there is no other
# working solution so far. In order to change the segmentation algorithm
# one needs to change the command in line 998.
print('')
print('*****************************************************')
print('* Process T1w data with FreeSurfer')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')

FreeSurfer_path = os.path.join(BIDS_path, 'derivatives', 'FreeSurfer')
subID = 'sub-' + sub + merged + '_T1w_biasCorrected'

if os.path.isfile(os.path.join(FreeSurfer_path, subID, 'scripts', 'recon-all-status.log')):
	with open(os.path.join(FreeSurfer_path, subID, 'scripts', 'recon-all-status.log'), 'r') as f:
		finished = f.readlines()[-1]
	    
		if "without error" not in finished:
			reprocess_FreeSurfer = True

if reprocess_FreeSurfer:
	reprocess = True

if os.path.isfile(os.path.join(FreeSurfer_path, subID, 'scripts', 'recon-all-status.log')) and reprocess != True:
	print('Files exist already. Skipping process.')
else:
	# Run FreeSurfer with hires flag
	subprocess.run(["recon-all", "-all", "-hires", "-sd", FreeSurfer_path, "-i", T1w_biasCorrected, "-s", subID])

###########################################################################
# 8.2. Prepare FreeSurfer segmentation for nighres
# ------------------
# Here, we transform the segmentation of FreeSurfer into the original space
# and extract the white matter, gray matter as well as everything else (ee)
# labels from FreeSurfer's segmentation. 
print('')
print('*****************************************************')
print('* Preparing FreeSurfer segmentation for nighres')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_ee_binary.nii.gz')) and reprocess != True:
	print('Files exist already. Skipping process.')
else:
	# Re-slice segmentation to input dimension [from conformed FreeSurfer space] using nearest neighbor interpolation
	subprocess.run(["mri_convert", "-rt", "nearest", "-odt", "int", "-rl", os.path.join(FreeSurfer_path, subID, 'mri', 'orig', '001.mgz'), "-i", os.path.join(FreeSurfer_path, subID, 'mri', 'aseg.mgz'), "-o", os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_whole_brain_FreeSurfer.nii.gz')])

	# Extract white matter with subcortial regions from aseg
	subprocess.run(["mri_binarize", "--match", "2", "41", "4", "43", "5", "44", "10", "49", "11", "50", "12", "51", "13", "52", "14", "15", "17", "53", "18", "54", "26", "58", "28", "60",  "31", "63", "77", "--i", os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_whole_brain_FreeSurfer.nii.gz'), "--o", os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_wm_binary.nii.gz') ])

	# Extract gray matter from aseg
	subprocess.run(["mri_binarize", "--match", "3", "42", "--i", os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_whole_brain_FreeSurfer.nii.gz'), "--o",  os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_gm_binary.nii.gz')])

	# Extract everything else from aseg
	subprocess.run(["mri_binarize", "--match", "0", "7", "46", "8", "47", "16", "24", "30", "62", "80", "85", "251", "252", "253", "254", "255", "--i", os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_whole_brain_FreeSurfer.nii.gz'), "--o", os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_ee_binary.nii.gz')])

###########################################################################
# 8.3. Topology correction of white matter segmentation
# ------------------
# Here, we apply the topology correction module of nighres to the white
# matter segmentation. This is needed for the CRUISE algorithm.
print('')
print('*****************************************************')
print('* Topology correction of white matter segmentation')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
topology = nighres.shape.topology_correction(
	os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_wm_binary.nii.gz'),
	shape_type='binary_object',
	connectivity='wcs',
	propagation='object->background',
	minimum_distance=1e-06,
	save_data=True,
	overwrite=reprocess,
	output_dir=out_dir,
	file_name='sub-' + sub + merged + '_segmentation_wm_binary')

###########################################################################
# 8.4. Turn binary segmentations into probability maps
# ------------------
# Here, we take the binary segmentation from FreeSurfer and turn them into
# probability maps by first creating levelsets from the binary images and
# then creating probability maps from the levelsets with a distance of 0.5 
# mm.
print('')
print('*****************************************************')
print('* Turn binary segmentations into probability maps')
print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
print('*****************************************************')
p2l = nighres.surface.probability_to_levelset(
	os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_wm_binary.nii.gz'),
	save_data=True,
	overwrite=reprocess,
	output_dir=out_dir,
	file_name='sub-' + sub + merged)

nighres.surface.levelset_to_probability(
	p2l['result'],
	distance_mm=0.5,
	save_data=True,
	overwrite=reprocess,
	output_dir=out_dir,
	file_name='sub-' + sub + merged + '_segmentation_wm_binary')

p2l = nighres.surface.probability_to_levelset(
	os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_gm_binary.nii.gz'),
	save_data=True,
	overwrite=reprocess,
	output_dir=out_dir,
	file_name='sub-' + sub + merged)

nighres.surface.levelset_to_probability(
	p2l['result'],
	distance_mm=0.5,
	save_data=True,
	overwrite=reprocess,
	output_dir=out_dir,
	file_name='sub-' + sub + merged + '_segmentation_gm')

p2l = nighres.surface.probability_to_levelset(
	os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_ee_binary.nii.gz'),
	save_data=True,
	overwrite=reprocess,
	output_dir=out_dir,
	file_name='sub-' + sub + merged)

nighres.surface.levelset_to_probability(
	p2l['result'],
	distance_mm=0.5,
	save_data=True,
	overwrite=reprocess,
	output_dir=out_dir,
	file_name='sub-' + sub + merged + '_segmentation_ee')

#############################################################################
# 9. MGDM classification
# ---------------------
# Next, we use the masked data as input for tissue classification with the
# MGDM algorithm. MGDM works with a single contrast, but can be improved
# with additional contrasts potentially. Here, we make use of the T1map 
# only.
#
# The resulting segmentation is less accurate then the segmentation of
# FreeSurfer. Therefore, currently the segmentation results are used for
# separation of the hemispheres only - which is done to reduce memory
# consumption.
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

mgdm = nighres.brain.mgdm_segmentation(
	#contrast_image2=T1w_biasCorrected_masked,
	#contrast_type2='Mp2rage7T',
	contrast_image1=T1map_biasCorrected_masked,
	contrast_type1='T1map7T',
	n_steps=20,
	max_iterations=1000,
	atlas_file=atlas,
	normalize_qmaps=False,
	#adjust_intensity_priors=True,
	compute_posterior=True,
	posterior_scale=5.0,
	save_data=True,
	overwrite=False,
	output_dir=out_dir,
	file_name='sub-' + sub + merged)


# ###########################################################################
# Process per hemisphere
# ------------------
# From here on the processes will be run per hemisphere to reduce memory
# demands.
for hemi in ["left", "right"]:

	# Adjust naming scheme of mask for cropping
	if hemi == "left":
		crop_mask = 'lcrgm'
	elif hemi == "right":
		crop_mask = 'rcrgm'
		
	###########################################################################
	# 10. Extract hemisphere
	# ------------------
	# Here, we pull from the MGDM output the needed files to extract the
	# hemisphere.
	print('')
	print('*****************************************************')
	print('* Region extraction of ' + hemi + ' hemisphere')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
	if locals()['reprocess_' + hemi + 'Hemisphere']:
		reprocess = True

	cortex = nighres.brain.extract_brain_region(
		segmentation=mgdm['segmentation'],
		levelset_boundary=mgdm['distance'],
		maximum_membership=mgdm['memberships'],
		maximum_label=mgdm['labels'],
		atlas_file=atlas,
		extracted_region=hemi + '_cerebrum',
		estimate_tissue_densities=True,
		partial_volume_distance=1.0,
		save_data=True,
		overwrite=reprocess,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere',
		output_dir=out_dir)
		    
	#############################################################################
	# 11. Crop volume to hemisphere
	# --------------------------------
	# Here, we crop the volume to the hemisphere based on the output of extract
	# brain region. This will reduce the memory demand tremendously as well as
	# processing time without sacrificing any accuracy.
	print('')
	print('*****************************************************')
	print('* Crop volume to ' + hemi + ' hemisphere')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_T1map_cropped.nii.gz')) and reprocess != True:
		print('Files exists already. Skipping process.')
		inside_mask = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_binary_tpc-obj_cropped.nii.gz'))
		inside_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_l2p-proba_cropped.nii.gz'))
		region_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_gm_l2p-proba_cropped.nii.gz'))
		background_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_ee_l2p-proba_cropped.nii.gz'))
		T1map_masked_cropped = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_T1map_cropped.nii.gz'))
	else:
		# Load grey matter image, binarize image, and get information for cropping
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_xmask-' + crop_mask + '.nii.gz'))
		tmp = img.get_fdata()
		tmp[tmp<0] = 0
		tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
		crop,coord = crop_img(tmp, pad=4, return_offset=True)

		# As padding of the cropped image does not work on the coordinates, we
		# need to include more voxels. These will be set to zero to make sure
		# voxel at the border are zero.
		coord = list(coord)
		tmp_0 = slice(coord[0].start - 5, coord[0].stop + 5, coord[0].step)
		tmp_1 = slice(coord[1].start - 5, coord[1].stop + 5, coord[1].step)
		tmp_2 = slice(coord[2].start - 5, coord[2].stop + 5, coord[2].step)
		coord = tuple((tmp_0, tmp_1, tmp_2))

		del tmp_0
		del tmp_1
		del tmp_2

		# Apply cropping to binary white matter mak
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_wm_binary.nii.gz'))
		tmp = img.get_fdata()
		inside_mask = nb.Nifti1Image(tmp[coord], affine=img.affine, header=img.header)
		nb.save(inside_mask, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_binary_tpc-obj_cropped.nii.gz'))

		# Set voxels close to the image border to zero, otherwise nighres will fail. Probably it would be easier to use nibable instead of ANTs, but it's working. Shoould probably be done in a function anyways as this is used multiple times...
		inside_mask = ants.image_read(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_binary_tpc-obj_cropped.nii.gz'))
		inside_mask[:-4:-1,:,:] = 0
		inside_mask[:,:-4:-1,:] = 0
		inside_mask[:,:,:-4:-1] = 0
		inside_mask[0:4,:,:] = 0
		inside_mask[:,0:4,:] = 0
		inside_mask[:,:,0:4] = 0
		ants.image_write(inside_mask, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_binary_tpc-obj_cropped.nii.gz'))

		# Apply cropping to probability white matter mask
		#img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_wm_l2p-proba.nii.gz'))
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_wm_binary.nii.gz'))
		tmp = img.get_fdata()
		inside_proba = nb.Nifti1Image(tmp[coord], affine=img.affine, header=img.header)
		nb.save(inside_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_l2p-proba_cropped.nii.gz'))

		# Set voxels close to the image border to zero, otherwise nighres will fail.
		inside_proba = ants.image_read(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_l2p-proba_cropped.nii.gz'))
		inside_proba[:-4:-1,:,:] = 0
		inside_proba[:,:-4:-1,:] = 0
		inside_proba[:,:,:-4:-1] = 0
		inside_proba[0:4,:,:] = 0
		inside_proba[:,0:4,:] = 0
		inside_proba[:,:,0:4] = 0
		ants.image_write(inside_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_l2p-proba_cropped.nii.gz'))

		# Apply cropping to probability grey matter mask
		#img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_gm_l2p-proba.nii.gz'))
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_gm_binary.nii.gz'))
		tmp = img.get_fdata()
		region_proba = nb.Nifti1Image(tmp[coord], affine=img.affine, header=img.header)
		nb.save(region_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_gm_l2p-proba_cropped.nii.gz'))

		# Set voxels close to the image border to zero, otherwise nighres will fail.
		region_proba = ants.image_read(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_gm_l2p-proba_cropped.nii.gz'))
		region_proba[:-4:-1,:,:] = 0
		region_proba[:,:-4:-1,:] = 0
		region_proba[:,:,:-4:-1] = 0
		region_proba[0:4,:,:] = 0
		region_proba[:,0:4,:] = 0
		region_proba[:,:,0:4] = 0
		ants.image_write(region_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_gm_l2p-proba_cropped.nii.gz'))

		# Apply cropping to probability mask of everything else
		#img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_ee_l2p-proba.nii.gz'))
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_segmentation_ee_binary.nii.gz'))
		tmp = img.get_fdata()
		background_proba = nb.Nifti1Image(tmp[coord], affine=img.affine, header=img.header)
		nb.save(background_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_ee_l2p-proba_cropped.nii.gz'))

		# Set voxels close to the image border to zero, otherwise nighres will fail.
		background_proba = ants.image_read(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_ee_l2p-proba_cropped.nii.gz'))
		background_proba[:-4:-1,:,:] = 1
		background_proba[:,:-4:-1,:] = 1
		background_proba[:,:,:-4:-1] = 1
		background_proba[0:4,:,:] = 1
		background_proba[:,0:4,:] = 1
		background_proba[:,:,0:4] = 1

		ants.image_write(background_proba, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_ee_l2p-proba_cropped.nii.gz'))

		# Apply cropping to T1map
		img = nb.load(T1map_masked)
		tmp = img.get_fdata()
		T1map_masked = nb.Nifti1Image(tmp[coord], affine=img.affine, header=img.header)
		nb.save(T1map_masked, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_T1map_cropped.nii.gz'))

		# Set voxels close to the image border to zero, otherwise nighres will fail.
		tmp = ants.image_read(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_T1map_cropped.nii.gz'))
		tmp[:-4:-1,:,:] = 0
		tmp[:,:-4:-1,:] = 0
		tmp[:,:,:-4:-1] = 0
		tmp[0:4,:,:] = 0
		tmp[:,0:4,:] = 0
		tmp[:,:,0:4] = 0
		ants.image_write(tmp, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_T1map_cropped.nii.gz'))
		
		# Update file names
		inside_mask = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_binary_tpc-obj_cropped.nii.gz'))
		inside_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_wm_l2p-proba_cropped.nii.gz'))
		region_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_gm_l2p-proba_cropped.nii.gz'))
		background_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_segmentation_ee_l2p-proba_cropped.nii.gz'))
		T1map_masked_cropped = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_T1map_cropped.nii.gz'))

		del img
		del tmp
		print('Done.')

	#############################################################################
	# 12. CRUISE cortical reconstruction
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
		data_weight=0.9,
		regularization_weight=0.1,
		max_iterations=5000,
		normalize_probabilities=False,
		save_data=True,
		overwrite=reprocess,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere',
		output_dir=out_dir)

	###########################################################################
	# 13. Extract layers across cortical sheet and map on surface
	###########################################################################
	# 13.1 Volumetric layering for depth measures
	# ---------------------
	# Finally, we use the GM/WM boundary (GWB) and CSF/GM boundary (CGB) from
	# CRUISE to compute cortical depth with a volume-preserving technique.
	print('')
	print('*****************************************************')
	print('* Extract layers across the cortical sheet (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	layers = nighres.laminar.volumetric_layering(
		inner_levelset=cruise['gwb'],
		outer_levelset=cruise['cgb'],
		n_layers=20,
		save_data=True,
		overwrite=reprocess,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-allLayers',
		output_dir=out_dir)['boundaries']

	layers = nb.load(layers)

	###########################################################################
	# 13.2. Extract middle layer for surface generation and mapping
	# ---------------------
	# Here, we extract the levelset of the middle layer which will be used to
	# map the information on.
	print('')
	print('*****************************************************')
	print('* Extracting middle cortical layer (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-middleLayer.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		extractedLayers=nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-middleLayer.nii.gz'))
	else:
		tmp = layers.get_fdata()[:,:,:,10:11]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-middleLayer.nii.gz'))
		del tmp

	###########################################################################
	# 13.3 Cortical surface generation and inflation of middle layer
	# ---------------------
	# Here, we create a surface from the levelset of the middle layer and
	# inflate it afterwards for better visualisation.
	print('')
	print('*****************************************************')
	print('* Generate surface of middle cortical layer and inflate it (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	corticalSurface = nighres.surface.levelset_to_mesh(
		                    levelset_image=extractedLayers,
				connectivity='26/6',
		                    save_data=True,
		                    overwrite=reprocess,
		                    file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-middleLayer.vtk',
		                    output_dir=out_dir)

	inflatedSurface = nighres.surface.surface_inflation(
		                    surface_mesh=corticalSurface['result'],
				max_iter=10000,
				max_curv=8.0,
				#regularization=1.0,
		                    save_data=True,
		                    overwrite=reprocess,
		                    file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-middleLayer.vtk',
		                    output_dir=out_dir)

	###########################################################################
	# 13.4. Profile sampling of all layers 
	# ---------------------
	# Here, we sample the T1 values of the T1map onto all layers.
	print('')
	print('*****************************************************')
	print('* Profile sampling across cortical sheet (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	profile = nighres.laminar.profile_sampling(
		profile_surface_image=layers,
		intensity_image=T1map_masked_cropped,
		save_data=True,
		overwrite=reprocess,
		output_dir=out_dir,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers')['result']

	profile = nb.load(profile)

	###########################################################################
	# 13.5. Extract all cortical layers and map on surface 
	# ---------------------
	# Here, we extract the layers 3 to 18, removing layers close to the white
	# matter and cerebrospinal fluid to avoid partial volume effects.
	# Afterwards the information is mapped on the original and inflated
	# surface of the middle layer.
	print('')
	print('*****************************************************')
	print('* Extracting all cortical layers (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz'))
	else:
		tmp = profile.get_fdata()[:,:,:,3:19]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-allLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		intensity_image=meanProfile,
		surface_mesh=corticalSurface['result'],
		inflated_mesh=inflatedSurface['result'],
		mapping_method='closest_point',
		save_data=True,
		overwrite=reprocess,
		output_dir=out_dir,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-allLayers.vtk')


	###########################################################################
	# 13.6. Extract deep cortical layers and map on surface 
	# ---------------------
	# Here, we extract the layers 3 to 6, to cover the "deep" layers. 
	# Afterwards the information is mapped on the original and inflated
	# surface of the middle layer.
	print('')
	print('*****************************************************')
	print('* Extracting deep cortical layers (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-deepLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		tmp = profile.get_fdata()[:,:,:,3:7]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-deepLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-deepLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		intensity_image=meanProfile,
		surface_mesh=corticalSurface['result'],
		inflated_mesh=inflatedSurface['result'],
		mapping_method='closest_point',
		save_data=True,
		overwrite=reprocess,
		output_dir=out_dir,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-deepLayers.vtk')

	###########################################################################
	# 13.7. Extract inner middle cortical layers and map on surface 
	# ---------------------
	# Here, we extract the layers 7 to 10, to cover the "inner middle" layers. 
	# Afterwards the information is mapped on the original and inflated
	# surface of the middle layer.
	print('')
	print('*****************************************************')
	print('* Extracting inner middle cortical layers (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-innerLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		tmp = profile.get_fdata()[:,:,:,7:11]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-innerLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-innerLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		intensity_image=meanProfile,
		surface_mesh=corticalSurface['result'],
		inflated_mesh=inflatedSurface['result'],
		mapping_method='closest_point',
		save_data=True,
		overwrite=reprocess,
		output_dir=out_dir,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-innerLayers.vtk')

	###########################################################################
	# 13.8. Extract outer middle cortical layers and map on surface 
	# ---------------------
	# Here, we extract the layers 11 to 14, to cover the "outer middle" layers. 
	# Afterwards the information is mapped on the original and inflated
	# surface of the middle layer.
	print('')
	print('*****************************************************')
	print('* Extracting outer middle cortical layers (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-outerLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		tmp = profile.get_fdata()[:,:,:,11:15]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-outerLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-outerLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		intensity_image=meanProfile,
		surface_mesh=corticalSurface['result'],
		inflated_mesh=inflatedSurface['result'],
		mapping_method='closest_point',
		save_data=True,
		overwrite=reprocess,
		output_dir=out_dir,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-outerLayers.vtk')

	###########################################################################
	# 13.9. Extract superficial cortical layers and map on surface 
	# ---------------------
	# Here, we extract the layers 15 to 18, to cover the "superficial" layers. 
	# Afterwards the information is mapped on the original and inflated
	# surface of the middle layer.
	print('')
	print('*****************************************************')
	print('* Extracting superficial cortical layers (' + hemi + ' hemisphere)')
	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-superficialLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		tmp = profile.get_fdata()[:,:,:,15:19]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-superficialLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-superficialLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		intensity_image=meanProfile,
		surface_mesh=corticalSurface['result'],
		inflated_mesh=inflatedSurface['result'],
		mapping_method='closest_point',
		save_data=True,
		overwrite=reprocess,
		output_dir=out_dir,
		file_name='sub-' + sub + merged + '_' + hemi + 'Hemisphere_extractedLayers-superficialLayers.vtk')

	############################################################################
	# 14. Process additional data.
	# -------------------
	if reprocess_map_data:
		reprocess = True

	############################################################################
	# 14.1 Register additional data to (upsampled) T1map
	# -------------------
	if map_file_onto_surface:
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

		if os.path.isfile(os.path.join(out_dir, reg1 + '.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			map_data = os.path.join(out_dir, reg1 + '.nii.gz')
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
				interpolator = interpolation_method,
				verbose = True,
				)

			# Write file to disk
			ants.image_write(warpedImage, out_dir + reg1 + '.nii.gz') 

			# Update file name
			map_data = os.path.join(out_dir, reg1 + '.nii.gz')

	############################################################################
	# 14.2. Crop additional data to hemispehre
	# -------------------
	if map_file_onto_surface:
		print('')
		print('*****************************************************')
		print('* Crop additional data to ' + hemi + ' hemisphere')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')
		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			map_data = os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz')
		else:
			# Load grey matter image, binarize image, and get information for cropping
			img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_xmask-' + crop_mask + '.nii.gz'))
			tmp = img.get_fdata()
			tmp[tmp<0] = 0
			tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
			crop,coord = crop_img(tmp, pad=4, return_offset=True)

			# As padding of the cropped image does not work on the coordinates, we
			# need to include more voxels. These will be set to zero to make sure
			# voxel at the border are zero.
			coord = list(coord)
			tmp_0 = slice(coord[0].start - 5, coord[0].stop + 5, coord[0].step)
			tmp_1 = slice(coord[1].start - 5, coord[1].stop + 5, coord[1].step)
			tmp_2 = slice(coord[2].start - 5, coord[2].stop + 5, coord[2].step)
			coord = tuple((tmp_0, tmp_1, tmp_2))

			del tmp_0
			del tmp_1
			del tmp_2

			# Apply cropping
			tmp1 = nb.load(map_data)
			tmp2 = tmp1.get_fdata()
			
			# Probably the hemisphere can be left out anyways to get rid of the bad syntax.
			map_data = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
			nb.save(map_data, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz'))
			
			# Set voxels close to the image border to zero, otherwise nighres will fail. Probably it would be easier to use nibable instead of ANTs, but it's working...
			tmp = ants.image_read(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz'))
			tmp[:-4:-1,:,:] = 0
			tmp[:,:-4:-1,:] = 0
			tmp[:,:,:-4:-1] = 0
			tmp[0:4,:,:] = 0
			tmp[:,0:4,:] = 0
			tmp[:,:,0:4] = 0
			ants.image_write(tmp, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz'))

			del img
			del tmp
			del tmp1
			del tmp2
			print('Done.')

	###########################################################################
	# 14.3. Profile sampling of additional data of all layers and mapping it
	# 		onto the surface 
	# ---------------------
	# Here, we sample the information of the additional data onto all layers.
	# And we extract the layers 1 to 18, removing layers close to CSF to avoid
	# partial volume effects. Afterwards the information is mapped on the
	# original and inflated surface of the middle layer.
	if map_file_onto_surface:
		print('')
		print('*****************************************************')
		print('* Profile sampling of additional data (' + hemi + ' hemisphere)')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')
		profile = nighres.laminar.profile_sampling(
			profile_surface_image=layers,
			intensity_image=map_data,
			save_data=True,
			overwrite=reprocess,
			output_dir=out_dir,
			file_name='sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_extractedLayers')['result']

		profile = nb.load(profile)

		print('')
		print('*****************************************************')
		print('* Extracting all cortical layers of additional data (' + hemi + ' hemisphere)')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')

		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
			print('File exists already. Skipping process.')
			meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz'))
		else:
			tmp = profile.get_fdata()[:,:,:,1:19]
			extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
			nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers.nii.gz'))
			meanProfile = mean_img(extractedLayers)
			nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz'))
			del tmp

		nighres.surface.surface_mesh_mapping(
		    intensity_image=meanProfile,
		    surface_mesh=corticalSurface['result'],
		    inflated_mesh=inflatedSurface['result'],
		    mapping_method='closest_point',
		    save_data=True,
		    overwrite=reprocess,
		    output_dir=out_dir,
		    file_name='sub-' + sub + '_' + map_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers.vtk')

	############################################################################
	# 15. Process transformed data.
	# -------------------
	if reprocess_map_data:
		reprocess = True

	############################################################################
	# 15.1 Apply transformation to data to be mapped on the surface
	# -------------------
	# Define file name for output of transformation.
	if map_file_onto_surface and map_transform_file_onto_surface:
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
						interpolator = interpolation_method,
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
					interpolator = interpolation_method,
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

	############################################################################
	# 15.1. Crop additionally transformed data to hemisphere
	# -------------------
	if map_transform_file_onto_surface:
		print('')
		print('*****************************************************')
		print('* Crop transformed data to ' + hemi + ' hemisphere')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')
		# Check if a list of multiple files exists to be transformed
		if isinstance(transform_data_output,list):
			if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[-1] + '_' + hemi + 'Hemisphere_cropped.nii.gz')) and reprocess != True:
				print('File exists already. Skipping process.')
				transform_data = []
				for tmp in transform_data_output:
					transform_data.append(os.path.join(out_dir, 'sub-' + sub + '_' + tmp + '_' + hemi + 'Hemisphere_cropped.nii.gz'))
			else:
				# Load grey matter image, binarize image, and get information for cropping
				img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_xmask-' + crop_mask + '.nii.gz'))
				tmp = img.get_fdata()
				tmp[tmp<0] = 0
				tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
				crop,coord = crop_img(tmp, pad=4, return_offset=True)

				# As padding of the cropped image does not work on the coordinates, we
				# need to include more voxels. These will be set to zero to make sure
				# voxel at the border are zero.
				coord = list(coord)
				tmp_0 = slice(coord[0].start - 5, coord[0].stop + 5, coord[0].step)
				tmp_1 = slice(coord[1].start - 5, coord[1].stop + 5, coord[1].step)
				tmp_2 = slice(coord[2].start - 5, coord[2].stop + 5, coord[2].step)
				coord = tuple((tmp_0, tmp_1, tmp_2))
				
				del tmp_0
				del tmp_1
				del tmp_2

				# Apply cropping
				length = len(transform_data)
				transform_data = []
				for i in range(length):
					tmp1 = nb.load(transform_data[i])
					tmp2 = tmp1.get_fdata()
					tmp3 = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
					nb.save(tmp3, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_cropped.nii.gz'))
									
					# Set voxels close to the image border to zero, otherwise nighres will fail. Probably it would be easier to use nibable instead of ANTs, but it's working...
					tmp = ants.image_read(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_cropped.nii.gz'))
					tmp[:-4:-1,:,:] = 0
					tmp[:,:-4:-1,:] = 0
					tmp[:,:,:-4:-1] = 0
					tmp[0:4,:,:] = 0
					tmp[:,0:4,:] = 0
					tmp[:,:,0:4] = 0
					ants.image_write(tmp, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_cropped.nii.gz'))

					transform_data.append(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_cropped.nii.gz'))

				del img
				del tmp
				del tmp1
				del tmp2
				del tmp3
				print('Done.')

		else:
			if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz')) and reprocess != True:
				print('File exists already. Skipping process.')
				transform_data = os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz')
			else:
				# Load grey matter image, binarize image, and get information for cropping
				img = nb.load(os.path.join(out_dir, 'sub-' + sub + merged + '_' + hemi + 'Hemisphere_xmask-' + crop_mask + '.nii.gz'))
				tmp = img.get_fdata()
				tmp[tmp<0] = 0
				tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
				crop,coord = crop_img(tmp, pad=4, return_offset=True)

				# As padding of the cropped image does not work on the coordinates, we
				# need to include more voxels. These will be set to zero to make sure
				# voxel at the border are zero.
				coord = list(coord)
				tmp_0 = slice(coord[0].start - 5, coord[0].stop + 5, coord[0].step)
				tmp_1 = slice(coord[1].start - 5, coord[1].stop + 5, coord[1].step)
				tmp_2 = slice(coord[2].start - 5, coord[2].stop + 5, coord[2].step)
				coord = tuple((tmp_0, tmp_1, tmp_2))
				
				del tmp_0
				del tmp_1
				del tmp_2

				# Apply cropping
				tmp1 = nb.load(transform_data)
				tmp2 = tmp1.get_fdata()
				transform_data = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
				nb.save(transform_data, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz'))

				tmp = ants.image_read(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz'))
				tmp[:-4:-1,:,:] = 0
				tmp[:,:-4:-1,:] = 0
				tmp[:,:,:-4:-1] = 0
				tmp[0:4,:,:] = 0
				tmp[:,0:4,:] = 0
				tmp[:,:,0:4] = 0
				ants.image_write(tmp, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_cropped.nii.gz'))

				del img
				del tmp
				del tmp1
				del tmp2
				print('Done.')

	###########################################################################
	# 15.2. Profile sampling of transformed data of all layers and mapping it
	# 		onto the surface 
	# ---------------------
	# Here, we sample the information of the transformed data onto all layers.
	# And we extract the layers 1 to 18, removing layers close to CSF to avoid
	# partial volume effects. Afterwards the information is mapped on the original
	# and inflated surface of the middle layer.
	if map_transform_file_onto_surface:
		print('')
		print('*****************************************************')
		print('* Profile sampling of transformed data (' + hemi + ' hemisphere)')
		print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print('*****************************************************')	

		if isinstance(transform_data_output,list):
			length = len(transform_data_output)
			for i in range(length):
				print('')
				print('*****************************************************')
				print('* Profile sampling and extracting all cortical layers of additional data (' + hemi + ' hemisphere)')
				print('*****************************************************')

				profile = nighres.laminar.profile_sampling(
					profile_surface_image=layers,
					intensity_image=transform_data[i],
					save_data=True,
					overwrite=reprocess,
					output_dir=out_dir,
					file_name='sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_extractedLayers')['result']

				profile = nb.load(profile)

				if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
					print('File exists already. Skipping process.')
					meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz'))
				else:
					tmp = profile.get_fdata()[:,:,:,1:19]
					extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
					nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_extractedLayers-allLayers.nii.gz'))
					meanProfile = mean_img(extractedLayers)
					nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz'))
					del tmp

				nighres.surface.surface_mesh_mapping(
					intensity_image=meanProfile,
					surface_mesh=corticalSurface['result'],
					inflated_mesh=inflatedSurface['result'],
					mapping_method='closest_point',
					save_data=True,
					overwrite=reprocess,
					output_dir=out_dir,
					file_name='sub-' + sub + '_' + transform_data_output[i] + '_' + hemi + 'Hemisphere_extractedLayers-allLayers.vtk')
		else:
			profile = nighres.laminar.profile_sampling(
			    profile_surface_image=layers,
			    intensity_image=transform_data,
			    save_data=True,
			    overwrite=reprocess,
			    output_dir=out_dir,
			    file_name='sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_extractedLayers')['result']

			profile = nb.load(profile)

			print('')
			print('*****************************************************')
			print('* Extracting all cortical layers of additional data (' + hemi + ' hemisphere)')
			print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
			print('*****************************************************')

			if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
				print('File exists already. Skipping process.')
				meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz'))
			else:
				tmp = profile.get_fdata()[:,:,:,1:19]
				extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
				nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers.nii.gz'))
				meanProfile = mean_img(extractedLayers)
				nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers_mean.nii.gz'))
				del tmp

			nighres.surface.surface_mesh_mapping(
			    intensity_image=meanProfile,
			    surface_mesh=corticalSurface['result'],
			    inflated_mesh=inflatedSurface['result'],
			    mapping_method='closest_point',
			    save_data=True,
			    overwrite=reprocess,
			    output_dir=out_dir,
			    file_name='sub-' + sub + '_' + transform_data_output + '_' + hemi + 'Hemisphere_extractedLayers-allLayers.vtk')

	# Set reprocess flag to false in case of re-processing a hemisphere only.
	if locals()['reprocess_' + hemi + 'Hemisphere']:
		reprocess = False

###########################################################################
# 16. Clean up
# ---------------------
if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_leftHemisphere_extractedLayers-superficialLayers_map-inf.vtk')) and os.path.isfile(os.path.join(out_dir, 'sub-' + sub + merged + '_rightHemisphere_extractedLayers-superficialLayers_map-inf.vtk')):
	os.remove(os.path.join(out_dir, 'is_running'))
	#open(os.path.join(out_dir, 'processed_successfully'), mode='a').close()
	print('')
	print('*****************************************************')
	print('* Processing of subject ' + sub + ' finished successfully at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
	print('*****************************************************')
else:   
	print('')
	print('*****************************************************')
	print(' Processing not finished! Please check log file for errors.')
	print('*****************************************************')
	
	
############################################################################
# 7. Combination of native and upsampled data (USING PYTHON)
#    Works but results differ from MATLAB version and yields worse segmentation
# ----------------
# Here, we combine the slab and whole brain data by a weighted averaged.
# The slab does not cover the entire temporal lobe due to very low SNR.
# Therefore, weighted averaging is applied in z-direction (superior to
# inferior) in the last third number of slices gradually increasing 
# the weighting with distance.
#if hires == True:
#	print('')
#	print('*****************************************************')
#	print('* Combination of native and upsampled data')
#	print('* Started at: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
#	print('*****************************************************')
#
#	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected.nii.gz')) and reprocess != True:
#		print('File exists already. Skipping process.')
#
#		# Update file names
#		T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected_masked.nii.gz')
#		T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_masked.nii.gz')
#		T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected_masked.nii.gz')
#	else:
#		# Load data
#		T1map_biasCorrected = ants.image_read(T1map_biasCorrected)
#		T1map_slab_biasCorrected_reg = ants.image_read(T1map_slab_biasCorrected_reg)
#
#		T1map = ants.image_read(T1map)
#		T1map_slab_reg = ants.image_read(T1map_slab_reg)
#
#		T1w_biasCorrected = ants.image_read(T1w_biasCorrected)
#		T1w_slab_biasCorrected_reg = ants.image_read(T1w_slab_biasCorrected_reg)
#
#		mask = ants.image_read(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_brainmask.nii.gz'))
#		mask_slab_reg = ants.image_read(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_brainmask_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz'))
#
#		# Create volume filled with zeros
#		clone = ants.image_clone(T1w_biasCorrected)
#		weighted = ants.image_clone(T1w_biasCorrected)
#		weighted[:,:,:] = 0
#
#		# Get white matter segmentation from whole brain image
#		print('Get white matter segmentation from T1w whole brain image')
#		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_WM_segmentation.nii.gz')): 
#			print('File exists already. Skipping process.')
#			WM_segmentation = ants.image_read(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_WM_segmentation.nii.gz'))
#		else:
#			WM_segmentation = ants.atropos(a=T1w_biasCorrected, x=mask, m='[0.2,1x1x1]', c='[3,0]', i='kmeans[3]', v=1)['segmentation']
#			WM_segmentation = ants.threshold_image(WM_segmentation, low_thresh=3, high_thresh=3, inval=1, outval=0, binary=False)
#			WM_segmentation = ants.iMath(WM_segmentation, 'GetLargestComponent')
#			WM_segmentation = ants.morphology(WM_segmentation, operation='dilate', radius=1, mtype='grayscale', shape='box')
#			ants.image_write(WM_segmentation, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w' + resampled + '_WM_segmentation.nii.gz'))
#
#		# Get white matter segmentation from slab image
#		print('')
#		print('Get white matter segmentation from T1w slab image')
#		if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_WM_segmentation.nii.gz')): 
#			print('File exists already. Skipping process.')
#			WM_segmentation_slab = ants.image_read(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_WM_segmentation.nii.gz'))
#		else:
#			WM_segmentation_slab = ants.atropos(a=T1w_slab_biasCorrected_reg, x=mask_slab_reg, m='[0.2,1x1x1]', c='[3,0]', i='kmeans[3]', v=1)['segmentation']
#			WM_segmentation_slab = ants.threshold_image(WM_segmentation_slab, low_thresh=3, high_thresh=3, inval=1, outval=0, binary=False)
#			WM_segmentation_slab = ants.iMath(WM_segmentation_slab, 'GetLargestComponent')
#			WM_segmentation_slab = ants.morphology(WM_segmentation_slab, operation='dilate', radius=1, mtype='grayscale', shape='box')
#			ants.image_write(WM_segmentation, os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_WM_segmentation.nii.gz'))
#
#		# Calulate ratio between T1w whole brain white matter segmentation and high resolution slab data
#		ratio_T1w = WM_segmentation.mean() / WM_segmentation_slab.mean()
#		
#		# Mask T1map to calulate ratio between quantitateve T1 whole brain white matter segmentation and high resolution slab data
#		WM_segmentation = ants.mask_image(T1map_biasCorrected,WM_segmentation,binarize=False)
#		WM_segmentation_slab = ants.mask_image(T1map_slab_biasCorrected_reg,WM_segmentation_slab,binarize=False)
#		
#		# Calulate ratio between T1map whole brain white matter segmentation and high resolution slab data
#		ratio_T1map = WM_segmentation.mean() / WM_segmentation_slab.mean()
#
#		# Get dimensions of image for looping
#		size = T1w_biasCorrected.shape
#
#		# Convert to numpy
#		T1map_biasCorrected_image = T1map_biasCorrected.numpy()
#		T1map_slab_biasCorrected_reg_image = T1map_slab_biasCorrected_reg.numpy()
#
#		T1map_image = T1map.numpy()
#		T1map_slab_reg_image = T1map_slab_reg.numpy()
#
#		T1w_biasCorrected_image = T1w_biasCorrected.numpy()
#		T1w_slab_biasCorrected_reg_image = T1w_slab_biasCorrected_reg.numpy()
#
#		T1map_biasCorrected_weighted_image = weighted.numpy()
#		T1map_weighted_image = weighted.numpy()
#		T1w_biasCorrected_weighted_image = weighted.numpy()
#		
#		# Weight whole brain image by ratio to harmonize intensities between whole brain and slab data
#		T1map_image = T1map_image / ratio_T1map
#		T1map_biasCorrected_image = T1map_biasCorrected_image / ratio_T1map
#		T1w_biasCorrected_image = T1w_biasCorrected_image / ratio_T1w
#
#		print('')
#		print('Combining data...')
#		for x in range(size[0]-1):
#			for y in range(size[1]-1):
#				for z in range(size[2]-1,-1,-1):
#					# If high resolution slab (and new image) is 0, then use low resolution whole brain data
#					if T1map_biasCorrected_image[x,y,z] != 0 and T1map_slab_biasCorrected_reg_image[x,y,z] == 0 and T1map_weighted_image[x,y,z] == 0:
#						T1map_weighted_image[x,y,z] = T1map_image[x,y,z]
#						T1map_biasCorrected_weighted_image[x,y,z] = T1map_biasCorrected_image[x,y,z]
#						T1w_biasCorrected_weighted_image[x,y,z] = T1w_biasCorrected_image[x,y,z]
#					# If both, the low resolution whole brain and high resolution slab data are not zero ...
#					elif T1map_biasCorrected_image[x,y,z] != 0 and T1map_slab_biasCorrected_reg_image[x,y,z] != 0 and T1map_weighted_image[x,y,z] == 0:
#						# ... use high resolution slab data for the superior two third of the brain
#						if z >= round(size[2]*2/3):
#							T1map_weighted_image[x,y,z] = T1map_slab_reg_image[x,y,z]
#							T1map_biasCorrected_weighted_image[x,y,z] = T1map_slab_biasCorrected_reg_image[x,y,z]
#							T1w_biasCorrected_weighted_image[x,y,z] = T1w_slab_biasCorrected_reg_image[x,y,z]
#						# ... for the inferior one third of the brain calculate a weighted average of the low resolution whole brain and high resolution slab data depending on the distance.
#						else:
#							counter = 0
#							# Get distance here by iterating through the data until slab data is zero
#							while T1map_slab_biasCorrected_reg_image[x,y,z-counter] != 0 and T1map_weighted_image[x,y,z] == 0:
#								counter += 1
#								if z-counter < 1:
#									counter -= 1
#									break
#
#							# Weight low resolution as well as high resolution slab data accordingly by the distance
#							for weight_in_z in range(counter):
#								T1map_biasCorrected_weighted_image[x,y,z-weight_in_z]=T1map_biasCorrected_image[x,y,z-weight_in_z]*weight_in_z/counter+T1map_slab_biasCorrected_reg_image[x,y,z-weight_in_z]*(1-weight_in_z/counter)
#								
#								T1map_weighted_image[x,y,z-weight_in_z]=T1map_image[x,y,z-weight_in_z]*weight_in_z/counter+T1map_slab_reg_image[x,y,z-weight_in_z]*(1-weight_in_z/counter)
#								
#								T1w_biasCorrected_weighted_image[x,y,z-weight_in_z]=T1w_biasCorrected_image[x,y,z-weight_in_z]*weight_in_z/counter+T1w_slab_biasCorrected_reg_image[x,y,z-weight_in_z]*(1-weight_in_z/counter)
#					
#					if T1map_biasCorrected_weighted_image[x,y,z] < 0:
#						T1map_biasCorrected_weighted_image[x,y,z] = 0
#
#		print('Done')
#		
#		# Copy numpy array into image
#		T1map_biasCorrected = weighted.new_image_like(T1map_biasCorrected_weighted_image)
#		T1map = weighted.new_image_like(T1map_weighted_image)
#		T1w_biasCorrected = weighted.new_image_like(T1w_biasCorrected_weighted_image)
#
#		# Mask image
#		T1map_biasCorrected_masked = ants.mask_image(T1map_biasCorrected,mask)
#		T1map_masked = ants.mask_image(T1map,mask)
#		T1w_biasCorrected_masked = ants.mask_image(T1w_biasCorrected,mask)
#
#		# Write to disk
#		ants.image_write(T1map_biasCorrected, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected.nii.gz'))
#		ants.image_write(T1map_biasCorrected_masked, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected_masked.nii.gz'))
#
#		ants.image_write(T1map, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map.nii.gz'))
#		ants.image_write(T1map_masked, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_masked.nii.gz'))
#
#		ants.image_write(T1w_biasCorrected, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected.nii.gz'))
#		ants.image_write(T1w_biasCorrected_masked, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected_masked.nii.gz'))
#
#		# Update file names
#		T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected_masked.nii.gz')
#		T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_masked.nii.gz')
#		T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected_masked.nii.gz')
