'''
Mapping of quantitative T1 values from MP2RAGE data onto surface
=======================================

This is a pipeline processing MP2RAGE data by performing the following steps:

01. Setup
02. MP2RAGE background cleaning
03. Resampling to 500 µm (only for hires option)
04. Imhomogeneity correction and skull stripping
05. Registration of whole brain to slab data (only for hires option)
06. Weighted images combination of whole brain and slab data (only for hires option)
07. Atlas-guided tissue classification using MGDM
08. Region extraction (left hemisphere) 
09. Crop volume (left hemisphere)
10. CRUISE cortical reconstruction (left hemisphere)
11. Extract layers across cortical sheet and map on surface (left hemisphere)
12. Region extraction (right hemisphere)
13. Crop volume (right hemisphere)
14. CRUISE cortical reconstruction (right hemisphere)
15. Extract layers across cortical sheet and map on surface (right hemisphere)

Version 0.9 (17.06.2022)
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
# THINGS TO DO
# -------------------
# 1. Specify files, paths, and flag from outside this script
# 2. Get rid of MATLAB:
# 	Switch from SPM's bias correction method to N4 of ANTs for easier use
# 	Re-write the weightedAverage script for Python
#	MP2RAGE background cleaning
# 3. Apply transformation to multiple volumes instead of a single file
# 4. Differentiate between data type to be mapped on the surface automatically, e.g. QSM, fMRI, other types.
#

############################################################################
# Installation instructions
# -------------------
# Things needed to be installed:
# 1. Working MATLAB installation
# 2. Working SPM installation
# 3. MATLAB scripts (weightedAverage and biasCorrection) added to Path
#
# You need to change the path to the tissue probability model for the bias
# field correction method. This needs to be done in
# 
# ./biasCorrection/preproc_sensemap.m on line 19
#

############################################################################
# Usage instructions
# -------------------
# This script requires data at two different resolutions. A whole brain 
# MP2RAGE at an isotropic resolution of 700 µm and a high resolution
# MP2RAGE slab with an isotropic resolution of 500 µm. The data needs to
# be organized according to BIDS.
#
# In the section set parameters of the setup section below, you need to
# specify the folder to the BIDS directory and the label of the subject
# you want to process.
#
# A few flags have been created which allow to reprocess data at various
# stages of the pipeline (e.g. entirely, from the segmentation onwards,
# mapping of additional data, or per hemisphere).
#
# Furthermore, you can specify a single volume which will be mapped on
# the surface additionally. This is meant for mapping another contrast 
# on the surface, e.g. QSM data or fMRI results. Soon, it will be
# possible to specify a directory for mapping multiple volume onto the
# surface.
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
from nilearn.image import mean_img
from nilearn.image import crop_img
from nibabel.processing import conform

############################################################################
# 1.2. Set parameters
# -------------------
# Define BIDS path
BIDS_path = '/tmp/luesebrink/'

# Define path from where data is to be copied into "BIDS_path". Could either
# be to create a backup of the data before processing or transfer to a
# compute server.
# If the path does not exist, this option will be omitted and the data from
# "BIDS_path" will be used instead.
copy_data_from = 'gerd:/media/luesebrink/bmmr_data/data/sensemap/all/young/'

# Define subject following BIDS
sub = 'wtl'

# Process with high resolution MP2RAGE slab?
hires = True

# Map specific file onto the surface. Requries full path to a NIfTI file.
# If the path points to a non-existing file, the according option will
# be omitted.
# Results will be written to <BIDS_path>/derivatives/sub-<label>/.
map_data = '/tmp/luesebrink/sub-wtl/anat/sub-wtl_run-01_T1map.nii.gz'
# map_data_output = ''

# Transform data in the same space as 'map_data' which is then mapped onto
# the surface. Could for example be resulting maps of fMRI data. Requries
# full path to a NIfTI file.
# If the path points to a non-existing file, the according option will
# be omitted. In the future, this option will accept a directory
# containing NIfTI files. The transformation is then applied to all 
# files within the directory.
# Results will be written to <BIDS_path>/derivatives/sub-<label>/.
transform_data = '/tmp/luesebrink/sub-wtl/anat/sub-wtl_run-01_T1map.nii.gz'
# transform_data_output = ''

#
# PROBABLY I SHOULD ADD A FLAG TO DIFFERENTIATE BETWEEN DATA TYPES TO BE
# MAPPED ON THE SURFACE. LIKE QSM, FMRI, AND OTHER TYPE
#
# CAN BE DONE AUTOMATICALLY BASED ON THE VOLUME GIVEN IN BIDS FORMAT
# 	-> sub-<label>_..._Chimap.nii.gz for QSM
#	-> sub-<label>_..._bold.nii.gz for BOLD data
#


# Flag to overwrite all existing data
reprocess = False

# Flag to start reprocessing from segmentation onwards
reprocess_segmentation = False

# Flag to reprocess left or right hemisphere only
reprocess_leftHemisphere = False
reprocess_rightHemisphere = False

# Flag to reprocess additional data
reprocess_map_data = False

# Define path to atlas (probably not needed in newer versions as it should be the default one)
atlas = '/data/hu_luesebrink/venv/nighres/lib/python3.6/site-packages/nighres/atlases/brain-segmentation-prior3.0/brain-atlas-quant-3.0.9.txt'

############################################################################
# 1.3. Copy data to compute server and define file names
# -------------------
# Set paths
in_dir = BIDS_path + 'sub-' + sub + '/anat/'
out_dir = BIDS_path + 'derivatives/sub-' + sub + '/'
os.system('mkdir -p ' + in_dir)
os.system('mkdir -p ' + out_dir)

# Define file names
INV1 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-1_MP2RAGE.nii.gz')
INV2 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-2_MP2RAGE.nii.gz')
T1map = os.path.join(in_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')
UNI = os.path.join(in_dir, 'sub-' + sub + '_run-01_UNIT1.nii.gz')

INV1_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_inv-1_MP2RAGE.nii.gz')
INV2_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_inv-2_MP2RAGE.nii.gz')
T1map_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_T1map.nii.gz')
UNI_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_UNIT1.nii.gz')

# Copy data
if os.path.isdir(copy_data_from):
	print('')
	print('*****************************************************')
	print('* Data transfer to working directory.')
	print('*****************************************************')
	if os.path.isfile(os.path.join(in_dir, 'sub-' + sub + '_run-01_UNIT1.nii.gz'))  and reprocess != True:
		print('Files exists already. Skipping data transfer.')
	else:
		os.system('scp -r ' + copy_data_from + sub + '/* ' + BIDS_path + 'sub-' + sub + '/anat/')

# Check if data exists.
print('')
print('*****************************************************')
print('* Checking if files exists.')
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
print('*****************************************************')
if os.path.isfile(map_data):
	map_file_onto_surface = True
	print('File found for mapping of additional data onto surface!')
else:
	map_file_onto_surface = False
	print('Could not find file for surface mapping. Omitting flag!')

# Check if file for applying the transformation of the additional data exists.
if os.path.isfile(transform_data):
	map_transform_file_onto_surface = True
	print('File found for applying transform of additional data!')
else:
	map_transform_file_onto_surface = False
	print('Could not find file for applying transform of additional data. Omitting flag!')

# Set flag if high resolution slab is used.
if hires == True:
	filename = '_merged_run-01+02'
else:
	filename = '_run-01'

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
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	reg = str(10)
	output = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')
	os.system('matlab -nosplash -nodisplay -r \"removeBackgroundnoise(\'' + UNI + '\', \'' + INV1 + '\', \'' + INV2 + '\', \'' + output + '\', ' + reg + '); exit;\"')

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
# Should be changes to the resolution of the high resolution data at some
# point and not expect 500 µm data.
if hires == True:
	print('')
	print('*****************************************************')
	print('* Resampling MP2RAGE to an isotropic resolution of 500 µm')
	print('*****************************************************')

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
	os.system('cp ' + T1map + ' ' + out_dir + 'sub-' + sub + '_run-01_T1map.nii.gz')
	T1map = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')

############################################################################
# 4. Inhomogeneity correction and skull stripping
# ----------------
# Here, we perform inhomogeneity correction and skull stripping using SPM's
# segment routine. Based on the WM, GM, and CSF segmentation a brainmask is
# created and applied to the inhomogeneity corrected volumes.
print('')
print('*****************************************************')
print('* Inhomogeneity correction and skull stripping of whole brain data')
print('*****************************************************')

if hires == True:
	checkFile = 'sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz'
else:
	checkFile = 'sub-' + sub + '_run-01_T1map_biasCorrected_masked.nii.gz'

if os.path.isfile(os.path.join(out_dir, checkFile)) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	os.system('matlab -nosplash -nodisplay -r \"preproc_sensemap(\'' + T1w + '\'); exit;\"')
	os.system('matlab -nosplash -nodisplay -r \"preproc_sensemap(\'' + T1map + '\'); exit;\"')

if hires == True:
	print('')
	print('*****************************************************')
	print('* Inhomogeneity correction and skull stripping of slab data')
	print('*****************************************************')
	# Copy data and update file name
	os.system('cp ' + T1map_slab + ' ' + out_dir + 'sub-' + sub + '_run-02_T1map.nii.gz')
	T1map_slab = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map.nii.gz')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		os.system('matlab -nosplash -nodisplay -r \"preproc_sensemap(\'' + T1w_slab + '\'); exit;\"')
		os.system('matlab -nosplash -nodisplay -r \"preproc_sensemap(\'' + T1map_slab + '\'); exit;\"')

	# Update file names
	mask = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled_brainmask.nii.gz')
	T1w_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_resampled_biasCorrected.nii.gz')
	T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_resampled_biasCorrected_masked.nii.gz')
	T1map_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled_biasCorrected.nii.gz')
	T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')

	T1map_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz')
	T1w_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected.nii.gz')
	T1map_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_masked.nii.gz')
else:
	# Mask uncorrected T1map if it does not exist already
	print('')
	print('*****************************************************')
	print('* Masking T1map')
	print('*****************************************************')
	# Update file names	
	T1w_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_biasCorrected.nii.gz')
	T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_biasCorrected_masked.nii.gz')
	T1map_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_biasCorrected.nii.gz')
	T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_biasCorrected_masked.nii.gz')
	mask = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_brainmask.nii.gz')
	
	# Check if volume exists
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_masked.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
	# Mask image otherwise
		maskedImage = ants.mask_image(ants.image_read(T1map), ants.image_read(mask))
		ants.image_write(maskedImage, os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_masked.nii.gz'))
		print('Done.')

	# Update file names
	T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_masked.nii.gz')

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
				reg_iterations = (200, 100, 20 ,10),
				verbose = True,
				#outprefix = out_dir + 'sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked_registered_to_sub-' + sub + '_run-02_T1map_biasCorrected_masked_',
				)
		print('')
		print('*****************************************************')
		print('* Apply transformations')
		print('*****************************************************')
		# Apply transformation to several files.
		warpedImage = ants.apply_transforms(
				fixed = ants.image_read(T1map_biasCorrected),
				moving = ants.image_read(T1map_slab),
		                transformlist = registeredImage['invtransforms'],
				interpolator = 'bSpline',
				verbose = True,
				)

		ants.image_write(warpedImage, out_dir + 'sub-' + sub + '_run-02_T1map_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')

		warpedImage = ants.apply_transforms(
				fixed = ants.image_read(T1map_biasCorrected),
				moving = ants.image_read(T1map_slab_biasCorrected),
		                transformlist = registeredImage['invtransforms'],
				interpolator = 'bSpline',
				verbose = True,
				)

		ants.image_write(warpedImage, out_dir + 'sub-' + sub + '_run-02_T1map_biasCorrected_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz') 

		warpedImage = ants.apply_transforms(
				fixed = ants.image_read(T1w_biasCorrected),
				moving = ants.image_read(T1w_slab_biasCorrected),
		                transformlist = registeredImage['invtransforms'],
				interpolator = 'bSpline',
				verbose = True,
				)

		ants.image_write(warpedImage, out_dir + 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_' + sub + '_run-01_T1w_resampled_biasCorrected_masked.nii.gz') 
		
	# Update file names
	T1w_slab_biasCorrected_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_' + sub + '_run-01_T1w_resampled_biasCorrected_masked.nii.gz')
	T1map_slab_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')
	T1map_slab_biasCorrected_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')

############################################################################
# 5.1. Register additional data to (upsampled) T1map
# -------------------
if reprocess_map_data:
	reprocess = True

if hires == True:
	print('')
	print('*****************************************************')
	print('* Register additional data to upsampled T1map')
	print('*****************************************************')
	reg1 = 'sub-' + sub + '_map_data_registered_to_sub-' + sub + '_run-01_T1map_resampled_biasCorrected'
else:
	print('')
	print('*****************************************************')
	print('* Register additional data to T1map')
	print('*****************************************************')
	reg1 = 'sub-' + sub + '_map_data_registered_to_sub-' + sub + '_run-01_T1map_biasCorrected'

if map_file_onto_surface:
	if os.path.isfile(os.path.join(out_dir, reg1 + '.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		# Register additional data non-linearly to T1map using mutual information as similarity metric
		registeredImage = ants.registration(
				fixed = ants.image_read(T1map_biasCorrected),
				moving = ants.image_read(map_data),
				type_of_transform = 'SyNRA',
				reg_iterations = (200, 100, 20 ,10),
				verbose = True,
				outprefix = out_dir + reg1 + '_',
				)

		# Apply transformation
		warpedImage = ants.apply_transforms(
				fixed = ants.image_read(T1map_biasCorrected),
				moving = ants.image_read(map_data),
		                transformlist = registeredImage['fwdtransforms'],
				interpolator = 'bSpline',
				verbose = True,
				)

		# Write file to disk
		ants.image_write(warpedImage, out_dir + reg1 + '.nii.gz') 

# Update file name
map_data = os.path.join(out_dir, reg1 + '.nii.gz')

############################################################################
# 5.2. Apply transformation to data to be mapped on the surface
# -------------------
if hires == True:
	reg2 = 'sub-' + sub + '_transform_data_registered_to_' + sub + '_run-01_T1map_resampled_biasCorrected.nii.gz'
else:
	reg2 = 'sub-' + sub + '_transform_data_registered_to_' + sub + '_run-01_T1map_biasCorrected.nii.gz'

if map_file_onto_surface and map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Apply transformation of registration to additional data')
	print('*****************************************************')
	if os.path.isfile(os.path.join(out_dir, reg2)) and reprocess != True:
		print('File exists already. Skipping process.')
	else:
		# Apply transformation
		warpedImage = ants.apply_transforms(
				fixed = ants.image_read(T1map_biasCorrected),
				moving = ants.image_read(transform_data),
		                transformlist = registeredImage['fwdtransforms'],
				interpolator = 'bSpline',
				verbose = True,
				)

		# Write file to disk
		ants.image_write(warpedImage, out_dir + reg2) 
		
# Update file name
transform_data = os.path.join(out_dir, reg2)

if reprocess_map_data:
	reprocess = False

############################################################################
# 6. Combination of native and upsampled data
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

#############################################################################
# 7. MGDM classification
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
                save_data=True,
                overwrite=reprocess,
                output_dir=out_dir,
                file_name='sub-' + sub + filename)

###########################################################################
# 8. Region Extraction (left hemisphere)
# ------------------
# Here, we pull from the MGDM output the needed regions for cortical
# reconstruction: the GM cortex ('region'), the underlying WM (with filled
# subcortex and ventricles, 'inside') and the surrounding CSF (with masked
# regions, 'background')
print('')
print('*****************************************************')
print('* Region extraction of left hemisphere')
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
        file_name='sub-' + sub + filename + '_leftHemisphere',
        output_dir=out_dir)

#############################################################################
# 9. Crop volume to left hemisphere
# --------------------------------
# Here, we crop the volume to the left hemisphere based on the output of
# MGDM. This will reduce the memory demand tremendously as well as
# processing time without sacrificing any accuracy.
print('')
print('*****************************************************')
print('* Crop volume to left hemisphere')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_T1map_leftHemisphere_cropped.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	inside_mask = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xmask-lcrwm_cropped.nii.gz'))
	inside_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xproba-lcrwm_cropped.nii.gz'))
	region_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xproba-lcrgm_cropped.nii.gz'))
	background_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xproba-lcrbg_cropped.nii.gz'))
	T1map_masked_leftHemisphere = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_T1map_leftHemisphere_cropped.nii.gz'))
else:
	# Load grey matter image, binarize image, and get information for cropping
	img = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xmask-lcrgm.nii.gz'))
	tmp = img.get_fdata()
	tmp[tmp<0] = 0
	tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
	crop,coord = crop_img(tmp, pad=4, return_offset=True)

	# Apply cropping
	tmp1 = nb.load(cortex['inside_mask'])
	tmp2 = tmp1.get_fdata()
	inside_mask = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_mask, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xmask-lcrwm_cropped.nii.gz'))

	# Apply cropping
	tmp1 = nb.load(cortex['inside_proba'])
	tmp2 = tmp1.get_fdata()
	inside_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_proba, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xproba-lcrwm_cropped.nii.gz'))

	# Apply cropping
	tmp1 = nb.load(cortex['region_proba'])
	tmp2 = tmp1.get_fdata()
	region_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(region_proba, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xproba-lcrgm_cropped.nii.gz'))

	# Apply cropping
	tmp1 = nb.load(cortex['background_proba'])
	tmp2 = tmp1.get_fdata()
	background_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(background_proba, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xproba-lcrbg_cropped.nii.gz'))

	tmp1 = nb.load(T1map_masked)
	tmp2 = tmp1.get_fdata()
	T1map_masked_leftHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(T1map_masked_leftHemisphere, os.path.join(out_dir, 'sub-' + sub + filename + '_T1map_leftHemisphere_cropped.nii.gz'))

	del img
	del tmp
	del tmp1
	del tmp2
	print('Done.')

############################################################################
# 9.1. Crop additional data to left hemispehre
# -------------------
if reprocess_map_data:
	reprocess = True

if map_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Crop additional data to left hemisphere')
	print('*****************************************************')
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_map_data_leftHemisphere_cropped.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		map_data_leftHemisphere = os.path.join(out_dir, 'sub-' + sub + '_map_data_leftHemisphere_cropped.nii.gz')
	else:
		# Load grey matter image, binarize image, and get information for cropping
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xmask-lcrgm.nii.gz'))
		tmp = img.get_fdata()
		tmp[tmp<0] = 0
		tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
		crop,coord = crop_img(tmp, pad=4, return_offset=True)

		# Apply cropping
		tmp1 = nb.load(map_data)
		tmp2 = tmp1.get_fdata()
		map_data_leftHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
		nb.save(map_data_leftHemisphere, os.path.join(out_dir, 'sub-' + sub + '_map_data_leftHemisphere_cropped.nii.gz'))

		del img
		del tmp
		del tmp1
		del tmp2
		print('Done.')

############################################################################
# 9.2. Crop additionally transformed data to left hemispehre
# -------------------
if map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Crop transformed data to left hemisphere')
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_transform_data_leftHemisphere_cropped.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		transform_data_leftHemisphere = os.path.join(out_dir, 'sub-' + sub + '_transform_data_leftHemisphere_cropped.nii.gz')
	else:
		# Load grey matter image, binarize image, and get information for cropping
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_xmask-lcrgm.nii.gz'))
		tmp = img.get_fdata()
		tmp[tmp<0] = 0
		tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
		crop,coord = crop_img(tmp, pad=4, return_offset=True)

		# Apply cropping
		tmp1 = nb.load(transform_data)
		tmp2 = tmp1.get_fdata()
		transform_data_leftHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
		nb.save(transform_data_leftHemisphere, os.path.join(out_dir, 'sub-' + sub + '_transform_data_leftHemisphere_cropped.nii.gz'))

		del img
		del tmp
		del tmp1
		del tmp2
		print('Done.')

if reprocess_map_data:
	reprocess = False

#############################################################################
# 10. CRUISE cortical reconstruction (left hemisphere)
# --------------------------------
# the WM inside mask as a (topologically spherical) starting point to grow a
# refined GM/WM boundary and CSF/GM boundary
print('')
print('*****************************************************')
print('* Cortical reconstruction with CRUISE')
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
                        file_name='sub-' + sub + filename + '_leftHemisphere',
                        output_dir=out_dir)

###########################################################################
# 11. Extract layers across cortical sheet and map on surface
###########################################################################
# 11.1 Volumetric layering for depth measures (left hemisphere)
# ---------------------
# Finally, we use the GM/WM boundary (GWB) and CSF/GM boundary (CGB) from
# CRUISE to compute cortical depth with a volume-preserving technique.
print('')
print('*****************************************************')
print('* Extract layers across the cortical sheet (left hemisphere)')
print('*****************************************************')

layers = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise['gwb'],
                        outer_levelset=cruise['cgb'],
                        n_layers=20,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers-allLayers',
                        output_dir=out_dir)['boundaries']
layers = nb.load(layers)

###########################################################################
# 11.2. Extract middle layer for surface generation and mapping (left hemisphere)
# ---------------------
# Here, we extract the levelset of the middle layer which will be used to
# map the information on.
print('')
print('*****************************************************')
print('* Extracting middle cortical layer (left hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-middleLayer.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	extractedLayers=nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-middleLayer.nii.gz'))
else:
	tmp = layers.get_fdata()[:,:,:,10:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-middleLayer.nii.gz'))
	del tmp

###########################################################################
# 11.3 Cortical surface generation and inflation of middle layer (left hemisphere)
# ---------------------
# Here, we create a surface from the levelset of the middle layer and
# inflate it afterwards for better visualisation.
print('')
print('*****************************************************')
print('* Generate surface of middle cortical layer and inflate it (left hemisphere)')
print('*****************************************************')

corticalSurface = nighres.surface.levelset_to_mesh(
                        levelset_image=extractedLayers,
			connectivity='26/6',
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

inflatedSurface = nighres.surface.surface_inflation(
                        surface_mesh=corticalSurface['result'],
			max_iter=10000,
			max_curv=8.0,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

###########################################################################
# 11.4. Profile sampling of all layers (left hemisphere)
# ---------------------
# Here, we sample the T1 values of the T1map onto all layers.
print('')
print('*****************************************************')
print('* Profile sampling across cortical sheet (left hemisphere)')
print('*****************************************************')

profile = nighres.laminar.profile_sampling(
                        profile_surface_image=layers,
                        intensity_image=T1map_masked_leftHemisphere,
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers')['result']
profile = nb.load(profile)

###########################################################################
# 11.5. Extract all cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 3 to 18, removing layers close to the white
# matter and cerebrospinal fluid to avoid partial volume effects.
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting all cortical layers (left hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
else:
	tmp = profile.get_fdata()[:,:,:,3:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-allLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers-allLayers.vtk')


###########################################################################
# 11.6. Extract deep cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 3 to 6, to cover the "deep" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting deep cortical layers (left hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-deepLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:7]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-deepLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-deepLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers-deepLayers.vtk')

###########################################################################
# 11.7. Extract inner middle cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 7 to 10, to cover the "inner middle" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting inner middle cortical layers (left hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-innerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,7:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-innerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-innerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers-innerLayers.vtk')

###########################################################################
# 11.8. Extract outer middle cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 11 to 14, to cover the "outer middle" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting outer middle cortical layers (left hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-outerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,11:15]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-outerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-outerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers-outerLayers.vtk')

###########################################################################
# 11.9. Extract superficial cortical layers and map on surface (left hemisphere)
# ---------------------
# Here, we extract the layers 15 to 18, to cover the "superficial" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting superficial cortical layers (left hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-superficialLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,15:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-superficialLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_leftHemisphere_extractedLayers-superficialLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_leftHemisphere_extractedLayers-superficialLayers.vtk')

###########################################################################
# 11.10. Profile sampling of additional data of all layers and mapping it onto
# the surface (left hemisphere)
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
	print('* Profile sampling of additional data (left hemisphere)')
	print('*****************************************************')
	profile = nighres.laminar.profile_sampling(
		                profile_surface_image=layers,
		                intensity_image=map_data_leftHemisphere,
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_map_data_leftHemisphere_extractedLayers')['result']
	profile = nb.load(profile)

	print('')
	print('*****************************************************')
	print('* Extracting all cortical layers of additional data (left hemisphere)')
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_map_data_leftHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_map_data_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	else:
		tmp = profile.get_fdata()[:,:,:,1:19]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_map_data_leftHemisphere_extractedLayers-allLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_map_data_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		                intensity_image=meanProfile,
		                surface_mesh=corticalSurface['result'],
		                inflated_mesh=inflatedSurface['result'],
		                mapping_method='closest_point',
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_map_data_leftHemisphere_extractedLayers-allLayers.vtk')

###########################################################################
# 11.11. Profile sampling of transformed data of all layers and mapping it onto
# the surface (left hemisphere)
# ---------------------
# Here, we sample the T1 values of the transformed data onto all layers.
# And we extract the layers 1 to 18, removing layers close to the
# cerebrospinal fluid to avoid partial volume effects. Afterwards the
# information is mapped on the original and inflated surface of the middle
# layer.
if map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Profile sampling of transformed data (left hemisphere)')
	print('*****************************************************')
	profile = nighres.laminar.profile_sampling(
		                profile_surface_image=layers,
		                intensity_image=transform_data_leftHemisphere,
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_transform_data_leftHemisphere_extractedLayers')['result']
	profile = nb.load(profile)

	print('')
	print('*****************************************************')
	print('* Extracting all cortical layers of additional data (left hemisphere)')
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_transform_data_leftHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_transform_data_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	else:
		tmp = profile.get_fdata()[:,:,:,1:19]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_transform_data_leftHemisphere_extractedLayers-allLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_transform_data_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		                intensity_image=meanProfile,
		                surface_mesh=corticalSurface['result'],
		                inflated_mesh=inflatedSurface['result'],
		                mapping_method='closest_point',
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_transform_data_leftHemisphere_extractedLayers-allLayers.vtk')

# Set reprocess flag to false in case left hemisphere or mapping of
# additional data should be done only.
if reprocess_leftHemisphere or reprocess_map_data:
	reprocess = False

###########################################################################
# 12. Region Extraction (right hemisphere)
# ------------------
# Here, we pull from the MGDM output the needed regions for cortical
# reconstruction: the GM cortex ('region'), the underlying WM (with filled
# subcortex and ventricles, 'inside') and the surrounding CSF (with masked
# regions, 'background')
print('')
print('*****************************************************')
print('* Region extraction of right hemisphere')
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
                                            file_name='sub-' + sub + filename + '_rightHemisphere',
                                            output_dir=out_dir)

#############################################################################
# 13. Crop volume to right hemisphere
# --------------------------------
# Here, we crop the volume to the right hemisphere based on the output of
# MGDM. This will reduce the memory demand tremendously as well as
# processing time without sacrificing any accuracy.
print('')
print('*****************************************************')
print('* Crop volume to right hemisphere')
print('*****************************************************')
if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_T1map_rightHemisphere_cropped.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	inside_mask = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xmask-rcrwm_cropped.nii.gz'))
	inside_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xproba-rcrwm_cropped.nii.gz'))
	region_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xproba-rcrgm_cropped.nii.gz'))
	background_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xproba-rcrbg_cropped.nii.gz'))
	T1map_masked_rightHemisphere = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_T1map_rightHemisphere_cropped.nii.gz'))
else:
	img = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xmask-rcrgm.nii.gz'))
	tmp = img.get_fdata()
	tmp[tmp<0] = 0
	tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
	crop,coord = crop_img(tmp, pad=4, return_offset=True)

	tmp1 = nb.load(cortex['inside_mask'])
	tmp2 = tmp1.get_fdata()
	inside_mask = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_mask, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xmask-rcrwm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['inside_proba'])
	tmp2 = tmp1.get_fdata()
	inside_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_proba, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xproba-rcrwm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['region_proba'])
	tmp2 = tmp1.get_fdata()
	region_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(region_proba, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xproba-rcrgm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['background_proba'])
	tmp2 = tmp1.get_fdata()
	background_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(background_proba, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xproba-rcrbg_cropped.nii.gz'))

	tmp1 = nb.load(T1map_masked)
	tmp2 = tmp1.get_fdata()
	T1map_masked_rightHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(T1map_masked_rightHemisphere, os.path.join(out_dir, 'sub-' + sub + filename + '_T1map_rightHemisphere_cropped.nii.gz'))

	del img
	del tmp
	del tmp1
	del tmp2
	print('Done.')

############################################################################
# 13.1. Crop additional data to right hemispehre
# -------------------
if reprocess_map_data:
	reprocess = True

if map_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Crop additional data to right hemisphere')
	print('*****************************************************')
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_map_data_rightHemisphere_cropped.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		map_data_rightHemisphere = os.path.join(out_dir, 'sub-' + sub + '_map_data_rightHemisphere_cropped.nii.gz')
	else:
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xmask-rcrgm.nii.gz'))
		tmp = img.get_fdata()
		tmp[tmp<0] = 0
		tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
		crop,coord = crop_img(tmp, pad=4, return_offset=True)

		tmp1 = nb.load(map_data)
		tmp2 = tmp1.get_fdata()
		map_data_rightHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
		nb.save(map_data_rightHemisphere, os.path.join(out_dir, 'sub-' + sub + '_map_data_rightHemisphere_cropped.nii.gz'))

		del img
		del tmp
		del tmp1
		del tmp2
		print('Done.')

############################################################################
# 13.2. Crop additionally transformed data to left hemispehre
# -------------------
if map_transform_file_onto_surface:
	print('')
	print('*****************************************************')
	print('* Crop transformed data to right hemisphere')
	print('*****************************************************')
	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_transform_data_rightHemisphere_cropped.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		transform_data_rightHemisphere = os.path.join(out_dir, 'sub-' + sub + '_transform_data_rightHemisphere_cropped.nii.gz')
	else:
		img = nb.load(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_xmask-rcrgm.nii.gz'))
		tmp = img.get_fdata()
		tmp[tmp<0] = 0
		tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
		crop,coord = crop_img(tmp, pad=4, return_offset=True)

		tmp1 = nb.load(transform_data)
		tmp2 = tmp1.get_fdata()
		transform_data_rightHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
		nb.save(transform_data_rightHemisphere, os.path.join(out_dir, 'sub-' + sub + '_transform_data_rightHemisphere_cropped.nii.gz'))

		del img
		del tmp
		del tmp1
		del tmp2
		print('Done.')

if reprocess_map_data:
	reprocess = False

#############################################################################
# 14. CRUISE cortical reconstruction (right hemisphere)
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
                        file_name='sub-' + sub + filename + '_rightHemisphere',
                        output_dir=out_dir)

###########################################################################
# 15. Extract layers across cortical sheet and map on surface (right hemisphere)
###########################################################################
# 15.1 Volumetric layering for depth measures (right hemisphere)
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
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers-allLayers',
                        output_dir=out_dir)['boundaries']
layers = nb.load(layers)

###########################################################################
# 15.2. Extract middle layer for surface generation and mapping (right hemisphere)
# ---------------------
# Here, we extract the levelset of the middle layer which will be used to
# map the information on.
print('')
print('*****************************************************')
print('* Extracting middle cortical layer (right hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-middleLayer.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = layers.get_fdata()[:,:,:,10:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-middleLayer.nii.gz'))
	del tmp

###########################################################################
# 15.3 Cortical surface generation and inflation of middle layer (right hemisphere)
# ---------------------
# Here, we create a surface from the levelset of the middle layer and
# inflate it afterwards for better visualisation.
corticalSurface = nighres.surface.levelset_to_mesh(
                        levelset_image=extractedLayers,
			connectivity='26/6',
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

inflatedSurface = nighres.surface.surface_inflation(
                        surface_mesh=corticalSurface['result'],
			max_iter=10000,
			max_curv=8.0,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

###########################################################################
# 15.4. Profile sampling of all layers (right hemisphere)
# ---------------------
# Here, we sample the T1 values of the T1map onto all layers.
profile = nighres.laminar.profile_sampling(
                        profile_surface_image=layers,
                        intensity_image=T1map_masked_rightHemisphere,
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers')['result']
profile = nb.load(profile)

###########################################################################
# 15.5. Extract all cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 3 to 18, removing layers close to the white
# matter and cerebrospinal fluid to avoid partial volume effects.
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting all cortical layers (right hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-allLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers-allLayers.vtk')


###########################################################################
# 15.6. Extract deep cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 3 to 6, to cover the "deep" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting deep cortical layers (right hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-deepLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:7]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-deepLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-deepLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers-deepLayers.vtk')

###########################################################################
# 15.7. Extract inner middle cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 7 to 10, to cover the "inner middle" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting inner middle cortical layers (right hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-innerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,7:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-innerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-innerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers-innerLayers.vtk')

###########################################################################
# 15.8. Extract outer middle cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 11 to 14, to cover the "outer middle" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting outer middle cortical layers (right hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-outerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,11:15]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-outerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-outerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers-outerLayers.vtk')

###########################################################################
# 15.9. Extract superficial cortical layers and map on surface (right hemisphere)
# ---------------------
# Here, we extract the layers 15 to 18, to cover the "superficial" layers. 
# Afterwards the information is mapped on the original and inflated
# surface of the middle layer.
print('')
print('*****************************************************')
print('* Extracting superficial cortical layers (right hemisphere)')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-superficialLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,15:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-superficialLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + filename + '_rightHemisphere_extractedLayers-superficialLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + filename + '_rightHemisphere_extractedLayers-superficialLayers.vtk')

###########################################################################
# 15.10. Profile sampling of additional data of all layers and mapping it onto
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
	print('*****************************************************')
	profile = nighres.laminar.profile_sampling(
		                profile_surface_image=layers,
		                intensity_image=map_data_rightHemisphere,
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_map_data_rightHemisphere_extractedLayers')['result']
	profile = nb.load(profile)

	print('')
	print('*****************************************************')
	print('* Extracting all cortical layers of additional data (right hemisphere)')
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_map_data_rightHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_map_data_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	else:
		tmp = profile.get_fdata()[:,:,:,1:19]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_map_data_rightHemisphere_extractedLayers-allLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_map_data_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		                intensity_image=meanProfile,
		                surface_mesh=corticalSurface['result'],
		                inflated_mesh=inflatedSurface['result'],
		                mapping_method='closest_point',
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_map_data_rightHemisphere_extractedLayers-allLayers.vtk')

###########################################################################
# 15.11. Profile sampling of transformed data of all layers and mapping it onto
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
	print('*****************************************************')
	profile = nighres.laminar.profile_sampling(
		                profile_surface_image=layers,
		                intensity_image=transform_data_rightHemisphere,
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_transform_data_rightHemisphere_extractedLayers')['result']
	profile = nb.load(profile)

	print('')
	print('*****************************************************')
	print('* Extracting all cortical layers of additional data (right hemisphere)')
	print('*****************************************************')

	if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_transform_data_rightHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
		print('File exists already. Skipping process.')
		meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_transform_data_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	else:
		tmp = profile.get_fdata()[:,:,:,1:19]
		extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
		nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_transform_data_rightHemisphere_extractedLayers-allLayers.nii.gz'))
		meanProfile = mean_img(extractedLayers)
		nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_transform_data_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
		del tmp

	nighres.surface.surface_mesh_mapping(
		                intensity_image=meanProfile,
		                surface_mesh=corticalSurface['result'],
		                inflated_mesh=inflatedSurface['result'],
		                mapping_method='closest_point',
		                save_data=True,
		                overwrite=reprocess,
		                output_dir=out_dir,
		                file_name='sub-' + sub + '_transform_data_rightHemisphere_extractedLayers-allLayers.vtk')

# Set reprocess flag to false in case left hemisphere or mapping of
# additional data should be done only.
if reprocess_rightHemisphere or reprocess_map_data:
	reprocess = False

print('')
print('*****************************************************')
print('* Processing finished successfully.')
print('*****************************************************')
