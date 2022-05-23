'''
Mapping of quantitative T1 values from MP2RAGE data onto surface
=======================================

This is a pipeline processing MP2RAGE data by performing the following steps:

01. Setup
02. MP2RAGE background cleaning
03. Resampling to 500 µm
04. Imhomogeneity correction and skull stripping
05. Registration of whole brain to slab data
06. Weighted images combination of full brain and slab data
07. Atlas-guided tissue classification using MGDM
08. Region extraction (left hemisphere) 
09. Crop volume (left hemisphere)
10. CRUISE cortical reconstruction (left hemisphere)
11. Extract layers across cortical sheet and map on surface (left hemisphere)
12. Region extraction (right hemisphere)
13. Crop volume (right hemisphere)
14. CRUISE cortical reconstruction (right hemisphere)
15. Extract layers across cortical sheet and map on surface (right hemisphere)
'''

############################################################################
# THINGS TO DO
# -------------------
# MAKE USE OF ANTsPy
# GIVE FILE FOR MAPPING ONTO SURFACE
# For cleaner pipeline, I should put pipelines for different purposes into different files
#	Whole brain and partial high resolution slab data
#	Whole brain data

############################################################################
# Instructions
# -------------------
# Things needed to be installed and added to PATH:
# 1. Shell scripts
# 2. MATLAB scripts
# 3. ANTs installation (probably can be done with Python entirely using ANTsPy)
# 4. SPM installation

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
import nibabel as nb
from nilearn.image import mean_img
from nilearn.image import crop_img
from nibabel.processing import conform

############################################################################
# 1.2. Set parameters
# -------------------
# Define BIDS path
BIDS_path = '/tmp/luesebrink/'
copy_data_from = 'gerd:/media/luesebrink/bmmr_data/data/sensemap/all/young/'

# Define subject following BIDS
sub = 'ben'

# Overwrite existing data?
reprocess = False

# Specify file to map onto surface here (needs full path)
# To make it work, the needs to be done in
# 	Registration
#	Cropping
#	Mapping
map_file = '/tmp/luesebrink/file.nii.gz'
if os.path.isfile(map_file):
	map_onto_surface = True
else:
	map_onto_surface = False

# Define path to atlas (probably not needed in newer versions as it should be the default one)
atlas = '/data/hu_luesebrink/venv/nighres/lib/python3.6/site-packages/nighres/atlases/brain-segmentation-prior3.0/brain-atlas-quant-3.0.9.txt'

############################################################################
# 1.3. Copy data to compute server and define file names
# -------------------
# Set paths
in_dir = BIDS_path + 'sub-' + sub + '/'
out_dir = BIDS_path + 'derivatives/sub-' + sub + '/'
os.system('mkdir -p ' + out_dir)

# Copy data
if os.path.isfile(os.path.join(in_dir, 'sub-' + sub + '_run-02_UNIT1.nii.gz'))  and reprocess != True:
	print('Files exists already. Skipping data transfer.')
else:
	os.system('scp -r ' + copy_data_from + sub + ' ' + BIDS_path + 'sub-' + sub + '/')

# Define file names
INV1 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-1_MP2RAGE.nii.gz')
INV2 = os.path.join(in_dir, 'sub-' + sub + '_run-01_inv-2_MP2RAGE.nii.gz')
T1map = os.path.join(in_dir, 'sub-' + sub + '_run-01_T1map.nii.gz')
UNI = os.path.join(in_dir, 'sub-' + sub + '_run-01_UNIT1.nii.gz')

INV1_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_inv-1_MP2RAGE.nii.gz')
INV2_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_inv-2_MP2RAGE.nii.gz')
T1map_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_T1map.nii.gz')
UNI_slab = os.path.join(in_dir, 'sub-' + sub + '_run-02_UNIT1.nii.gz')

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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	reg = str(10)
	OUT = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')
	os.system('removeBackground.sh ' + UNI + ' ' + INV1 + ' ' + INV2 + ' ' + OUT + ' ' + reg)

	OUT = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w.nii.gz')
	os.system('removeBackground.sh ' + UNI_slab + ' ' + INV1_slab + ' ' + INV2_slab + ' ' + OUT + ' ' + reg)	

# Update file names
T1w = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w.nii.gz')
T1w_slab = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w.nii.gz')

############################################################################
# 3. Resampling to higher resolution data
# -------------------
# Resample image to an isotropic resolution of 500 µm.
print('')
print('*****************************************************')
print('* Resampling volumes to an isotropic resolution of 500 µm')
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

# Update file names
T1map = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled.nii.gz')
T1w = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_resampled.nii.gz')

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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	os.system('removeBiasfield_sensemap.sh ' + T1w)
	os.system('removeBiasfield_sensemap.sh ' + T1map)

# Update file names
mask = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled_brainmask.nii.gz')
T1map_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled_biasCorrected.nii.gz')
T1w_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1w_resampled_biasCorrected.nii.gz')
T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-01_T1map_resampled_biasCorrected_masked.nii.gz')

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
	os.system('removeBiasfield_sensemap.sh ' + T1map_slab)
	os.system('removeBiasfield_sensemap.sh ' + T1w_slab)

# Update file names
T1map_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected.nii.gz')
T1w_slab_biasCorrected = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected.nii.gz')
T1map_slab_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1map_biasCorrected_masked.nii.gz')

############################################################################
# 5. Register native 7sur00 µm to upsampled 500 µm data
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

#
#
# I should try to make use of ANTsPy here instead of the actual ANTs installation.
# Probably, I need to change the preset, but otherwise that should work easily.
#
#
print('')
print('*****************************************************')
print('* Register native 500 µm to upsampled 500 µm data')
print('*****************************************************')

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_' + sub + '_run-01_T1map_biasCorrected_resampled_masked.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	os.system('antsRegistrationSyN_sensemap.sh -d 3 -f ' + T1map_slab_biasCorrected_masked + ' -m ' + T1map_biasCorrected_masked + ' -o ' + out_dir + 'sub-' + sub + '_run-01_biasCorrected_resampled_masked_registered_to_sub-' + sub + '_run-02_biasCorrected_masked_ -t sr')

	# Apply transformation to slab of T1map
	os.system('antsApplyTransforms -d 3 -e 0 -n BSpline[4] --float --verbose -i ' + T1map_slab + ' -r ' + T1map + ' -o ' + out_dir + 'sub-' + sub + '_run-02_registered_to_' + sub + '_run-01_T1map_biasCorrected_resampled_masked.nii.gz -t ' + out_dir + 'sub-' + sub + '_run-01_biasCorrected_resampled_masked_registered_to_sub-' + sub + '_run-02_biasCorrected_masked_1InverseWarp.nii.gz -t [' + out_dir + 'sub-' + sub + '_run-01_biasCorrected_resampled_masked_registered_to_sub-' + sub + '_run-02_biasCorrected_masked_0GenericAffine.mat,1]')

	# Apply transformation to the bias corrected slab of T1map
	os.system('antsApplyTransforms -d 3 -e 0 -n BSpline[4] --float --verbose -i ' + T1map_slab_biasCorrected + ' -r ' + T1map_biasCorrected + ' -o ' + out_dir + 'sub-' + sub + '_run-02_biasCorrected_registered_to_' + sub + '_run-01_T1map_biasCorrected_resampled_masked.nii.gz -t ' + out_dir + 'sub-' + sub + '_run-01_biasCorrected_resampled_masked_registered_to_sub-' + sub + '_run-02_biasCorrected_masked_1InverseWarp.nii.gz -t [' + out_dir + 'sub-' + sub + '_run-01_biasCorrected_resampled_masked_registered_to_sub-' + sub + '_run-02_biasCorrected_masked_0GenericAffine.mat,1]')

	# Apply transformation to slab of T1w
	os.system('antsApplyTransforms -d 3 -e 0 -n BSpline[4] --float --verbose -i ' + T1w_slab_biasCorrected + ' -r ' + T1w_biasCorrected + ' -o ' + out_dir + 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_' + sub + '_run-01_T1map_biasCorrected_resampled_masked.nii.gz -t ' + out_dir + 'sub-' + sub + '_run-01_biasCorrected_resampled_masked_registered_to_sub-' + sub + '_run-02_biasCorrected_masked_1InverseWarp.nii.gz -t [' + out_dir + 'sub-' + sub + '_run-01_biasCorrected_resampled_masked_registered_to_sub-' + sub + '_run-02_biasCorrected_masked_0GenericAffine.mat,1]')

# Update file names
T1w_slab_biasCorrected_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_T1w_biasCorrected_registered_to_' + sub + '_run-01_T1map_biasCorrected_resampled_masked.nii.gz')
T1map_slab_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_registered_to_' + sub + '_run-01_T1map_biasCorrected_resampled_masked.nii.gz')
T1map_slab_biasCorrected_reg = os.path.join(out_dir, 'sub-' + sub + '_run-02_biasCorrected_registered_to_' + sub + '_run-01_T1map_biasCorrected_resampled_masked.nii.gz')

############################################################################
# 6. Combination of native and upsampled data
# ----------------
# Here, we combine the slab and whole brain data by a weighted averaged
# using a custom MATLAB script. As the slab does not cover the entire
# temporal lobe, SNR is very low in the higher resolution MP2RAGE.
# Therefore, weighted averaging is applied in z-direction (superior to
# inferior) in the last third number of slices gradually increasing 
# the weighting with distance.
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
	os.system('combineData.sh ' + T1w_biasCorrected + ' ' + T1w_slab_biasCorrected_reg + ' ' + T1w_WM + ' ' + mask + ' ' + out_dir + 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected.nii.gz')
	os.system('combineData.sh ' + T1map + ' ' + T1map_slab_reg + ' ' + T1map_WM + ' ' + mask + ' ' + out_dir + '/sub-' + sub + '_merged_run-01+02_T1map.nii.gz')
	os.system('combineData.sh ' + T1map_biasCorrected + ' ' + T1map_slab_biasCorrected_reg + ' ' + T1map_WM + ' ' + mask + ' ' + out_dir + 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected.nii.gz')

# Update file names
T1w_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1w_biasCorrected_masked.nii.gz')
T1map_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_masked.nii.gz')
T1map_biasCorrected_masked = os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_biasCorrected_masked.nii.gz')

#############################################################################
# 7. MGDM classification
# ---------------------
# Next, we use the masked data as input for tissue classification with the
# MGDM algorithm. MGDM works with a single contrast, but can  be improved
# with additional contrasts potentially. Here, we make use of the T1map as
# well T1w part of the MP2RAGE.
mgdm_results = nighres.brain.mgdm_segmentation(
                        contrast_image1=T1map_biasCorrected_masked,
                        contrast_type1='T1map7T',
			contrast_image2=T1w_biasCorrected_masked,
			contrast_type2='Mp2rage7T',
                        n_steps=10,
                        max_iterations=1000,
                        atlas_file=atlas,
                        normalize_qmaps=False,
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02')

###########################################################################
# 8. Region Extraction (left hemisphere)
# ------------------
# Here, we pull from the MGDM output the needed regions for cortical
# reconstruction: the GM cortex ('region'), the underlying WM (with filled
# subcortex and ventricles, 'inside') and the surrounding CSF (with masked
# regions, 'background')
cortex = nighres.brain.extract_brain_region(segmentation=mgdm_results['segmentation'],
                                            levelset_boundary=mgdm_results['distance'],
                                            maximum_membership=mgdm_results['memberships'],
                                            maximum_label=mgdm_results['labels'],
			                    atlas_file=atlas,
                                            extracted_region='left_cerebrum',
                                            save_data=True,
                                            overwrite=reprocess,
                                            file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere',
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
if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_leftHemisphere_cropped.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	inside_mask = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xmask-lcrwm_cropped.nii.gz'))
	inside_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xproba-lcrwm_cropped.nii.gz'))
	region_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xproba-lcrgm_cropped.nii.gz'))
	background_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xproba-lcrbg_cropped.nii.gz'))
	T1map_masked_leftHemisphere = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_leftHemisphere_cropped.nii.gz'))
else:
	img = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xmask-lcrgm.nii.gz'))
	tmp = img.get_fdata()
	tmp[tmp<0] = 0
	tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
	crop,coord = crop_img(tmp, pad=4, return_offset=True)

	tmp1 = nb.load(cortex['inside_mask'])
	tmp2 = tmp1.get_fdata()
	inside_mask = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_mask, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xmask-lcrwm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['inside_proba'])
	tmp2 = tmp1.get_fdata()
	inside_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_proba, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xproba-lcrwm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['region_proba'])
	tmp2 = tmp1.get_fdata()
	region_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(region_proba, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xproba-lcrgm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['background_proba'])
	tmp2 = tmp1.get_fdata()
	background_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(background_proba, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_xproba-lcrbg_cropped.nii.gz'))

	tmp1 = nb.load(T1map_masked)
	tmp2 = tmp1.get_fdata()
	T1map_masked_leftHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(T1map_masked_leftHemisphere, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_leftHemisphere_cropped.nii.gz'))
	del tmp
	del tmp1
	del tmp2


#############################################################################
# 10. CRUISE cortical reconstruction (left hemisphere)
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
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere',
                        output_dir=out_dir)

###########################################################################
# 11. Extract layers across cortical sheet and map on surface
###########################################################################
# 11.1 Volumetric layering for depth measures (left hemisphere)
# ---------------------
# Finally, we use the GM/WM boundary (GWB) and CSF/GM boundary (CGB) from
# CRUISE to compute cortical depth with a volume-preserving technique.
layers = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise['gwb'],
                        outer_levelset=cruise['cgb'],
                        n_layers=20,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-allLayers',
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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-middleLayer.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	extractedLayers=nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-middleLayer.nii.gz'))
else:
	tmp = layers.get_fdata()[:,:,:,10:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-middleLayer.nii.gz'))
	del tmp

###########################################################################
# 11.3 Cortical surface generation and inflation of middle layer (left hemisphere)
# ---------------------
# Here, we create a surface from the levelset of the middle layer and
# inflate it afterwards for better visualisation.
corticalSurface = nighres.surface.levelset_to_mesh(
                        levelset_image=extractedLayers,
			connectivity='26/6',
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

inflatedSurface = nighres.surface.surface_inflation(
                        surface_mesh=corticalSurface['result'],
			max_iter=10000,
			max_curv=8.0,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

###########################################################################
# 11.4. Profile sampling of all layers (left hemisphere)
# ---------------------
# Here, we sample the T1 values of the T1map onto all layers.
profile = nighres.laminar.profile_sampling(
                        profile_surface_image=layers,
                        intensity_image=T1map_masked_leftHemisphere,
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers')['result']
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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	meanProfile = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
else:
	tmp = profile.get_fdata()[:,:,:,3:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-allLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-allLayers.vtk')


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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-deepLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:7]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-deepLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-deepLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-deepLayers.vtk')

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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-innerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,7:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-innerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-innerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-innerLayers.vtk')

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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-outerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,11:15]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-outerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-outerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-outerLayers.vtk')

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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-superficialLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,15:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-superficialLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-superficialLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_leftHemisphere_extractedLayers-superficialLayers.vtk')

###########################################################################
# 12. Region Extraction (right hemisphere)
# ------------------
# Here, we pull from the MGDM output the needed regions for cortical
# reconstruction: the GM cortex ('region'), the underlying WM (with filled
# subcortex and ventricles, 'inside') and the surrounding CSF (with masked
# regions, 'background')
cortex = nighres.brain.extract_brain_region(segmentation=mgdm_results['segmentation'],
                                            levelset_boundary=mgdm_results['distance'],
                                            maximum_membership=mgdm_results['memberships'],
                                            maximum_label=mgdm_results['labels'],
			                    atlas_file=atlas,
                                            extracted_region='right_cerebrum',
                                            save_data=True,
                                            overwrite=reprocess,
                                            file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere',
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
if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_rightHemisphere_cropped.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
	inside_mask = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xmask-rcrwm_cropped.nii.gz'))
	inside_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xproba-rcrwm_cropped.nii.gz'))
	region_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xproba-rcrgm_cropped.nii.gz'))
	background_proba = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xproba-rcrbg_cropped.nii.gz'))
	T1map_masked_rightHemisphere = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_rightHemisphere_cropped.nii.gz'))
else:
	img = nb.load(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xmask-rcrgm.nii.gz'))
	tmp = img.get_fdata()
	tmp[tmp<0] = 0
	tmp = nb.Nifti1Image(tmp, affine=img.affine, header=img.header)
	crop,coord = crop_img(tmp, pad=4, return_offset=True)

	tmp1 = nb.load(cortex['inside_mask'])
	tmp2 = tmp1.get_fdata()
	inside_mask = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_mask, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xmask-rcrwm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['inside_proba'])
	tmp2 = tmp1.get_fdata()
	inside_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(inside_proba, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xproba-rcrwm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['region_proba'])
	tmp2 = tmp1.get_fdata()
	region_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(region_proba, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xproba-rcrgm_cropped.nii.gz'))

	tmp1 = nb.load(cortex['background_proba'])
	tmp2 = tmp1.get_fdata()
	background_proba = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(background_proba, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_xproba-rcrbg_cropped.nii.gz'))

	tmp1 = nb.load(T1map_masked)
	tmp2 = tmp1.get_fdata()
	T1map_masked_rightHemisphere = nb.Nifti1Image(tmp2[coord], affine=tmp1.affine, header=tmp1.header)
	nb.save(T1map_masked_rightHemisphere, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_T1map_rightHemisphere_cropped.nii.gz'))
	del tmp
	del tmp1
	del tmp2

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
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere',
                        output_dir=out_dir)

###########################################################################
# 15. Extract layers across cortical sheet and map on surface
###########################################################################
# Finally, we use the GM/WM boundary (GWB) and CSF/GM boundary (CGB) from
# CRUISE to compute cortical depth with a volume-preserving technique.
# ---------------------
layers = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise['gwb'],
                        outer_levelset=cruise['cgb'],
                        n_layers=20,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-allLayers',
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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-middleLayer.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = layers.get_fdata()[:,:,:,10:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-middleLayer.nii.gz'))
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
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-middleLayer.vtk',
                        output_dir=out_dir)

inflatedSurface = nighres.surface.surface_inflation(
                        surface_mesh=corticalSurface['result'],
			max_iter=10000,
			max_curv=8.0,
                        save_data=True,
                        overwrite=reprocess,
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-middleLayer.vtk',
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
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers')['result']
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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-allLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-allLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-allLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-allLayers.vtk')


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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-deepLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,3:7]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-deepLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-deepLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-deepLayers.vtk')

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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-innerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,7:11]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-innerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-innerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-innerLayers.vtk')

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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-outerLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,11:15]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-outerLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-outerLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-outerLayers.vtk')

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

if os.path.isfile(os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-superficialLayers_mean.nii.gz')) and reprocess != True:
	print('File exists already. Skipping process.')
else:
	tmp = profile.get_fdata()[:,:,:,15:19]
	extractedLayers = nb.Nifti1Image(tmp, affine=layers.affine, header=layers.header)
	nb.save(extractedLayers, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-superficialLayers.nii.gz'))
	meanProfile = mean_img(extractedLayers)
	nb.save(meanProfile, os.path.join(out_dir, 'sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-superficialLayers_mean.nii.gz'))
	del tmp

nighres.surface.surface_mesh_mapping(
                        intensity_image=meanProfile,
                        surface_mesh=corticalSurface['result'],
                        inflated_mesh=inflatedSurface['result'],
                        mapping_method='closest_point',
                        save_data=True,
                        overwrite=reprocess,
                        output_dir=out_dir,
                        file_name='sub-' + sub + '_merged_run-01+02_rightHemisphere_extractedLayers-superficialLayers.vtk')
