# Introduction
This is a pipeline for mapping quantitative T1 values of MP2RAGE data on the brain surface.

The script requires whole brain MP2RAGE data organized according to BIDS (https://bids-specification.readthedocs.io/). You can make use of a high resolution MP2RAGE slab additionally. The slab will be merged into the whole brain volume automatically and used for further processing. 

Furthermore, you can specify volumes which will be mapped on the surface additionally. This is meant for mapping information of another contrast on the surface, e.g. QSM data or fMRI results.

The pipeline consists of following stages:
01. Setup
02. MP2RAGE background cleaning
03. Resampling to 500 µm (only for hires option)
04. Imhomogeneity correction and skull stripping
05. Registration of whole brain to slab data (only for hires option)
06. Weighted image combination of full brain and slab data (only for hires option)
07. Atlas-guided tissue classification using MGDM
08. Region extraction (left hemisphere) 
09. Crop volume (left hemisphere)
10. CRUISE cortical reconstruction (left hemisphere)
11. Extract layers across cortical sheet and map on surface (left hemisphere)
12. Region extraction (right hemisphere)
13. Crop volume (right hemisphere)
14. CRUISE cortical reconstruction (right hemisphere)
15. Extract layers across cortical sheet and map on surface (right hemisphere)

## Installation instructions
Things needed to be installed:
1. Nighres (https://nighres.readthedocs.io/en/latest/installation.html)
2. antspy (https://github.com/ANTsX/ANTsPy)
3. MATLAB
4. SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/download/)
5. MP2RAGE-related-scripts (https://github.com/JosePMarques/MP2RAGE-related-scripts)
6. Custom MATLAB scripts (weightedAverage, removeBackgroundnoise, and biasCorrection)

You need to change the path to the tissue probability model of SPM12 for the bias
field correction method. This needs to be done in ./biasCorrection/preproc_sensemap.m on line 19

## Brief instructions for use
1. For detailed description cf. surfaceMapping.py
2. Data needs be formated according to BIDS
3. Paths, files, and flags need to be changed at the beginning of the Python script in the "set parameters" section

## Things to do
1. Get rid of MATLAB dependencies
2. Specify files, paths, and flags from outside the script
3. Flag to write all data to disk or final results only
4. Add logging feature

## Version
0.96 (08.07.2022)

## Contact information
Dr. Falk Luesebrink
(falk dot luesebrink at ovgu dot de)
