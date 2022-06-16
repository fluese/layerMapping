# Introduction
This is a pipeline processing MP2RAGE data to map information on a surface by performing the following steps:

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

## Installation instructions
Things needed to be installed:
1. Working MATLAB installation
2. Working SPM installation
3. MATLAB scripts (weightedAverage and biasCorrection) added to Path

You need to change the path to the tissue probability model for the bias field correction method. This needs to be done in
./biasCorrection/preproc_sensemap.m on line 19

## Brief instructions for use
1. For detailed description cf. surfaceMapping.py
2. Data needs be formated according to BIDS
3. Paths, files, and flags need to be changed at the beginning of the Python script in the "set parameters" section

## Things to do
1. Get rid of MATLAB dependencies
2. Specify files, paths, and flag from outside this script
3. Apply transformation to multiple volumes instead of a single file
4. Differentiate between data type to be mapped on the surface, e.g. QSM, fMRI, other types.
