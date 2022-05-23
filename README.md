# Introduction
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

# Installation instructions
Things needed to be installed and added to PATH:
1. Shell scripts (not included in the repository yet) 
2. MATLAB scripts (not included in the repository yet)
3. ANTs installation (probably can be done with Python entirely using ANTsPy)
4. SPM installation

# Instructions for use
Data should be formated according to BIDS. Change paths and subject name to run.

# Things to do
1. Make use of ANTsPy (instead of ANTs)
2. Give file dor mapping onto surface
3. For cleaner code, put pipelines into different files and call these
