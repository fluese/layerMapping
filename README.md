# Introduction
This is a pipeline for depth dependent mapping of quantitative T1 values on the brain surface using MP2RAGE data as primary input.

The script requires whole brain MP2RAGE data organized according to BIDS (https://bids-specification.readthedocs.io/). You can make use of a high resolution MP2RAGE slab additionally. The slab will be merged into the whole brain volume automatically and used for further processing. However, currently this is hardcoded to an isotropic resolution of 500 µm.

Furthermore, you can specify volumes which will be mapped on the surface additionally. This is meant for mapping information of another contrast on the surface, e.g. QSM data or fMRI results.

The pipeline consists of following stages:
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

## Installation instructions
Things needed to be installed:
1. Nighres (tested with v1.4: https://nighres.readthedocs.io/en/latest/installation.html)
2. antspy (tested with v0.3.2: https://github.com/ANTsX/ANTsPy)
3. FreeSurfer (tested with v7.3.2: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)
4. MATLAB (tested with R2022a)
5. MP2RAGE-related-scripts (https://github.com/JosePMarques/MP2RAGE-related-scripts)
6. Tools for NIfTI and Analyze image (https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
7. Custom MATLAB scripts (weightedAverage and removeBackgroundnoise)

## Brief instructions for use
This script requires whole brain MP2RAGE data organized according to BIDS. The script expects the first and second inversion, the T1 weighted data as
well as the T1 map. You can make use of a high resolution MP2RAGE slab additionally. The slab will be merged into the whole brain volume automatically and used for further processing. To make use of a high resolution slab, the slab needs to be acquired as the second run in a session (or at least named as if it were acquired in the same session).

In the subsection "set parameters" of the setup section, you need to specify the folder to the BIDS directory and the label of the subject you want to process. This can be done in multiple ways. First, specify the path and file in the script directly. Secondly, when the script is called the next two inputs are for the subject ID and path to the BIDS directory, respectively. Lastly, if all is left empty, the user will be asked to specify a subject ID and the full path to the BIDS directory at the beginning of the script.

Example use (Case 1):
python3 layerMapping.py aaa /path/to/BIDS/data/

Example use (Case 2):
python3 layerMapping.py aaa

Example use (Case 3):
python3 layerMapping.py

In case 1 the subject with the ID 'aaa' will be processed. The according data is expected in '/path/to/BIDS/data/'. Output will be generated in  '/path/to/BIDS/data/derivatives/sub-aaa/'. In case 2, either the path to the data needs to be defined in the script itself in the variable 'BIDS_path' or if left empty the user is asked to specify the path to the data. In case 3, the same for the path is true for the subject ID. It can either be specified in the script itself in the 'sub' variable or if left empty, the user will be asked for an ID during processing.

Furthermore, you can specify a single volume which will be mapped on the surface additionally. This is meant for mapping another contrast  on the  surface, e.g. QSM data or fMRI results. The resulting transformation can be applied either to another volume or to all compressed NIfTI files of an entire directory. Currently, this can be done in the script only.

To enable logging of the output, one can send the screen output into a file. This can be done, e.g. by using following command:

python3 -u layerMapping.py aaa |& tee -a /tmp/luesebrink/sensemap/derivatives/sub-aaa/layerMapping.log

A few flags have been set up which allow to reprocess data at various stages of the pipeline (e.g. entirely, from the segmentation onwards, mapping of additional data, or per hemisphere). In case you want to map multiple files onto the surface additionally, this is especially useful as you don't have to process all other data again. Simply specify a file under 'map_data' that shall be mapped onto the surface and change the 'reprocess_map_data' flag to True.

## Sample data:
Sample data can be made avaiable upon request.

## Version
1.1 (23.02.2023)

## Contact information
Dr. Falk Luesebrink
(falk dot luesebrink at ovgu dot de)

## Acknowledgement
This work was supported by CRC 1436 "Neural Resources of Cognition" of the German Research Foundation (DFG) under project ID 425899996.
