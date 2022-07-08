% This script implements the regularization proposed in:
%
% O’Brien, et al, 2014. 
% Robust T1-Weighted Structural Brain Imaging and Morphometry at 7T Using 
% MP2RAGE. 
% PLOS ONE 9, e99676. doi:10.1371/journal.pone.0099676
%
% which allows the creation of MP2RAGE T1w images without the strong 
% background noise in air regions. 
%
% although in the original paper the method only worked on raw multichannel
% data, here that contrain has been overcome and the correction can be
% implemented if both SOS images of the two inversion times exist and a
% MP2RAGE T1w image that has been calculated directly from the multichannel
% data as initially proposed in Marques et al, Neuroimage, 2009
%
%

addpath(genpath('.'))

MP2RAGE.filenameUNI='/data/tu_luesebrink/data/sensemap_example_data/wtl_younger/original data/07UNI/o20161103_111226MR131221107523418918wip900MP2RAGEWB07isowtl2964113s013a1001.nii'; % standard MP2RAGE T1w image;
MP2RAGE.filenameINV1='/data/tu_luesebrink/data/sensemap_example_data/wtl_younger/original data/07T1/o20161103_111226MR131221107523418918wip900MP2RAGEWB07isowtl2964113s011a1001.nii';% Inversion Time 1 MP2RAGE T1w image;
MP2RAGE.filenameINV2='/data/tu_luesebrink/data/sensemap_example_data/wtl_younger/original data/07INV2/o20161103_111226MR131221107523418918wip900MP2RAGEWB07isowtl2964113s008a1001.nii';% Inversion Time 2 MP2RAGE T1w image;
MP2RAGE.filenameOUT='/data/tu_luesebrink/data/sensemap_example_data/wtl_younger/original data/test.nii';% image without background noise;

[MP2RAGEimgRobustPhaseSensitive]=RobustCombination(MP2RAGE,[]);
 
% The script will then ask you if you are happy with the image you are 
% getting.
% Ideally it should look like an image that looks like the standard MPRAGE 
% (no noise in the background). If it has too much noise on the background, 
% give it a bigger value.
% 
% If the value is too big you will start noticing that the image gets a 
% bigger bias field -  which will deteriorate the segmentation results.
% Once you are happy with the value you found for one subject you can use 
% the same for all the following subjects by just calling the function like 
% this:
regularization = 10 ;
[MP2RAGEimgRobustPhaseSensitive]=RobustCombination(MP2RAGE,regularization);
