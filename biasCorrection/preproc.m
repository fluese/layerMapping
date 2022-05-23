function preproc(data)
% Pre-processes high resolution high field MRI data intended for
% quantification of cortical (and sub-cortical) structures.
% 
% Run bias field correction of SPM12. Allows to extract WM, GM, and CSF
% segmentation and generation of a brainmask based on these segmentations.
%
% Falk Luesebrink (22.11.2017)
%
% Minor changes
% Falk Luesebrink (12.02.2019)

param.seg     = false;      % Write segmentation of GM, WM, CSF
param.mask    = false;      % Create brainmask (requires param.seg = true)
param.gzip    = true;      % Write compressed NIfTIs after processing

% Parameters for bias field correction:
% See 'biasCorrection.m' for in-depth explantion of parameters.
param.path    = '/data/tu_luesebrink/software/spm12/tpm/TPM.nii'; % Set path to tissue probability model
param.fwhm    = 30;        % FWHM (default: 60)
param.reg     = 0.1;     % Regularization (default: 0.001)
param.samp    = 2;         % Sampling distance (default: 3)

% Parameters for brainmask
param.dilate  = 10;       % Dilate brainmask by cube with edge length of param.dilate
param.erode   = 10;       % If greater than 0, erode brainmask after dilation

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Bias correction, segmentation, and brain mask creation %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('************************************\n')
fprintf('* Bias correction and segmentation  \n')
fprintf('* Processing file: %s\n', data);
fprintf('************************************\n\n')

[path, name, ext] = fileparts(data);
if strcmp(ext, '.gz')
    gunzip(data);
    data=data(1:end-3);
    param.gzip = true;
    [path, name, ext] = fileparts(data);
end
biasCorrection(data, param);

if param.mask == 1 && param.seg == 1
    % Creating brain mask by combining the segmentations, thresholding
    % it at greater than 0.5, and applying morphological operators to
    % refine it.
    fprintf('\n')
    fprintf('*****************************************\n')
    fprintf('* Creating brain mask from segmentation *\n')
    fprintf('*****************************************\n\n')    
   
    HeaderInfo = spm_vol_nifti([path '/c1' name ext]);
    i1         = spm_read_vols(HeaderInfo);
    HeaderInfo = spm_vol_nifti([path '/c2' name ext]);
    i2         = spm_read_vols(HeaderInfo);
    HeaderInfo = spm_vol_nifti([path '/c3' name ext]);
    i3         = spm_read_vols(HeaderInfo);

    mask = (i1+i2+i3)>0.5;
    mask = imdilate(mask,ones(param.dilate,param.dilate,param.dilate));
    if param.erode > 0
        mask = imerode(mask,ones(param.erode,param.erode,param.erode));
    end

    HeaderInfo = spm_vol_nifti([path '/c1' name ext]);

    HeaderInfo.fname = [path '/' name '_brainmask.nii'];
    HeaderInfo.private.dat.fname = HeaderInfo.fname;

    spm_write_vol(HeaderInfo,mask);
    if param.gzip
        gzip(HeaderInfo.fname)
        gzip([path '/c1' name ext])
        gzip([path '/c2' name ext])
        gzip([path '/c3' name ext])

        delete(HeaderInfo.fname)
        delete([path '/c1' name ext])
        delete([path '/c2' name ext])
        delete([path '/c3' name ext])
    end
end

if param.gzip
    disp(['Zip: ' path '/' name ext])
    gzip([path '/' name ext])
    disp(['Zip: ' path '/' name '_biasCorrected' ext])
    gzip([path '/' name '_biasCorrected' ext])
    %disp(['Remove: ' path '/' name ext])
    delete([path '/' name ext])
    %disp(['Remove: ' path '/' name '_biasCorrected' ext])
    delete([path '/' name '_biasCorrected' ext])
end
fprintf('Done.\n')
fprintf('************************************\n\n')
