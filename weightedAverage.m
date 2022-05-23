function weightedAverage(Full, Slab, WM, Mask, Out)

dbstop if error

fprintf('***********************************\n')
fprintf('* Combing slab and full brain data!\n')
fprintf('***********************************\n\n')
fprintf('Full: %s\n', Full)
fprintf('Slab: %s\n', Slab)
fprintf('WM: %s\n', WM)
fprintf('Mask: %s\n', Mask)
fprintf('Output: %s\n', Out)

[path, name, ext] = fileparts(Out);
if strcmp(ext, '.gz')
    temp=Out(1:end-3);
    zipped = true;
    [path, name, ext] = fileparts(temp);
end

if zipped == 0
    Out_masked = [path '/' name '_masked' ext];
else
    Out_masked = [path '/' name '_masked' ext '.gz'];
end

fprintf('Masked: %s\n', Out_masked)

% Load data (Should change to SPM stuff because I'm using it anyway)
Full = load_untouch_nii(Full);
Slab = load_untouch_nii(Slab);
WM   = load_untouch_nii(WM);
Mask = load_untouch_nii(Mask);

% Equalize precision
Slab.img = single(Slab.img);
Full.img = single(Full.img);
WM.img   = single(WM.img);
Mask.img = single(Mask.img);

% Get size of volume
sz=size(Slab.img);

% Create new NIfTI file
weightedVolume     = Slab;
weightedVolume.img = single(zeros(sz));
weightedVolume     = rmfield(weightedVolume,'untouch');

% Binarize white matter segmentation and mask
WM.img   = imbinarize(WM.img);
Mask.img = imbinarize(Mask.img);

% Erode white matter segmentation to remove outliers.
se     = strel('cube',5);
WM.img = imerode(WM.img,se);

% Calculate ratio of white matter intensities between full brain and slab
% data. Then adjust intensity of full data by the ratio before combining
% the data.
Masked_Full = Full.img.*WM.img;
Masked_Slab = Slab.img.*WM.img;
Ratio       = mean(Masked_Full(Masked_Full>0)) / mean(Masked_Slab(Masked_Slab>0));
Full.img    = Full.img./Ratio;

% 1. Check for overlap between whole brain acquisition and slab
% 2. If overlap, move along z-direction until there is no more overlap
% 3. Calculate weighted average along overlapping voxels
% 4. Store weighted average into new volume
for x=1:sz(1)
    for y=1:sz(2)
        for z=sz(3):-1:1
            if Full.img(x,y,z) ~= 0 && Slab.img(x,y,z) == 0 && weightedVolume.img(x,y,z) == 0
                    weightedVolume.img(x,y,z) = Full.img(x,y,z);
            elseif Full.img(x,y,z) ~= 0 && Slab.img(x,y,z) ~= 0 && weightedVolume.img(x,y,z) == 0
                if z >= round(sz(3)*2/3)
                    weightedVolume.img(x,y,z) = Slab.img(x,y,z);
                else
                    count = 0;
                    while Full.img(x,y,z-count) ~= 0 && Slab.img(x,y,z-count) ~= 0 && weightedVolume.img(x,y,z) == 0
                        count = count + 1;
                         if z-count < 1
                             count = count - 1;
                             break
                         end
                    end

                    for weight_in_z=0:count
                        weightedVolume.img(x,y,z-weight_in_z) = (Full.img(x,y,z-weight_in_z)*(weight_in_z./count)) + (Slab.img(x,y,z-weight_in_z)*(1-(weight_in_z./count)));
                    end
                end
            end
            
            if weightedVolume.img(x,y,z) < 0
                weightedVolume.img(x,y,z) = 0;
            end
        end
    end
end

% Save new NIfTI file
save_nii(weightedVolume, Out)

% Mask weighted volume
weightedVolume_masked     = weightedVolume;
weightedVolume_masked.img = weightedVolume.img .* Mask.img;
save_nii(weightedVolume_masked, Out_masked)
