function removeBackgroundnoise(UNI, INV1, INV2, OUT, regularization)
% Remove background noise of MP2RAGE data using the code of Marques and the
% method of O'Brien.
MP2RAGE.filenameUNI  = UNI;
MP2RAGE.filenameINV1 = INV1;
MP2RAGE.filenameINV2 = INV2;
MP2RAGE.filenameOUT  = OUT;

RobustCombination(MP2RAGE, regularization);
end
