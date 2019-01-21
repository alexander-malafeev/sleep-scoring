function [  pEMG ] = powerEMG( EMG, fs,  epochl, windowl )

df = 1/windowl;
% lower and upper bound of the frequency range
fl = 15; 
fh = 30;
% indexes of corresponding frequency bins in 15-30 Hz range 
ind_pEMG = round(fl/df)+1:round(fh/df)+1; 

maxep = numel(EMG)/(fs*epochl);
pEMG = zeros(1,maxep);


for ep=1:maxep
    % indexes of the samples corresponding to the beginning
    % and to the end of the current epoch
    i1 = (ep-1)*epochl*fs+1;
    i2 = (ep)*epochl*fs;
    data = EMG(i1:i2);
    
    % compute power spectra of the epoch
    PEMG = pwelch(data,hanning(windowl*fs),0,windowl*fs,fs);
    % find the power in the 15-30 Hz range
    pEMG(ep) = sum(PEMG(ind_pEMG));
end
pEMG = pEMG';
end