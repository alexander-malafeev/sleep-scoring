function [PEEG] = amf_spectrogram(S, fs, epochl, windowl)
% this function computes spectrogram of the signal

maxep = floor(numel(S)/(fs*epochl));
if (size(S,1)==1) ||  (size(S,2)==1)
    S = reshape(S, [fs*epochl], [] );
end

for ep=1:maxep
    data = S(:,ep);    
    PEEG(:,ep)=pwelch(data,hanning(windowl*fs),0,windowl*fs,fs);
end

end