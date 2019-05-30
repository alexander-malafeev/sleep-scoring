clear; clc; close all; fclose all;

% we need a function to read edf
% download matlab scripts from https://ch.mathworks.com/matlabcentral/fileexchange/31900-edfread
% and put them into the directory with this file

% it is the path to the folder with edf files
readPath = './../EDF/'

% directory with the output data
writePath = '../mat/'


% labels in EDF file
% in order to find necessary channels, dependign on the equipment labels
% can be different
% in this example we reference C3 to A2 ourselfs, sometimes there is 
% an already referenced channel in the recording

C3_lbl = 'C3';
A2_lbl = 'A2';

C3A2_lbl = 'C3_A2';
C4A1_lbl = 'C4_A1';
LOC_lbl = 'ROC_A1';
ROC_lbl = 'LOC_A1';
EMG_lbl = 'EMG';


% frequency of powerline; 60 Hz in USA
f_powerline = 50;


% it is the epoch length for the spectrogram calculation
epochl=20; % seconds

% window length used for spectrogram calculation
% we split every epoch into windows, in this case 20 second epoch is split
% into 4 windows, each 5 seconds long
% then spectra is computed for every window and avaraged
windowl=5; % seconds

% resolution of a spectrogram in frequency axis
df=1/windowl;
% default sampling rate
fs = 256;

% indexes of spectrogram columns to compute power in certain ranges
ind_pEMG=15/df+1:30/df+1; % 15-30 Hz range of EMG

%%ind_pEOG=1/df+1:5/df+1; % power in 1-5 Hz range of EOG


% set it to 0 if you don't want plots
plotspectra = 1;

if ( ~exist([writePath ], 'dir') ) 
   mkdir(writePath);
end


fileList = dir( [ readPath '*.edf' ] )



for f = 1:length(fileList)
    
    recording  = fileList(f).name
    recording = recording(1:end-4);
    
    fileName = [recording '.REC'];
    [HDR, data_rec] = edfread([readPath, fileName]);
    data_rec = data_rec';

    % here we are finding which rows of data matrix correspond to 
    % the channels of interest
    %chF3A2=ismember(HDR.Label,'F3-A2','rows');
    chC3A2=ismember(HDR.label,C3A2_lbl);
    %chO1A2=ismember(HDR.Label,'O1-A2','rows');
    %chF4A1=ismember(HDR.Label,'F4-A1','rows');
    chC4A1=ismember(HDR.label,C4A1_lbl);
    %chO2A1=ismember(HDR.Label,'O2-A1','rows');
    
    chC3=ismember(HDR.label,C3_lbl);
    chA2=ismember(HDR.label,A2_lbl);
    
    chEMG=ismember(HDR.label,EMG_lbl);
    %chEMG1=ismember(HDR.Label,'CHIN1-CHIN2','rows');
    %chEMG2=ismember(HDR.Label,'CHIN2-CHIN3','rows');
    chEOG1=ismember(HDR.label,LOC_lbl);
    chEOG2=ismember(HDR.label,ROC_lbl);
    
    % maximal index of used channels 
    % we need it to trim the data matrix
    nChan = max([find(chC3A2==1), find(chC4A1==1), find(chEOG1==1), find(chEOG2==1), find(chEMG==1) ]);


    % some data from the header
    %fs=HDR.SampleRate; %sample rate
    %fse=HDR.SampleRate; %sample rate EOG
    %fsm=HDR.SampleRate; %sample rate EMG
    fs = HDR.frequency(chC3A2);
    fse = fs;
    fsm = fs;
    % how many 20 second epochs are in our file
    maxep=floor(HDR.records/(epochl/HDR.duration)); 

    % arrays for the spectrogramms 
    % we will use only PC3A2, but you can uncomment and use others as well
    
    %PF3A2=zeros(fs/2/df+1,maxep);
    PC3A2=zeros(fs/2/df+1,maxep);
    %PO1A2=zeros(fs/2/df+1,maxep);
    %PF4A1=zeros(fs/2/df+1,maxep);
    PC4A1=zeros(fs/2/df+1,maxep);
    %PO2A1=zeros(fs/2/df+1,maxep);
    
    % array for power of EMG
    powEMG=zeros(1,maxep); % EMG power in the band 15-30 Hz
    


    %%S = zeros(1, maxep*epochl*fs);
    %HDR.AS.SPR = HDR.AS.SPR*0;
    for ep=1:maxep
        % read epochl seconds from the file
        %[data,HDR]=sread(HDR,epochl);
        
        % put corresponding epoch into the data matrix
        %data_rec( (ep-1)*epochl*fs+1:ep*epochl*fs, : ) = data(:,1:nChan);
        data(:,:) = data_rec( (ep-1)*epochl*fs+1:ep*epochl*fs, : );
        % compute PEMG
        PEMG=pwelch(data(:,chEMG),hanning(windowl*fs),0,windowl*fs,fs);

        % find the power in the 15-30 Hz band
        powEMG(ep)=sum(PEMG(ind_pEMG));
    end

    % notch filter to remove powerline noise    
    wo = f_powerline/(fs/2);  
    bw = wo/35;
    [b,a] = iirnotch(wo,bw);
    % apply filter for every channel
    for i = 1:size(data_rec,2)
        data_rec(:,i)= filter(b,a,data_rec(:,i));
    end

    % apply low pass filter with the cutoff frequency 35 Hz
    % we did not do it for the paper, but we noticed that it improves the 
    % performance, especially when recordings are noisy
    f_lp  = 35;
    [b1,a1] = butter(4,f_lp/(fs/2),'low');
    for i = 1:size(data_rec,2)
        data_rec(:,i)= filter(b1,a1,data_rec(:,i)) ;
    end

    %%C3A2 = data_rec(:,chC3A2);
    %%C4A1 = data_rec(:,chC4A1);

    % get channels of interest from the data matrix



    %%maxep =  floor(numel(C3A2)/(fs*epochl));

    %C3A2 = C3A2( 1:maxep*epochl*fs);
    %C4A1 = C4A1( 1:maxep*epochl*fs);
    % F3A2 = C3A2( 1:maxep*epochl*fs);
    % F4A1 = C4A1( 1:maxep*epochl*fs);
    % O1A2 = O1A2( 1:maxep*epochl*fs);
    % O2A1 = O2A1( 1:maxep*epochl*fs);

    % truncate signals, i.e. it should contain integer number of epochs
    data_rec = data_rec(1:maxep*epochl*fs,:);
    %C3A2 = data_rec(:,chC3A2);
    %A2 = data_rec(:,chA2);
    % structure Data is what we are going to save
    Data.windowl = windowl;


    FFT = PC3A2;



    % this array contains stages of sleep
    % we set it to zeros because for scoring your data it is not needed
    % if you want to train networks on your own data you have to set this array
    % 0 is Wake, 1, 2, 3 - stages 1-3 and 4 is for REM sleep
    stages = zeros(1,maxep);
    Data.stages = stages;

    % save the channel's labels
    Data.channel_names = HDR.label;


    % we resample the data to 128 Hz; your data should have sampling rate
    % higher than 128 Hz

    fn = 128; % new frequency

    % ===== resample=========
    g =  gcd(fn,fs);
    p = fn/g;
    q = fs/g;

    
    %A2 = resample(A2,p,q);
    % C3A2 = resample(C3A2,p,q);
    % C4A1 = resample(C4A1,p,q);
    % F3A2 = resample(C3A2,p,q);
    % F4A1 = resample(C4A1,p,q);
    % O1A2 = resample(O1A2,p,q);
    % O2A1 = resample(O2A1,p,q);

    % EMG and ocular channels
    C3A2 = resample(data_rec(:,chC3A2),p,q);
    sEMG = resample(data_rec(:,chEMG),p,q);
    sLOC = resample(data_rec(:,chEOG1),p,q);
    sROC = resample(data_rec(:,chEOG2),p,q);

    % set sampling rate of the Data structure to the new frequency
    Data.fs = fn;
    Data.epochl = epochl;

    % save the EEG signal
    %C3A2 = C3-A2;
    Data.C3A2 = reshape(C3A2, [Data.fs*epochl, maxep]);

    % here we compute the spectrogram
    [ PEEG ] = amf_spectrogram(C3A2, Data.fs, Data.epochl, Data.windowl);
    Data.Pspec = PEEG;
    % and the emg power
    [ PEMG ] = powerEMG( sEMG, Data.fs,  Data.epochl, Data.windowl );

    % here we do some standartization of the EMG data
    % we did not do it in the paper, but we noticed that it helps a lot
    % if recordings are noisy
    % first we log transform the data
    PEMG = log(PEMG+1.0);
    % then we subtract the lower 10th percentile to make the lowest 
    % values zero (during REM sleep)
    % percentil is computed only over values greater than zero
    % sometimes signals are zero when the amplifier was disconnected for
    % example, if we don't exclude zeros 10th percentile will also be zero
    PEMG = PEMG-prctile(PEMG(PEMG>0.001),10);
    % smooth it with a median filter
    PEMG = medfilt1(PEMG,5);
    % divide by 90th percentile and then by maximum to keep data in -1 1 range
    PEMG = PEMG./mean(PEMG(1:10));
    PEMG(PEMG>1) = 1;
    PEMG(PEMG<-1) = -1;
    %PEMG = PEMG./max(PEMG);
    Data.PEMG = PEMG;



    %Data.C4A1 = reshape(C4A1, [fs*epochl, maxep]);
    %Data.F3A2 = reshape(C3A2, [fs*epochl, maxep]);
    %Data.F4A1 = reshape(C4A1, [fs*epochl, maxep]);
    %Data.O1A2 = reshape(C3A2, [fs*epochl, maxep]);
    %Data.O2A1 = reshape(C4A1, [fs*epochl, maxep]);
    Data.LOC = reshape(sLOC, [Data.fs*epochl, maxep]);
    Data.ROC = reshape(sROC, [Data.fs*epochl, maxep]);
    Data.EMG = reshape(sEMG, [Data.fs*epochl, maxep]);

    % number of samples in the signal
    Data.max_sample  = numel(C3A2) ;
    % number of epochs
    Data.maxep = maxep;


    % here we plot the spectrogram and EMG power
    % pictures are stored to writePath/../img/' by default

    if ( ~exist([writePath '/../img/'  ], 'dir') ) 
        mkdir([writePath '/../img/'  ]);
    end
    %=================================================
      
    clim=[-20 30]; 
    faxis=0:1/5:Data.fs/2;  % frequency [Hz]
    eaxis=1:maxep;
    taxis=(eaxis-1)*epochl/3600; % time [h]
    if plotspectra
        FigHandle = figure;
        subplot(211)
        imagesc(taxis,faxis,10*log10(Data.Pspec),clim);
        axis xy;
        axis([taxis(1) taxis(end) 0 40])
        ylabel('frequency [Hz]')

        subplot(212)
        bar(taxis,PEMG)

        print(gcf, '-dtiff', '-painters', [writePath '/../img/' recording  '.tiff']);

    end

    % save data         
    save([writePath recording '.mat'], 'Data') 
end

   
