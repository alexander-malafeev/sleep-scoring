%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all; fclose all;

% this script takes the predicted stages from ./pred/ folder and 
% plots hypnogram and spectrogram for each recording
% they will be saved into the plot folder along with 
% the two text files. First contains sleep efficiency and proportions
% of stages. Second contains sleep stages.

data_dir = './pred/'
addpath ./../../tools/

epochl = 20;
fs = 128;

fileList = dir([data_dir '*.mat']);


for f = 1:length(fileList)
    recording  = fileList(f).name

    fname = strtrim( recording )
    
    load([data_dir fname])
    
    maxep = numel(y_);

    stages_ = int32(y_);
    stages = int32(y);
 
    prob = y_p;
    p = prob;
    p(:,2) = prob(:,5);
    p(:,3) = prob(:,2);
    p(:,4:5) = prob(:,3:4);

    Pspec = Pspec';

    % =============== Plot Spectra =====================

    clim=[-20 30]; 
   
    faxis=0:1/5:fs/2; % frequency axis in Hz
    eaxis=1:maxep;
    taxis=(eaxis-1)*epochl/3600; %time axis in hours
 
    FigHandle = figure('Position', [100, 100, 1049, 895]);        
        
    set(gca, 'XTickLabel', []);
    ylabel('Sleep stage');
    title( fname,'Interpreter','none' )

    ax2 = subplot(2, 1, 1);
    plot(eaxis,stages_)
    ylim([0 4 ]);
    ylabel('Sleep stage');
    yticks([0 1 2 3 4])
    yticklabels( [{'W'} {'1'} {'2'} {'3'} {'R'} ])
    
    ax3 = subplot(2, 1, 2);
    imagesc(eaxis,faxis,10*log10(Pspec'),clim);
    axis xy;
    axis([eaxis(1) eaxis(end) 0 40])
    ylabel('Frequency [Hz]')
    xlabel('time [h]');
    % without this option spectrogram will disappear
    set(gcf,'render','painters')
    colormap(ax3,'jet');
    print(gcf, '-dtiff', '-r100', ['./plot/' fname '.tiff'])

    % compute sleep efficiency and save it to the text file
    stages = stagesNum2Sym(stages_);
    num_W = numel(find(stages=='W'))./numel(stages);
    num_R = numel(find(stages=='R'))./numel(stages);
    num_12 = numel( union( find(stages=='1'), find(stages=='2') ))./numel(stages);
    num_3 =  numel(find(stages=='3'))./numel(stages);

    sleep_efficiency = num_R+num_12+num_3;

    report_fname = [fname '.txt'];
    fileID = fopen(['./plot/'   report_fname] ,'wt');
    fprintf(fileID,'%s %f\n\n','sleep efficiency = ',sleep_efficiency);
    fprintf(fileID,'%s %8s\n','stage','amount');
    fprintf(fileID,'%3s %8.2f\n','W',num_W );
    fprintf(fileID,'%3s %8.2f\n','R',num_R );
    fprintf(fileID,'%3s %8.2f\n','1-2',num_12 );
    fprintf(fileID,'%3s %8.2f\n','3',num_3 );
    fclose(fileID);


    report_fname_st = [ fname '_stages.txt'];
    fileID = fopen(['./plot/'  report_fname_st] ,'wt');
    fprintf(fileID,'%s',stages);
    fclose(fileID);

end
      

