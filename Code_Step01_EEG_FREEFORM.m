% This code generates the RAW and FTA features for provided dataName and provided window.
% Last modified by Bhoj Raj Thapa on 08/16/2023
% Close all figures, clear workspace, and clear command window 
close all; clear; clc; 

%% Global Variables
% Define the path where the data is stored
dataPath = 'D:\AAAI\data\NatureData-FreeForm';
cellfun(@(x) fprintf('dataName: %s\n', x), {dir(fullfile(dataPath, '*.mat')).name}); %Printing the data files
% Define the path where results will be save
resultPath = 'D:\AAAI\results';
if ~exist(resultPath, 'dir'), mkdir(resultPath); end
% Define the name of the data file to be loaded
dataName = 'FREEFORMSubjectC1512102StLRHand';
disp(['dataName: ' dataName '.mat is selected.'])
%Selecting the time window: premovement is [-0.85s,0s] and movement is [0s,0.85s]
tstart = -0.85; % Define start time to extract EEG. Relative to the start of a trial.
tend = 0; % Define end time to extract EEG. Relative to the start of trial.

codePath = pwd; %Script's directory
% Update resultPath to include the 'dataName' directory
resultPath = fullfile(resultPath,dataName);
% Check if a folder with the name 'dataName' exists in the results path
% If it doesn't exist, create it
if ~exist(resultPath, 'dir'), mkdir(resultPath); end

% Define a folder name based on the time window used for EEG extraction
resultFolderName = ['Data_from' num2str(tstart) '_to' num2str(tend)];
% Update resultPath to include the 'resultFolderName' directory
resultPath = fullfile(resultPath,resultFolderName);
% Check if the folder named 'resultFolderName' exists in the current result path
% If it doesn't exist, create it
if ~exist(resultPath, 'dir'), mkdir(resultPath); end
%% Loading the data information
load([dataPath '\' dataName '.mat']); % Load the data file corresponding to the subject
marker = o.marker; % Marker array
RawEEG = o.data; % EEG data (22 Channels)
ChannelNames = o.chnames; % Channel Names
fs = o.sampFreq; % Sampling frequency
[tlength, nch] = size(RawEEG); % Number of channels
nclass = 3; %Number of classes
%%
%%%%%%%%%%%%%%%%%%%%%%%%%% CHANNEL_DECODIFICATION_3_ CLASSES %%%%%%%%%%%%%%%%%%%%%%%%%%%
j = 1;
k = 1;
l = 1;
m = 1;
%Find beginnings of trials and store them in index vectors
for it = 1:tlength-1
    if (marker(it) == 0) && (marker(it+1) == 1) % Class1 = Left Hand
        indexC1(j,1) = it+1;
        j=j+1;
    elseif (marker(it) == 0) && (marker(it+1) == 2) % Class2 = Right Hand
        indexC2(k,1) = it+1;
        k=k+1;
    elseif (marker(it) == 0) && (marker(it+1) == 3) % Class3 = Neutral
        indexC3(l,1) = it+1;
        l=l+1;
    else
        indexND(m,1) = it; % Nothing displayed to validate total trial numbers
        m=m+1;
    end
end

% Validating the trial extraction based on the marker signal
h = figure;
subplot(2,1,1)
hold on
plot(marker,'k')
plot(indexC1,1,'r*')
plot(indexC2,2,'g*')

xlabel('Time Index')
ylabel('Marker')
title('Single Entire Session')
set(gca,'fontsize', 18);
subplot(2,1,2)
hold on
plot(marker,'k')
plot(indexC1,1,'r*')
plot(indexC2,2,'g*')

xlim([1*10^5 1.15*10^5])
xlabel('Time Index')
ylabel('Marker')
title('Selected Time Interval (Zoomed In)')
set(gca,'fontsize', 18);

saveas(h,fullfile(resultPath,'Fig_TrialExtraction.fig'));
saveas(h,fullfile(resultPath,'Fig_TrialExtraction.tif'));
disp('Trial extraction figures are saved.')
%%
%%%%%%%%%%%%%%%%%%%%% DATA SEGMENTATION %%%%%%%%%%%%%%%%%%%%%%%%%

ntrial = length(indexC1)+length(indexC2); % Number of trials
classID = [ones(size(indexC1)).*1; ones(size(indexC2)).*2];
classTrialIndex = [indexC1; indexC2];
trialEEG = zeros(abs(tend-tstart)*fs,nch,ntrial);

for itr = 1: ntrial
    trialEEG(:,:,itr) = RawEEG(classTrialIndex(itr)+tstart*fs:classTrialIndex(itr)+tend*fs-1,:);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Feature1: RAW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntpoints = size(trialEEG,1);
RAWfeature = nan(ntrial,ntpoints*(nch-1));
for itr = 1:ntrial
    for ich = 1: nch-1
        RAWfeature(itr,ntpoints*(ich-1)+1:ntpoints*ich) = trialEEG(:,ich,itr);
    end
end
disp('RAW features are extracted.')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Feature2: Cartesian FTA, REAL & IMAGINARY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Set range of frequency components to be used in feature
fmin = 0;
fmax = 5;

% Validation of the inputs
if(fmin > fmax)
    disp("WARNING: fmin > fmax in the defined frequency range.");
    return;
end

trialEEGfft = fft(trialEEG); % FFT of the segmented EEG
lFFT = size(trialEEGfft, 1); % length of FFTf

%Find frequency axis for one-sided FFT from sampling frequency and
%two-sided FFT length
if(mod(lFFT,2) == 0)
    frequencyAxis = 0:fs/lFFT:fs/2;

    %In the case that the fft has an even number of points (zero is not
    %dead center of shifted fft) turn two sided fft into one sided by including all
    %elements from the DC component to the only non-repeated frequency.
    %Double elements from one to positiveFreqLast
    fft1side = [trialEEGfft(1,:,:); 2*trialEEGfft(2:lFFT/2,:,:); ...
        trialEEGfft(lFFT/2+1,:,:)];
else
    frequencyAxis = 0:fs/(lFFT-1):fs/2;

    %Otherwise, all components but zero (DC) are doubled, since
    %fft is symmetric w.r.t the magnitude frequencyAxis

    fft1side = [trialEEGfft(1,:,:); 2*trialEEGfft(2:(lFFT-1)/2+1,:,:)];
end

nFTApoints = length(frequencyAxis);

% find number of frequency components with respect to fmin and fmax
indexfmintemp = find(frequencyAxis>=fmin);
indexfmin = indexfmintemp(1);
indexfmaxtemp = find(frequencyAxis>fmax);
indexfmax = indexfmaxtemp(1) - 1;

nFreqComp = length(indexfmin:indexfmax);

%Don't include imaginary DC (0) component in features
if(fmin > 0)
    nFTAfeature = 2 * nFreqComp;
else
    nFTAfeature = 2 * nFreqComp - 1;
end


trialEEGfftReal1side = real(fft1side);
trialEEGfftImaginary1side = imag(fft1side);

FTAfeature = nan(ntrial,(nch-1)*nFTAfeature);
for itr = 1:ntrial     
    for ich = 1: nch-1
        featuretemp = zeros(nFTAfeature, 1);
    
        if fmin == 0
            featuretemp(1,1) = trialEEGfftReal1side(indexfmin,ich,itr);
            featuretemp(2:2:end,1) = trialEEGfftReal1side(indexfmin+1:indexfmax,ich,itr);
            featuretemp(3:2:end,1) = trialEEGfftImaginary1side (indexfmin+1:indexfmax,ich,itr);
        else 
            featuretemp(1:2:end,1) = trialEEGfftReal1side(indexfmin:indexfmax,ich,itr);
            featuretemp(2:2:end,1) = trialEEGfftImaginary1side (indexfmin:indexfmax,ich,itr);
        end
        FTAfeature(itr,nFTAfeature*(ich-1)+1:nFTAfeature*ich) = featuretemp; 
    end
end

disp('FTA features are extracted.')
%% Saving RAW and FTA Features
save(fullfile(resultPath,'Data_EEGfeatureRAW.mat'),'RAWfeature','classID')
save(fullfile(resultPath,'Data_EEGfeatureFTA.mat'),'FTAfeature','classID')
disp(['RAW and FTA featurs are saved in ' resultPath ' .'])
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plotting Features: 1) ERP and 2) Cartesian FTA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ChToPlot = "C3"; % Channel name to plot

ChToPlotIndex = find(ChToPlot == ChannelNames);
if isempty(ChToPlotIndex)
    fprintf('=====================ERROR========================\n')
    fprintf('The entered channel to plot does not exist or is wrong\n')
else
    % Plotting RAW EEG
    trialEEGToPlot = squeeze(trialEEG(:,ChToPlotIndex,:));

    h = figure;
    tplot = tstart:1/fs:tend;
    tplot = tplot(1:end-1)';
    hold on;
    plot(tplot,mean(trialEEGToPlot(:,classID==1),2),'r','LineWidth',2);
    plot(tplot,mean(trialEEGToPlot(:,classID==2),2),'b','LineWidth',2);
    plot(tplot,mean(trialEEGToPlot(:,classID==3),2),'g','LineWidth',2);
    plot(tplot,mean(trialEEGToPlot(:,classID==1),2)+std(trialEEGToPlot(:,classID==1),[],2),':r','LineWidth',1.5);
    plot(tplot,mean(trialEEGToPlot(:,classID==1),2)-std(trialEEGToPlot(:,classID==1),[],2),':r','LineWidth',1.5);
    plot(tplot,mean(trialEEGToPlot(:,classID==2),2)+std(trialEEGToPlot(:,classID==2),[],2),':b','LineWidth',1.5);
    plot(tplot,mean(trialEEGToPlot(:,classID==2),2)-std(trialEEGToPlot(:,classID==2),[],2),':b','LineWidth',1.5);
    plot(tplot,mean(trialEEGToPlot(:,classID==3),2)+std(trialEEGToPlot(:,classID==3),[],2),':g','LineWidth',1.5);
    plot(tplot,mean(trialEEGToPlot(:,classID==3),2)-std(trialEEGToPlot(:,classID==3),[],2),':g','LineWidth',1.5);
    grid on;
    xlim([tstart tend]);
    ylim([-7 7])
    xlabel('Relative time from a trial start [sec]')
    ylabel('Voltage [uV]')
    legend({'Left Hand','Right Hand','Neutral'},'Location','southeast')
    title([dataName     ', ' convertStringsToChars(ChToPlot)]);
    set(gca,'fontsize', 18);
    
    fileName= ['Fig_', dataName ,'_', convertStringsToChars(ChToPlot) , '_ERP'];
    saveas(h,fullfile(resultPath,[fileName,'.fig']));
    saveas(h,fullfile(resultPath,[fileName,'.tif']));

    
    % Plotting FTA
    FTrealToPlot = squeeze(trialEEGfftReal1side(:,ChToPlotIndex,:));
    FTimaginaryToPlot = squeeze(trialEEGfftImaginary1side(:,ChToPlotIndex,:));
    
    h = figure;
    fplot = frequencyAxis;
    xlimRange = [fplot(1) fplot(nFreqComp)+1];
    ylimRange = [-300 300];
    subplot(2,1,1)
    hold on
    plot(fplot,mean(FTrealToPlot(:,classID==1),2),'r','LineWidth',2);
    plot(fplot,mean(FTrealToPlot(:,classID==2),2),'b','LineWidth',2);
    plot(fplot,mean(FTrealToPlot(:,classID==3),2),'g','LineWidth',2);
    plot(fplot,mean(FTrealToPlot(:,classID==1),2)+std(FTrealToPlot(:,classID==1),[],2),':r','LineWidth',1.5);
    plot(fplot,mean(FTrealToPlot(:,classID==1),2)-std(FTrealToPlot(:,classID==1),[],2),':r','LineWidth',1.5);
    plot(fplot,mean(FTrealToPlot(:,classID==2),2)+std(FTrealToPlot(:,classID==2),[],2),':b','LineWidth',1.5);
    plot(fplot,mean(FTrealToPlot(:,classID==2),2)-std(FTrealToPlot(:,classID==2),[],2),':b','LineWidth',1.5);
    plot(fplot,mean(FTrealToPlot(:,classID==3),2)+std(FTrealToPlot(:,classID==3),[],2),':g','LineWidth',1.5);
    plot(fplot,mean(FTrealToPlot(:,classID==3),2)-std(FTrealToPlot(:,classID==3),[],2),':g','LineWidth',1.5);
    grid on;
    xlim(xlimRange);
    ylim(ylimRange)
    title([dataName     ', ' convertStringsToChars(ChToPlot)]);
    xlabel('Frequency [Hz]')
    ylabel('Real')
    set(gca,'fontsize', 18);
    
    subplot(2,1,2)
    hold on
    plot(fplot,mean(FTimaginaryToPlot(:,classID==1),2),'r','LineWidth',2);
    plot(fplot,mean(FTimaginaryToPlot(:,classID==2),2),'b','LineWidth',2);
    plot(fplot,mean(FTimaginaryToPlot(:,classID==3),2),'g','LineWidth',2);
    plot(fplot,mean(FTimaginaryToPlot(:,classID==1),2)+std(FTimaginaryToPlot(:,classID==1),[],2),':r','LineWidth',1.5);
    plot(fplot,mean(FTimaginaryToPlot(:,classID==1),2)-std(FTimaginaryToPlot(:,classID==1),[],2),':r','LineWidth',1.5);
    plot(fplot,mean(FTimaginaryToPlot(:,classID==2),2)+std(FTimaginaryToPlot(:,classID==2),[],2),':b','LineWidth',1.5);
    plot(fplot,mean(FTimaginaryToPlot(:,classID==2),2)-std(FTimaginaryToPlot(:,classID==2),[],2),':b','LineWidth',1.5);
    plot(fplot,mean(FTimaginaryToPlot(:,classID==3),2)+std(FTimaginaryToPlot(:,classID==3),[],2),':g','LineWidth',1.5);
    plot(fplot,mean(FTimaginaryToPlot(:,classID==3),2)-std(FTimaginaryToPlot(:,classID==3),[],2),':g','LineWidth',1.5);
    grid on;
    xlim(xlimRange);
    ylim(ylimRange)
    xlabel('Frequency [Hz]')
    ylabel('Imaginary')
    set(gca,'fontsize', 18);
        
    fileName= ['Fig_', dataName ,'_', convertStringsToChars(ChToPlot) , '_FTA'];
    saveas(h,fullfile(resultPath,[fileName,'.fig']));
    saveas(h,fullfile(resultPath,[fileName,'.tif']));
end
disp(['Results for ' char(ChToPlot) ' is plotted and saved.'])
%% End of the script
disp('The script executed succefully.')