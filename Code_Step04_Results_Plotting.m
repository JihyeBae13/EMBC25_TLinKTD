%{
  This code generates and saves bar graphs to compare TL/NoTL results for different
  subject, feature (RAW and FTA) and window ([-0.85,0] [0, 0.85])
  combinations. 
  Last updated by Reece McDorman on 7/24/2024
  Last reviewed and updated by Bhoj Raj Thapa on 8/15/2024
%}
clc 
clear
close all
%% Global Variables Definition

% Define the path to the directory where results are stored
parentTLResultPath = 'D:\AAAI\results';
codePath = pwd;
% Define the datasets, features, and time windows to be used
dataNames = ['FREEFORMSubjectB1511112StLRHand';'FREEFORMSubjectC1512082StLRHand'; 'FREEFORMSubjectC1512102StLRHand'];
dataNamesShortform = ["B","C1","C2"]; %Shortforms for the dataNames.
featureNames = ['RAW';'FTA'];
tstarts = [-0.85; 0]; %Start time (in seconds) for premovement and movement window respectively. 
tends = [0; 0.85]; %End time (in seconds) for premovement and movement window respectively. 

% Define the classifier name and number of Monte Carlo runs
classifierName = 'KTD_TL';
nMCrun = 10; %Monte Carlo Runs

%Shorthand: Calculate the number of datasets, features, windows, combinations, and cases
nDataNames = size(dataNames, 1);
nFeatureNames = size(featureNames, 1);
if(size(tstarts,1)~=size(tends,1))
    disp('Each window has to have both start and end time. Please check variables: tstarts and ends.')
end
nWindows = size(tstarts, 1);
nCombinations = nchoosek(3,2) * 2; %Combinations of total dataName with order.
nCases = nCombinations * nWindows * nFeatureNames; % Total number of cases to analyze


%% Tabulate TL results

%TL results for all 24 (nCombinations * nFeatureNames * nWindows = 6*2*2)
%cases are tabulated here; for each case, there is a success rate value after first epoch; and
%also the path the success rate value after first epoch came from (to verify everything is organized correctly)
TLresults = cell(nCombinations, nFeatureNames * nWindows, 2);

%TL combination row in TLresults table we are currently generating
combinationCounter = 1; %Counter for each combination to add a new row.

% Loop through each combination of source and target datasets (rows)
for iSource = 1:nDataNames
    for iTarget = 1:nDataNames
        % Only proceed if source and target are different
        if ~strcmp(dataNames(iSource,:), dataNames(iTarget,:))
            %Loop through each combination of features and windows
            %(columns)
            featureWindowCounter = 0;
            for iFeature = 1:nFeatureNames
                for iWindow = 1:nWindows
                    featureWindowCounter = featureWindowCounter+1;
                    % Generate the folder name based on, source, target, feature, and time window
                    folderName = [classifierName 'training' featureNames(iFeature,:) '_source' ...
                    dataNames(iSource,:) 'from' num2str(tstarts(iWindow)) 'to' num2str(tends(iWindow))];
                    
                    % Construct the path to the results for the target dataset
                    TLresultPath = [parentTLResultPath '\' dataNames(iTarget,:) ...
                    '\Data_from' num2str(tstarts(iWindow)) '_to' num2str(tends(iWindow)) '\' folderName];
                          
                    % Load the results and store the success rate
                    TLresultsfile = load(fullfile(TLresultPath,'ResultsWparms.mat')); %Should load a matrix with size (nMCRun,nEpoch)
                    TLsuccessRate = TLresultsfile.successRate;
                    
                    % Store the path and success rate in the cell array
                    TLresult = cell(1); %Initializing an empty cell
                    TLresult{1} = TLresultPath;
                    TLresult{2} = TLsuccessRate(:, 1);
                    
                    % Put TLresult in one (row, column) of TLresults 
                    TLresults(combinationCounter, featureWindowCounter, :) = TLresult;
                end
            end
            %Go to next row
            combinationCounter = combinationCounter + 1;
        end
    end
end

%% Tabulate NoTL results

% Initialize the cell array to store paths and results without transfer learning (NoTL)
NoTLresults = cell(nDataNames, nFeatureNames * nWindows, 2);

% Loop through each of the three sessions/datasets to load results without
% transfer learning (rows)
for dataName = 1:nDataNames
    %Loop through each combination of features and windows
    %(columns)
    featureWindowCounter = 0;
    for iFeature = 1:nFeatureNames
        for iWindow = 1:nWindows
            featureWindowCounter = featureWindowCounter+1;
            % Construct the path to the results for the dataset without TL
            dataPathTargetNoTL = [parentTLResultPath '\' dataNames(dataName, :) ...
            '\Data_from' num2str(tstarts(iWindow)) '_to' num2str(tends(iWindow)) '\' classifierName 'source' featureNames(iFeature, :)]; 
            
            % Load the results and store the success rate
            NoTLresultsfile = load(fullfile(dataPathTargetNoTL,'ResultsWparms_TLsource.mat'));
            NoTLsuccessRate = NoTLresultsfile.successRate;
            
            % Store the path and success rate in the cell array
            NoTLresult = cell(1); %Initializing an empty cell
            NoTLresult{1} = dataPathTargetNoTL;
            NoTLresult{2} = NoTLsuccessRate(:, 1);
            
            % Put the NoTL result in one (row, column) of TLresults
            NoTLresults(dataName, featureWindowCounter, :) = NoTLresult;
        end
    end
end

%% Organize data for plotting
%Initializing variables to store statistics of RAW Features
meanDataToPlotRaw = zeros(nCombinations,nWindows*2); %Stores mean of nMCRun for first epoch.
stdDataToPlotRaw = zeros(nCombinations,nWindows*2); %Stores sandard deviation of nMCRun for first epoch.
ttestResultsRaw01 = zeros(nCombinations,nWindows); %Stores independent t-test result of nMCRun for first epoch.

%Initializing variables to store statistics of FTA Features
meanDataToPlotFta = zeros(nCombinations,nWindows*2); %Stores mean of nMCRun for first epoch.
stdDataToPlotFta = zeros(nCombinations,nWindows*2); %Stores sandard deviation of nMCRun for first epoch.
ttestResultsFta01 = zeros(nCombinations,nWindows); %Stores independent t-test result of nMCRun for first epoch.

expectedSourceSequence = ["B","B","C1","C1","C2","C2"];
expectedTargetSequence = ["C1","C2","B","C2","B","C1"];

%Make sure source and target sequences have matching size
if(length(expectedSourceSequence)==length(expectedTargetSequence)) 
    %Go through source/target combinations to fill data to plot
    for iCombination = 1:length(expectedSourceSequence)
        %Find data names for the current source/target combination
        %corresponding to the shorthands
        sourceDataName = dataNames(strcmp(expectedSourceSequence(iCombination),dataNamesShortform),:);
        targetDataName = dataNames(strcmp(expectedTargetSequence(iCombination),dataNamesShortform),:);
        
        %Find row in TL table corresponding to source sourceDataName and
        %target targetDataName (first column is arbitrarily chosen for searching)
        rawFullPathTL = TLresults(:,1,1); 

        for iPath = 1:length(rawFullPathTL)
            splittedPath = split(rawFullPathTL{iPath},'\');
            %Note: may have to change this line depending on folder names
            %used
            resultsPosition = find(strcmp(splittedPath,'results'));
            %Find target in selected path (row)
            targetFromPath = splittedPath{resultsPosition+1};
            if(targetFromPath==targetDataName)
                %Find source in selected path
                sourceFromPath = splittedPath{resultsPosition+3};
                %If they are equal, we have found the correct row in
                %TLresults
                if(contains(sourceFromPath,sourceDataName))
                    requiredTLIndex = iPath;
                    break;
                end
            end
        end

        %Find row in NoTL table corresponding to targetDataName (first column arbitrarily chosen for searching)
        rawFullPathNoTL = NoTLresults(:,1,1);
        for iPath = 1:length(rawFullPathNoTL)
            splittedPathNoTL = split(rawFullPathNoTL{iPath},'\');
            %Note: may have to change this line depending on folder names
            %used
            resultsPositionNoTL = find(strcmp(splittedPathNoTL,'results'));
            %If they are equal, we have found the correct row in
            %NoTLresults
            if(targetDataName == splittedPathNoTL{resultsPositionNoTL+1})
                requiredNoTLIndex = iPath;
                break;
            end
        end

        %RAW Feature Statistics 
        
        %Extract raw data from TLresults and NoTLresults row found above 
        preMovWithTLRaw = cell2mat(TLresults(requiredTLIndex,1,2)); 
        postMovWithTLRaw = cell2mat(TLresults(requiredTLIndex,2,2));
        preMovWithNoTLRaw = cell2mat(NoTLresults(requiredNoTLIndex,1,2)); 
        postMovWithNoTLRaw = cell2mat(NoTLresults(requiredNoTLIndex,2,2));

        %Calculate statistics and add them to corresponding tables
        meanDataToPlotRaw(iCombination,:) = [mean(preMovWithTLRaw),mean(preMovWithNoTLRaw),mean(postMovWithTLRaw),mean(postMovWithNoTLRaw)];
        stdDataToPlotRaw(iCombination,:) = [std(preMovWithTLRaw),std(preMovWithNoTLRaw),std(postMovWithTLRaw),std(postMovWithNoTLRaw)];
        ttestResultsRaw01(iCombination,:) = [ttest2(preMovWithTLRaw,preMovWithNoTLRaw,'Alpha',0.01),ttest2(postMovWithTLRaw,postMovWithNoTLRaw,'Alpha',0.01)];

        % FTA Feature Statistics

        %Extract fta data from TLresults and NoTLresults row found above
        preMovWithTLFta = cell2mat(TLresults(requiredTLIndex,3,2));
        postMovWithTLFta = cell2mat(TLresults(requiredTLIndex,4,2));
        preMovWithNoTLFta = cell2mat(NoTLresults(requiredNoTLIndex,3,2));
        postMovWithNoTLFta = cell2mat(NoTLresults(requiredNoTLIndex,4,2));

        %Calculate statistics and add them to corresponding tables
        meanDataToPlotFta(iCombination,:) = [mean(preMovWithTLFta),mean(preMovWithNoTLFta),mean(postMovWithTLFta),mean(postMovWithNoTLFta)];
        stdDataToPlotFta(iCombination,:) = [std(preMovWithTLFta),std(preMovWithNoTLFta),std(postMovWithTLFta),std(postMovWithNoTLFta)];
        ttestResultsFta01(iCombination,:) = [ttest2(preMovWithTLFta,preMovWithNoTLFta,'Alpha',0.01),ttest2(postMovWithTLFta,postMovWithNoTLFta,'Alpha',0.01)];
    end
end


% Reshape results for raw and fta error bar plots and asterisk (*)
% significance marks 
% Note: nCombinations * nWindows * 2 = 6*2*2 = 24 datapoints (*2 because we are
% including one noTL bar for every TL bar) to be plotted. nCombinations *
% nWindows = 6*2 = 12 positions for significance marks. 
meanDataToPlotRaw1D = reshape(meanDataToPlotRaw', [1, nCombinations * nWindows * 2]);
ttestResultsRaw1D = reshape(ttestResultsRaw01', [1, nCombinations * nWindows]);
stdDataToPlotRaw1D = reshape(stdDataToPlotRaw', [1,nCombinations  * nWindows * 2]);

meanDataToPlotFta1D = reshape(meanDataToPlotFta', [1, nCombinations * nWindows * 2]);
ttestResultsFta1D = reshape(ttestResultsFta01', [1, nCombinations * nWindows]);
stdDataToPlotFta1D = reshape(stdDataToPlotFta', [1,nCombinations * nWindows * 2]);

%% Determine x-values for error bars and asterisk (*) significance marks
errorBarPosX = nan(1, nCases / 2);
for icase = 1:nCases/2
    errorBarPosX(icase) = floor((icase - 1) / (nFeatureNames)) + mod(icase - 1, nFeatureNames) / 3.5 + 0.86;
end
errorBarPosXTL = errorBarPosX(1:2:end);

%Define indices to seperate premovement and movement results
preSelectionIndices = [ones(6, 2) zeros(6, 2)];
preSelectionIndices1D = reshape(preSelectionIndices', [1, 24]);
postSelectionIndices = [zeros(6, 2) ones(6, 2)];
postSelectionIndices1D = reshape(postSelectionIndices', [1, 24]);

%% Common specifications for bar graphs
blackBarGraphColor = '[0.4,0.4,0.4]'; % The color of bar graph for transfer learning
lightBarGraphColor = '[0.9,0.9,0.9]'; % The color of bar graph for transfer learning
asteriskGap = 0.14; % Adjust this for horizontal positioning
yOffsetPremovement = 2; % Vertical offset for asterisks
yOffsetMovement = 2; % Vertical offset for asterisks
angleForXTick = 45;
barHeight = 6;
barWidth = 6.2;
%% Plot RAW data
%% RAW, Premovement
h11 = figure('Units', 'inches', 'Position', [2 3 barWidth barHeight],'Name','RAW_Premovement');

ax = gca;
ax.XTick = ax.XTick * 0.5;

%Select premovement data from results
meanDataToPlotRawPre = reshape(meanDataToPlotRaw(preSelectionIndices==1),[6,2]);
meanDataToPlotRawPre1D = meanDataToPlotRaw1D(preSelectionIndices1D==1);
stdDataToPlotRawPre1D = stdDataToPlotRaw1D(preSelectionIndices1D==1);
ttestResultsRawPre1D = ttestResultsRaw1D(preSelectionIndices1D(1:2:end)==1);

b = bar(meanDataToPlotRawPre * 100);
b(1).FaceColor = blackBarGraphColor;
b(2).FaceColor = lightBarGraphColor;
% Get bar positions
xPos = b.XData; % X positions of the bars
yPos = b.YData; % Heights of the bars
xPosSig = xPos(ttestResultsRawPre1D==1);
yPosSig = yPos(ttestResultsRawPre1D==1);

hold on;

% Plot error bars
errorbar(errorBarPosX, meanDataToPlotRawPre1D * 100, stdDataToPlotRawPre1D * 100, 'Color', 'black', 'LineStyle', 'none', 'lineWidth', 1.5, 'CapSize', 10);
ylim([0, 100]);
set(gca, 'FontSize', 20);
set(gca, 'XTickLabelRotation', 45);
xticklabels({strcat(expectedSourceSequence(1),'\rightarrow',expectedTargetSequence(1)), strcat(expectedSourceSequence(2),'\rightarrow',expectedTargetSequence(2)) ...
 strcat(expectedSourceSequence(3),'\rightarrow',expectedTargetSequence(3)), strcat(expectedSourceSequence(4),'\rightarrow',expectedTargetSequence(4)) ...
 strcat(expectedSourceSequence(5),'\rightarrow',expectedTargetSequence(5)), strcat(expectedSourceSequence(6),'\rightarrow',expectedTargetSequence(6))});
xtickangle(angleForXTick)

l=legend({'with TL', 'without TL'});
fontsize(l, 20, 'points')
ylabel('Success Rate (%)');

% Plot statistically significant marks (asterisks or *)
% text(errorBarPosXTL - asteriskGap, meanDataToPlotRawPre1D(1:2:end)*100 + 4 .* ttestResultsRawPre1D- ~ttestResultsRawPre1D * 1000, '*', 'Interpreter', 'latex', 'FontSize', 20);
text(xPosSig - asteriskGap, yPosSig + yOffsetPremovement, '*', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'Interpreter', 'latex', 'FontSize', 20);
grid on;
%title("RAW");

%% RAW, Movement
h12 = figure('Units', 'inches', 'Position', [2 3 barWidth barHeight],'Name','RAW_Movement');

%Select movement data from results
meanDataToPlotRawPost = reshape(meanDataToPlotRaw(postSelectionIndices==1),[6,2]);
meanDataToPlotRawPost1D = meanDataToPlotRaw1D(postSelectionIndices1D==1);
stdDataToPlotRawPost1D = stdDataToPlotRaw1D(postSelectionIndices1D==1);
ttestResultsRawPost1D = ttestResultsRaw1D(postSelectionIndices1D(1:2:end)==1);

b = bar(meanDataToPlotRawPost * 100);
b(1).FaceColor = blackBarGraphColor;
b(2).FaceColor = lightBarGraphColor;
% Get bar positions
xPos = b.XData; % X positions of the bars
yPos = b.YData; % Heights of the bars
xPosSig = xPos(ttestResultsRawPost1D==1);
yPosSig = yPos(ttestResultsRawPost1D==1);

hold on;

% Plot error bars
errorbar(errorBarPosX, meanDataToPlotRawPost1D * 100, stdDataToPlotRawPost1D * 100, 'Color', 'black', 'LineStyle', 'none', 'lineWidth', 1.5, 'CapSize', 10);
ylim([0, 100]);
set(gca, 'FontSize', 20);
set(gca, 'XTickLabelRotation', 45);
xticklabels({strcat(expectedSourceSequence(1),'\rightarrow',expectedTargetSequence(1)), strcat(expectedSourceSequence(2),'\rightarrow',expectedTargetSequence(2)) ...
 strcat(expectedSourceSequence(3),'\rightarrow',expectedTargetSequence(3)), strcat(expectedSourceSequence(4),'\rightarrow',expectedTargetSequence(4)) ...
 strcat(expectedSourceSequence(5),'\rightarrow',expectedTargetSequence(5)), strcat(expectedSourceSequence(6),'\rightarrow',expectedTargetSequence(6))});
xtickangle(angleForXTick)
legend({'with TL', 'without TL'});
ylabel('Success Rate (%)');

% Plot statistically significant marks (asterisks or *)
text(xPosSig - asteriskGap, yPosSig + yOffsetMovement, '*', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'Interpreter', 'latex', 'FontSize', 20);
grid on;
%title("RAW");

%% Plot FTA data

%% FTA, Premovement

h21 = figure('Units', 'inches', 'Position', [2 3 barWidth barHeight],'Name','FTA_Premovement');

%Select premovement data from results
meanDataToPlotFtaPre = reshape(meanDataToPlotFta(preSelectionIndices==1),[6,2]);
meanDataToPlotFtaPre1D = meanDataToPlotFta1D(preSelectionIndices1D==1);
stdDataToPlotFtaPre1D = stdDataToPlotFta1D(preSelectionIndices1D==1);
ttestResultsFtaPre1D = ttestResultsFta1D(preSelectionIndices1D(1:2:end)==1);

b = bar(meanDataToPlotFtaPre * 100);
b(1).FaceColor = blackBarGraphColor;
b(2).FaceColor = lightBarGraphColor;
% Get bar positions
xPos = b.XData; % X positions of the bars
yPos = b.YData; % Heights of the bars
xPosSig = xPos(ttestResultsFtaPre1D==1);
yPosSig = yPos(ttestResultsFtaPre1D==1);

hold on;

% Plot error bars
errorbar(errorBarPosX, meanDataToPlotFtaPre1D * 100, stdDataToPlotFtaPre1D * 100, 'Color', 'black', 'LineStyle', 'none', 'lineWidth', 1.5, 'CapSize', 10);
ylim([0, 100]);
set(gca, 'FontSize', 20);
xticklabels({strcat(expectedSourceSequence(1),'\rightarrow',expectedTargetSequence(1)), strcat(expectedSourceSequence(2),'\rightarrow',expectedTargetSequence(2)) ...
 strcat(expectedSourceSequence(3),'\rightarrow',expectedTargetSequence(3)), strcat(expectedSourceSequence(4),'\rightarrow',expectedTargetSequence(4)) ...
 strcat(expectedSourceSequence(5),'\rightarrow',expectedTargetSequence(5)), strcat(expectedSourceSequence(6),'\rightarrow',expectedTargetSequence(6))});
l = legend({'with TL', 'without TL'});
xtickangle(angleForXTick)
fontsize(l, 20, 'points');
ylabel('Success Rate (%)');

% Plot statistically significant marks (asterisks or *)
text(xPosSig - asteriskGap, yPosSig + yOffsetPremovement, '*', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'Interpreter', 'latex', 'FontSize', 20);
grid on;
%title("FTA");

%% FTA, Movement
h22 = figure('Units', 'inches', 'Position', [2 3 barWidth barHeight],'Name','FTA_Movement');

%Select movement data from results
meanDataToPlotFtaPost = reshape(meanDataToPlotFta(postSelectionIndices==1),[6,2]);
meanDataToPlotFtaPost1D = meanDataToPlotFta1D(postSelectionIndices1D==1);
stdDataToPlotFtaPost1D = stdDataToPlotFta1D(postSelectionIndices1D==1);
ttestResultsFtaPost1D = ttestResultsFta1D(postSelectionIndices1D(1:2:end)==1);

b = bar(meanDataToPlotFtaPost * 100);
b(1).FaceColor = blackBarGraphColor;
b(2).FaceColor = lightBarGraphColor;
% Get bar positions
xPos = b.XData; % X positions of the bars
yPos = b.YData; % Heights of the bars
xPosSig = xPos(ttestResultsFtaPost1D==1);
yPosSig = yPos(ttestResultsFtaPost1D==1);

hold on;

% Plot error bars
errorbar(errorBarPosX, meanDataToPlotFtaPost1D * 100, stdDataToPlotFtaPost1D * 100, 'Color', 'black', 'LineStyle', 'none', 'lineWidth', 1.5, 'CapSize', 10);
ylim([0, 100]);
set(gca, 'FontSize', 20);
xticklabels({strcat(expectedSourceSequence(1),'\rightarrow',expectedTargetSequence(1)), strcat(expectedSourceSequence(2),'\rightarrow',expectedTargetSequence(2)) ...
 strcat(expectedSourceSequence(3),'\rightarrow',expectedTargetSequence(3)), strcat(expectedSourceSequence(4),'\rightarrow',expectedTargetSequence(4)) ...
 strcat(expectedSourceSequence(5),'\rightarrow',expectedTargetSequence(5)), strcat(expectedSourceSequence(6),'\rightarrow',expectedTargetSequence(6))});
xtickangle(angleForXTick)
legend({'with TL', 'without TL'});
ylabel('Success Rate (%)');

% Plot statistically significant marks (asterisks or *)
text(xPosSig - asteriskGap, yPosSig + yOffsetMovement, '*', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'Interpreter', 'latex', 'FontSize', 20);
grid on;
%title("FTA");

%% Save generated figures
graphResultPath = [parentTLResultPath,'\','graphTLResults'];
% Check if the directory specified by 'graphResultPath' exists
% The 'exist' function returns 7 if the directory exists, otherwise it returns a different value
if exist(graphResultPath, 'dir') ~= 7
    mkdir(graphResultPath);
end
% Save the figure 'h11' in two formats (TIFF and FIG) with the name 'Fig_ResultsRAW_Premovement'
saveas(h11, [graphResultPath,'\','Fig_ResultsRAW_Premovement.tif']);
saveas(h11, [graphResultPath,'\','Fig_ResultsRAW_Premovement.fig']);
% Save the figure 'h12' in two formats (TIFF and FIG) with the name 'Fig_ResultsRAW_Movement'
saveas(h12, [graphResultPath,'\','Fig_ResultsRAW_Movement.tif']);
saveas(h12, [graphResultPath,'\','Fig_ResultsRAW_Movement.fig']);
% Save the figure 'h21' in two formats (TIFF and FIG) with the name 'Fig_ResultsFTA_Premovement'
saveas(h21, [graphResultPath,'\','Fig_ResultsFTA_Premovement.tif']);
saveas(h21, [graphResultPath,'\','Fig_ResultsFTA_Premovement.fig']);
% Save the figure 'h22' in two formats (TIFF and FIG) with the name 'Fig_ResultsFTA_Movement'
saveas(h22, [graphResultPath,'\','Fig_ResultsFTA_Movement.tif']);
saveas(h22, [graphResultPath,'\','Fig_ResultsFTA_Movement.fig']);
%% End of Code
disp('The script executed succefully.')