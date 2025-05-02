% Data Loading
filename = 'Churn_Modelling.csv';
data = readtable(filename);

% Data Preprocessing
data.Geography = grp2idx(data.Geography);
data.Gender = grp2idx(data.Gender);

% Selectinon relevant features and target variable
features = data(:, [4, 6:13]);
target = data.Exited;

% Splitting the dataset into training and testing sets
cv = cvpartition(height(data), 'HoldOut', 0.2);
idx = cv.test;

% Training set
XTrain = features(~idx, :);
YTrain = target(~idx, :);

% Convert feature table to array 
if istable(XTrain)
    XTrain = table2array(XTrain);
end

% Testing set
XTest = features(idx, :);
YTest = target(idx, :);
% Convert feature table to array if necessary
if istable(XTest)
    XTest = table2array(XTest);
end

% SMOTE for Imbalanced Data
rng(0); % Set a seed for reproducibility
minorityClass = 1;
k = 5; % Number of nearest neighbors for SMOTE
minorityIndices = find(YTrain == minorityClass);
minoritySamples = XTrain(minorityIndices, :);

% Synthetic Sample Generation with SMOTE
syntheticSamples = [];
for i = 1:size(minoritySamples, 1)
    distances = sum((XTrain - minoritySamples(i,:)).^2, 2);
    [~, sortedIndices] = sort(distances);
    neighborIndices = sortedIndices(2:k+1);

    for j = 1:k
        neighborSample = XTrain(neighborIndices(j), :);
        diff = neighborSample - minoritySamples(i, :);
        gap = rand(1);
        syntheticSample = minoritySamples(i, :) + gap * diff;
        syntheticSamples = [syntheticSamples; syntheticSample];
    end
end

% Augment Training Data with synthetic samples
augmentedXTrain = [XTrain; syntheticSamples];
augmentedYTrain = [YTrain; ones(size(syntheticSamples, 1), 1)];

% Model Training Naive Bayes with SMOTE-Augmented Data
naiveBayesModel = fitcnb(augmentedXTrain, augmentedYTrain, 'DistributionNames', 'normal');

% Model Evaluation
YPredOptimal = predict(naiveBayesModel, XTest);

% Evaluation Metrics
confMatOptimal = confusionmat(YTest, YPredOptimal);
TP = confMatOptimal(1,1);
FN = confMatOptimal(1,2);
FP = confMatOptimal(2,1);
TN = confMatOptimal(2,2);

% Calculation and Display of Metrics
accuracyOptimal = sum(YPredOptimal == YTest) / length(YTest);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('NB Accuracy: %.2f%%\n', accuracyOptimal * 100);
fprintf('NB Precision: %.2f\n', precision);
fprintf('NB Recall: %.2f\n', recall);
fprintf('NB F1-Score: %.2f\n', f1_score);

% ROC-AUC Calculation
[~, scores] = predict(naiveBayesModel, XTest);
[Xroc, Yroc, ~, AUC] = perfcurve(YTest, scores(:,2), 1);






% Data Loading
filename = 'Churn_Modelling.csv';
data = readtable(filename);

% Data Preprocessing
data.Geography = grp2idx(data.Geography);
data.Gender = grp2idx(data.Gender);

% Selectinon relevant features and target variable
features = data(:, [4, 6:13]);
target = data.Exited;

% Splitting the dataset into training and testing sets
cv = cvpartition(height(data), 'HoldOut', 0.2);
idx = cv.test;

% Training set
XTrain = features(~idx, :);
YTrain = target(~idx, :);

% Convert table to array
if istable(XTrain)
    XTrain = table2array(XTrain);
end
if istable(YTrain)
    YTrain = table2array(YTrain);
end

% Testing set
XTest = features(idx, :);
YTest = target(idx, :);
if istable(XTest)
    XTest = table2array(XTest);
end
if istable(YTest)
    YTest = table2array(YTest);
end

% SMOTE for Imbalanced Data
rng(0); % Set a seed for reproducibility
minorityClass = 1;
k = 5; % Number of nearest neighbors for SMOTE
minorityIndices = find(YTrain == minorityClass);
minoritySamples = XTrain(minorityIndices, :);

% Synthetic Sample Generation with SMOTE
syntheticSamples = [];
for i = 1:size(minoritySamples, 1)
    distances = sum((XTrain - minoritySamples(i,:)).^2, 2);
    [~, sortedIndices] = sort(distances);
    neighborIndices = sortedIndices(2:k+1);

    for j = 1:k
        neighborSample = XTrain(neighborIndices(j), :);
        diff = neighborSample - minoritySamples(i, :);
        gap = rand(1);
        syntheticSample = minoritySamples(i, :) + gap * diff;
        syntheticSamples = [syntheticSamples; syntheticSample];
    end
end

% Augment Training Data with synthetic samples
augmentedXTrain = [XTrain; syntheticSamples];
augmentedYTrain = [YTrain; ones(size(syntheticSamples, 1), 1)];

% Splitting augmented training data into training and validation sets
cvInner = cvpartition(size(augmentedXTrain, 1), 'HoldOut', 0.2);
idxInner = cvInner.test;

% Subsets for hyperparameter tuning
XTrainSub = augmentedXTrain(~idxInner, :);
YTrainSub = augmentedYTrain(~idxInner, :);
XValid = augmentedXTrain(idxInner, :);
YValid = augmentedYTrain(idxInner, :);

% Initialization of variables for hyperparameter tuning with validation
optimalMaxNumSplits = 0;
optimalMinLeafSize = 0;
optimalMinParentSize = 0;
highestValidationAccuracy = 0;

% Defining ranges for hyperparameters
maxNumSplitsRange = 1:10; 
minLeafSizeRange = 1:10; 
minParentSizeRange = 10:10:100; 

% Grid search for hyperparameter tuning with validation
for maxNumSplits = maxNumSplitsRange
    for minLeafSize = minLeafSizeRange
        for minParentSize = minParentSizeRange
            % Create and train the model on the training subset
            treeTemplate = templateTree('MaxNumSplits', maxNumSplits, ...
                                        'MinLeafSize', minLeafSize, ...
                                        'MinParentSize', minParentSize);
            model = fitcensemble(XTrainSub, YTrainSub, 'Method', 'Bag', ...
                                 'NumLearningCycles', 100, ...
                                 'Learners', treeTemplate);

            % Evaluation of the model on the validation set
            [~, scores] = predict(model, XValid);
            positiveClassProbabilities = scores(:, 2);
            [~, ~, ~, auc] = perfcurve(YValid, positiveClassProbabilities, 1);

            % Update optimal parameters if current model is better on validation set
            if auc > highestValidationAccuracy
                highestValidationAccuracy = auc;
                optimalMaxNumSplits = maxNumSplits;
                optimalMinLeafSize = minLeafSize;
                optimalMinParentSize = minParentSize;
            end
        end
    end
end

% Trainning the optimal model on the full augmented training data
optimalTreeTemplate = templateTree('MaxNumSplits', optimalMaxNumSplits, ...
                                   'MinLeafSize', optimalMinLeafSize, ...
                                   'MinParentSize', optimalMinParentSize);
optimalModel = fitcensemble(augmentedXTrain, augmentedYTrain, 'Method', 'Bag', ...
                            'NumLearningCycles', 100, ...
                            'Learners', optimalTreeTemplate);

% Evaluation the optimal model on the test set
[~, scoresOptimal] = predict(optimalModel, XTest);
positiveClassProbabilitiesOptimal = scoresOptimal(:, 2);
[fpr, tpr, ~, aucOptimal] = perfcurve(YTest, positiveClassProbabilitiesOptimal, 1);

% Confusion matrix
YPredOptimal = predict(optimalModel, XTest);
confMatOptimal = confusionmat(YTest, YPredOptimal);

% Extractinon confusion matrix components
TP = confMatOptimal(1,1);
FN = confMatOptimal(1,2);
FP = confMatOptimal(2,1);
TN = confMatOptimal(2,2);

% Calculating metrics: accuracy, precision, recall, and F1-score
accuracyOptimal = sum(YPredOptimal == YTest) / length(YTest);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

% Displaying the results
fprintf('DTE Optimal MaxNumSplits: %d\n', optimalMaxNumSplits);
fprintf('DTE Optimal MinLeafSize: %d\n', optimalMinLeafSize);
fprintf('DTE Optimal MinParentSize: %d\n', optimalMinParentSize);
fprintf('DTE Accuracy of Optimal Decision Tree Model: %.2f%%\n', accuracyOptimal * 100);
fprintf('DTE Precision: %.2f\n', precision);
fprintf('DTE Recall: %.2f\n', recall);
fprintf('DTE F1-Score: %.2f\n', f1_score);


% Plot the ROC curves on the same graph
figure;
plot(fpr, tpr, 'b', 'LineWidth', 2); % ROC for DT
hold on; % Keep the same figure for the next plot
plot(Xroc, Yroc, 'r', 'LineWidth', 2); % ROC for NB
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Comparison of ROC Curves');
legend('Decision Tree Ensemble', 'Naive Bayes');
grid on;
hold off;

% Display AUC values for both models
fprintf('AUC for Decision Tree Ensemble: %.4f\n', aucOptimal);
fprintf('AUC for Naive Bayes: %.2f\n', AUC);

% Visualization of the Optimal Tree
view(optimalModel.Trained{1}, 'Mode', 'graph');
