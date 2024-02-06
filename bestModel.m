% Load the dataset with original column headers
opts = detectImportOptions('MIT_BIH_Arrhythmia_Database.csv');
opts.VariableNamingRule = 'preserve';  % Preserve original column headers
ecgData = readtable('MIT_BIH_Arrhythmia_Database.csv', opts);

% Extract features and labels
data = ecgData(:, 3:end);  % Extracting feature columns (excluding record and type)

% Map labels to binary classes
labels = ecgData(:, 2);
isArrhythmia = ismember(labels{:,:}, {'VEB', 'SVEB', 'F', 'Q'});
binaryLabels = categorical(isArrhythmia, [0, 1], {'Normal', 'Arrhythmia'});

% Split data into predictors and response
X = table2array(data);     % Convert table to array for predictors

% Perform data normalization
mu = mean(X);
sigma = std(X);
X = (X - mu) ./ sigma;  % Normalize each feature (column) to have zero mean and unit variance


% Split data into training, validation, and testing sets (60% train, 20% validation, 20% test)
cv = cvpartition(size(X, 1), 'HoldOut', 0.4);  % 60% train, 40% (validation + test) split
idxTrainVal = cv.training;
dataTrainVal = X(idxTrainVal, :);
labelsTrainVal = binaryLabels(idxTrainVal);
dataTest = X(~idxTrainVal, :);
labelsTest = binaryLabels(~idxTrainVal);

% Further split the remaining 40% into validation and test sets (equal split)
cvValTest = cvpartition(sum(~idxTrainVal), 'HoldOut', 0.5);
idxVal = cvValTest.training;
dataVal = dataTest(~idxVal, :);
labelsVal = labelsTest(~idxVal);
dataTest = dataTest(idxVal, :);
labelsTest = labelsTest(idxVal);

% Save training, validation, and test datasets along with labels
trainTable = array2table([double(labelsTrainVal), dataTrainVal]);
writetable(trainTable, 'train_data.csv');

valTable = array2table([double(labelsVal), dataVal]);
writetable(valTable, 'val_data.csv');

testTable = array2table([double(labelsTest), dataTest]);
writetable(testTable, 'test_data.csv');

% Perform PCA for dimensionality reduction
numComponents = 4;  % Set the desired number of components
coeff = pca(dataTrainVal);
dataTrainValReduced = dataTrainVal * coeff(:, 1:min(size(coeff,2), numComponents));
dataValReduced = dataVal * coeff(:, 1:min(size(coeff,2), numComponents));
dataTestReduced = dataTest * coeff(:, 1:min(size(coeff,2), numComponents));

% Visualize the reduced feature space for training data
figure;
gscatter(dataTrainValReduced(:,1), dataTrainValReduced(:,2), labelsTrainVal, 'rb', 'xo');
title('PCA: Reduced Feature Space for Training Data');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend('Normal', 'Arrhythmia');
grid on;

% Visualize the reduced feature space for validation data
figure;
gscatter(dataValReduced(:,1), dataValReduced(:,2), labelsVal, 'rb', 'xo');
title('PCA: Reduced Feature Space for Validation Data');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend('Normal', 'Arrhythmia');
grid on;

% Visualize the reduced feature space for test data
figure;
gscatter(dataTestReduced(:,1), dataTestReduced(:,2), labelsTest, 'rb', 'xo');
title('PCA: Reduced Feature Space for Test Data');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend('Normal', 'Arrhythmia');
grid on;

% Get the number of unique classes from label
numClasses = 2;

% Train SVM model on the reduced data
SVMModel = fitcecoc(dataTrainValReduced, labelsTrainVal);


% Predict labels for test data
YPredTest = predict(SVMModel, dataTestReduced);

% Evaluate the accuracy on the test set
accuracyTest = sum(YPredTest == labelsTest) / numel(labelsTest);
fprintf('SVM Test Accuracy after PCA: %.2f%%\n', accuracyTest * 100);

% Confusion matrix
C = confusionmat(labelsTest, YPredTest);
figure;
confusionchart(C);

save('SVMModel.mat', 'SVMModel');
% 'mu', 'sigma', and 'coeff' are computed during preprocessing
save('preprocessingParams.mat', 'mu', 'sigma', 'coeff');

% Model Training Iterations:
% The current code is training the SVM model in each iteration without updating or fine-tuning it based on the previous results.
% Typically, in iterative training, you would update the model parameters based on the performance on the validation set.
% Otherwise, you might end up with similar results across iterations.

% Number of iterations
numIterations = 5;
% Initialize arrays to store results
trainAccuracies = zeros(numIterations, 1);
valAccuracies = zeros(numIterations, 1);
losses = zeros(numIterations, 1);
precision = zeros(numClasses, numIterations);
recall = zeros(numClasses, numIterations);

% Create waitbar
h = waitbar(0, 'Training in progress...');

for iter = 1:numIterations
    % Update waitbar
    waitbar(iter/numIterations, h, sprintf('Training iteration %d of %d...', iter, numIterations));

    % Train SVM model
    SVMModel = fitcecoc(dataTrainValReduced, labelsTrainVal);

    % Predict labels for training and validation data
    YPredTrain = predict(SVMModel, dataTrainValReduced);
    YPredVal = predict(SVMModel, dataValReduced);

    % Evaluate the accuracy on training and validation sets
    trainAccuracies(iter) = sum(YPredTrain == labelsTrainVal) / numel(labelsTrainVal);
    valAccuracies(iter) = sum(YPredVal == labelsVal) / numel(labelsVal);

    % Simulated cross-entropy loss (replace with actual loss calculation if available)
    labelsTrainValNum = double(labelsTrainVal);
    labelsValNum = double(labelsVal);

    % Convert predicted labels to numeric values for loss calculation
    [~, ~, YPredTrainNum] = unique(YPredTrain);
    [~, ~, YPredValNum] = unique(YPredVal);

    % Calculate losses
    lossTrain = log(1 + exp(-YPredTrainNum .* (labelsTrainValNum - 1)));
    lossVal = log(1 + exp(-YPredValNum .* (labelsValNum - 1)));
    losses(iter) = mean([lossTrain; lossVal]);

    % Store precision and recall for each class
    for i = 1:numClasses
        C = confusionmat(labelsVal, YPredVal);
        if sum(C(i, :)) == 0 || sum(C(:, i)) == 0
            precision(i, iter) = 0;
            recall(i, iter) = 0;
        else
            precision(i, iter) = C(i, i) / sum(C(:, i));
            recall(i, iter) = C(i, i) / sum(C(i, :));
        end
    end

    % Print accuracy and loss for each iteration
    fprintf('Iteration %d - Training Accuracy: %.2f%%, Validation Accuracy: %.2f%%, Loss: %.4f\n', iter, trainAccuracies(iter) * 100, valAccuracies(iter) * 100, losses(iter));
end

% Close the waitbar
close(h);

% Plot the accuracy and loss curves
figure;

subplot(2, 1, 1);
plot(1:numIterations, trainAccuracies, 'b', 'LineWidth', 1.5);
hold on;
plot(1:numIterations, valAccuracies, 'r', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Accuracy');
title('Training and Validation Accuracy vs. Iteration');
legend('Training Accuracy', 'Validation Accuracy');
grid on;

subplot(2, 1, 2);
plot(1:numIterations, losses, 'k', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Loss');
title('Simulated Cross-Entropy Loss vs. Iteration');
grid on;

% Display precision and recall for each class
fprintf('Precision and Recall for Each Class:\n');
for i = 1:numClasses
    fprintf('Class %d - Precision: %.2f, Recall: %.2f\n', i, mean(precision(i, :)), mean(recall(i, :)));
end
