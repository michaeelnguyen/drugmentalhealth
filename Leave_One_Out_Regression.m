% Multivariate regression 
% Sarah Brooke, Gabriel Catalano, Ji Chung, Michael Nguyen

%% Load the data from NSDUH_Table mat
Set = importdata('NSDUH_Table.mat');
% The set of models for crossvalidation is given as polynomial regressions
% of the 1st, 2nd and 3rd orders.
x1 = importdata('MI_Alc_CHat.mat')';
x2 = importdata('MI_Mj_CHat.mat')';
x3 = importdata('MI_Tobacco_CHat.mat')';
x4 = table2array(Set([200:209],10));
x5 = table2array(Set([200:209],11));
y = importdata('MI_MI_CHat.mat')';
%% Multivariant
% Randomizing the data of MPG
shuffle = randperm(length(y));

% Setting the training set of weights to be the 70% of the randomized values 
trainX1 = x1(shuffle(1:uint32(.7*length(y))));
trainX2 = x2(shuffle(1:uint32(.7*length(y))));
trainX3 = x3(shuffle(1:uint32(.7*length(y))));
trainX4 = x4(shuffle(1:uint32(.7*length(y))));
trainX5 = x5(shuffle(1:uint32(.7*length(y))));

% Setting the training set of mpg to be the 70% of the randomized values 
trainY = y(shuffle(1:uint32(.7*length(y))));

% Setting the testing set of weights to be the 30% of the non-used randomized values 
testX1 = x1(shuffle(uint32(.7*length(y)):end));
testX2 = x2(shuffle(uint32(.7*length(y)):end));
testX3 = x3(shuffle(uint32(.7*length(y)):end));
testX4 = x4(shuffle(uint32(.7*length(y)):end));
testX5 = x5(shuffle(uint32(.7*length(y)):end));

% Setting the testing set of weights to be the 30% of the non-used randomized values 
testY = y(shuffle(uint32(.7*length(y)):end));
%% A matrix of the third order of the training and testing set.
% Each column representing a order
A = [ones(size(trainX1)), trainX1, trainX2, trainX3, trainX4, trainX5];
w1 = A\trainY;
%% This is in order to get the total weight
totalMatrix = [ones(size(x1)), x1, x2, x3, x4, x5];
yPredTotal = totalMatrix*w1;
yPredTest = (w1(1,1)*ones(size(testX1)))+(w1(2,1)*testX1)+(w1(3,1)*testX2)+(w1(4,1)*testX3)+(w1(5,1)*testX4)+(w1(6,1)*testX5);
%% Find the sum squared error of the TESTING SET
squaredErrorTesting = sum((yPredTest - testY).^2);
disp('The sum squared error of the TESTING set for multivariate is');
disp(squaredErrorTesting);

yN2 = size(yPredTest,1);
meanSumErrorTest = (squaredErrorTesting./yN2);

squaredErrorTotal = sum((yPredTotal - y).^2);
ySizeTotal = size(yPredTotal, 1);
meanSumErrorTotal = (squaredErrorTotal./ySizeTotal);
disp('This is the TOTAL mean squared error for the multivariate');
disp(meanSumErrorTotal);

%% Leave One Out Regression

%Sorts multivariate weights from highest to lowest  
[largest_w1,largest_w1_index] = max(abs(w1(2:6,1)));
totalData = [x1,x2,x3,x4,x5];
% Selects attribute with greatest multivariate weight and puts in matrix
% with actual y values, ie mental health
importantData = [totalData(:,largest_w1_index(1)) y];

%% First Order Leave One Out

% Finds 1st order regression weights excluding first data point, then test on first data point
M = size(importantData,1);
oneOutSet = importantData(2:M,:);
N = size(oneOutSet,1);
A1 = [ones(N,1),oneOutSet(:,1)];
y1 = oneOutSet(:,2);
wFirstOrder = A1\y1;

% Test on first data point
yPredict = wFirstOrder(1,1) + (wFirstOrder(2,1)*importantData(1,1));
yActual = importantData(1,2);
firstOrderSSE = (yPredict-yActual).^2;
weightTotalfirstOrder = wFirstOrder;

% Finds 1st order regression weights removing one of each data point then
% testing on that data point
individualTestError = 0;
for x = 2:N
    firstOrderSSE = firstOrderSSE + individualTestError;
    oneOutSet = [importantData(1:x-1,:);importantData(x+1:M,:)];
    A1 = [ones(N,1),oneOutSet(:,1)];
    y1 = oneOutSet(:,2);
    wFirstOrder1 = A1\y1;
    yPredict = wFirstOrder1(1,1) + (wFirstOrder1(2,1)*importantData(x,1));
    yActual = importantData(x,2);
    individualTestError = (yPredict-yActual).^2;
    weightTotalfirstOrder = weightTotalfirstOrder + wFirstOrder1;
end

% Finds 1st order regression weights without last data point, then tests
% on last data point
oneOutSet = importantData(1:N,:);
A1 = [ones(N,1),oneOutSet(:,1)];
y1 = oneOutSet(:,2);
wFirstOrder2 = A1\y1;
yPredict = wFirstOrder2(1,1) + (wFirstOrder2(2,1)*importantData(end,1));
yActual = importantData(end,2);
firstOrderSSE1 = firstOrderSSE + ((yPredict-yActual).^2);
weightTotalfirstOrder = weightTotalfirstOrder + wFirstOrder2;
averageWeightsFirstOrder = weightTotalfirstOrder./M;
firstOrderMSE = firstOrderSSE1./M;
disp('Leave One Out 1st Order MSE is');
disp(firstOrderMSE);

%% Second Order Leave One Out

% Finds 2nd order regression weights excluding first data point, then test on first data point
M = size(importantData,1);
oneOutSet = importantData(2:M,:);
N = size(oneOutSet,1);
A2 = [ones(N,1),oneOutSet(:,1),(oneOutSet(:,1)).^2];
y2 = oneOutSet(:,2);
wSecondOrder = A2\y2;

% Test on first data point
yPredict = wSecondOrder(1,1) + (wSecondOrder(2,1)*importantData(1,1)) + (wSecondOrder(3,1)*(importantData(1,1).^2));
yActual = importantData(1,2);
SecondOrderSSE = (yPredict-yActual).^2;
weightTotalSecondOrder = wSecondOrder;

% Finds 2nd order regression weights removing one of each data point then
% testing on that data point
individualTestError = 0;
for x = 2:N
    SecondOrderSSE = SecondOrderSSE + individualTestError;
    oneOutSet = [importantData(1:x-1,:);importantData(x+1:M,:)];
    A2 = [ones(N,1),oneOutSet(:,1),(oneOutSet(:,1)).^2];
    y2 = oneOutSet(:,2);
    wSecondOrder1 = A2\y2;
    yPredict = wSecondOrder1(1,1) + (wSecondOrder1(2,1)*importantData(1,1)) + (wSecondOrder1(3,1)*(importantData(1,1).^2));
    yActual = importantData(x,2);
    individualTestError = (yPredict-yActual).^2;
    weightTotalSecondOrder = weightTotalSecondOrder + wSecondOrder1;
end
% Finds 2nd order regression weights without last data point, then tests
% on last data point
oneOutSet = importantData(1:N,:);
A2 = [ones(N,1),oneOutSet(:,1),(oneOutSet(:,1).^2)];
y2 = oneOutSet(:,2);
wSecondOrder2 = A2\y2;
yPredict = wSecondOrder2(1,1) + (wSecondOrder2(2,1)*importantData(1,1)) + (wSecondOrder2(3,1)*(importantData(1,1).^2));
yActual = importantData(end,2);
SecondOrderSSE1 = SecondOrderSSE + ((yPredict-yActual).^2);
weightTotalSecondOrder = weightTotalSecondOrder + wSecondOrder2;
averageWeightsSecondOrder = weightTotalSecondOrder./M;
SecondOrderMSE = SecondOrderSSE1./M;
disp('Leave One Out 2nd Order MSE is');
disp(SecondOrderMSE);

%% Third Order Leave One Out

% Finds 3rd order regression weights excluding first data point, then test on first data point
M = size(importantData,1);
oneOutSet = importantData(2:M,:);
N = size(oneOutSet,1);
A3 = [ones(N,1),oneOutSet(:,1),(oneOutSet(:,1)).^2,(oneOutSet(:,1)).^3];
y3 = oneOutSet(:,2);
wThirdOrder = A3\y3;

% Test on first data point
yPredict = wThirdOrder(1,1) + (wThirdOrder(2,1)*importantData(1,1)) + (wThirdOrder(3,1)*(importantData(1,1).^2))+(wThirdOrder(4,1)*(importantData(1,1).^3));
yActual = importantData(1,2);
ThirdOrderSSE = (yPredict-yActual).^2;
weightTotalThirdOrder = wThirdOrder;

% Finds 3rd order regression weights removing one of each data point then
% testing on that data point
individualTestError = 0;
for x = 2:N
    ThirdOrderSSE = ThirdOrderSSE + individualTestError;
    oneOutSet = [importantData(1:x-1,:);importantData(x+1:M,:)];
    A3 = [ones(N,1),oneOutSet(:,1),(oneOutSet(:,1)).^2,(oneOutSet(:,1)).^3];
    y3 = oneOutSet(:,2);
    wThirdOrder1 = A3\y3;
    yPredict = wThirdOrder1(1,1) + (wThirdOrder1(2,1)*importantData(1,1)) + (wThirdOrder1(3,1)*(importantData(1,1).^2))+(wThirdOrder1(4,1)*(importantData(1,1).^3));
    yActual = importantData(x,2);
    individualTestError = (yPredict-yActual).^2;
    weightTotalThirdOrder = weightTotalThirdOrder + wThirdOrder1;
end

% Finds 3rd order regression weights without last data point, then tests
% on last data point
oneOutSet = importantData(1:N,:);
A3 = [ones(N,1),oneOutSet(:,1),(oneOutSet(:,1).^2),oneOutSet(:,1).^3];
y3 = oneOutSet(:,2);
wThirdOrder2 = A3\y3;
yPredict = wThirdOrder2(1,1) + (wThirdOrder2(2,1)*importantData(1,1)) + (wThirdOrder2(3,1)*(importantData(1,1).^2))+(wThirdOrder2(4,1)*(importantData(1,1).^3));
yActual = importantData(end,2);
ThirdOrderSSE1 = ThirdOrderSSE + ((yPredict-yActual).^2);
weightTotalThirdOrder = weightTotalThirdOrder + wThirdOrder2;
averageWeightsThirdOrder = weightTotalThirdOrder./M;
ThirdOrderMSE = ThirdOrderSSE1./M;
disp('Leave One Out 3rd Order MSE is');
disp(ThirdOrderMSE);

%% Plot data

MSEmatrix = [firstOrderMSE SecondOrderMSE ThirdOrderMSE];
[MSEmin, MSEminIndex] = min(MSEmatrix);
xTest = [min(importantData,1)-1 : 0.1 : max(importantData,1)+4]';

% Creating the aTest matrix.
ATest1 = [ones(length(xTest),1), xTest];
ATest2 = [ones(length(xTest),1), xTest, xTest.^2];
ATest3 = [ones(length(xTest),1), xTest, xTest.^2, xTest.^3];

% Displaying and plotting the graph/points
plot(xTest, ATest1*averageWeightsFirstOrder, xTest,ATest2*averageWeightsSecondOrder, xTest,ATest3*averageWeightsThirdOrder); 

%legend('First Order', 'Second Order', 'Third Order')
scatter(importantData(:,1),y);
axis([(min(importantData(:,1))-1) (max(importantData(:,1))+1) (min(y)-10) (max(y)+10)]);
hold on 

plot(xTest, ATest1*averageWeightsFirstOrder);

plot(xTest,ATest2*averageWeightsSecondOrder);
    
plot(xTest,ATest3*averageWeightsThirdOrder);
       

