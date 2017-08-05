% Sarah Brooke, Gabriel Catalano, Ji Chung, Michael Nguyen

%% Load the data from NSDUH_Table mat
load 'NSDUH_Table.mat'

%% Set whatever is in the data of NSDUH_Table.mat to dSet
Set = importdata('NSDUH_Table.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this is for California's mjUse
dSet = table2array(Set([31:57],[5 6 8]))';
%% 1. Find the sample mean. Zero mean the data
meanCalc = mean(dSet,2);
% Finding the total number of data/columns inside the data
N = size(dSet,2);
Z = dSet - repmat(meanCalc, [1,N]);

%% 2. Using the zero-meaned data, find the sample covaraince matrix
CV = (1/N)*(Z)*(Z');

%% 3. Find the eigenvectors and eigenvalues of the sample covariance matrix. 
[Vp,Dp] = eig(CV);
% What is the eigenvector corresponding to the largest eigenvalue?
disp('The biggest eigenvector is:');
disp(max(Vp));
% What is the eigenvector corresponding to the smallest eigenvalue?
disp('The smallest eigenvector is:');
disp(min(Vp)); 

%% 4. Use eigsort function to sort the eigenvectors and eigenvalues in 
% order of largest eigenvalue to smallest eigenvalue. 
[V,D] = eigsort(Vp, Dp);
% The difference between the matrix of the sorted eigenvectors related to
% the matrix of the unsorted eigenvectors is that eigenvector1 and
% eigenvector2 is switched as well as the eigenvalues that goes with it. 


%% 5. Transform all data points with the PCA transformation. Make a 
% scatterplot of the data in original coordinate and a scatterplot of the
% data in the new coordinate, in two subplots on the same figure. 
C = V'*Z; 
subplot(2,1,1);
scatter(dSet(1,:),dSet(2,:), '+');
subplot(2,1,2);
scatter(C(1,:),C(2,:), 'o');
% Describe the difference. 
% The view is just rotated

%% 6. Make a scatter plot of the original data. On top of the scatterplot 
% of the original data, plot the reconstructed data (so in the original
% naive base) projected onto the 1st eigenvector (largest principal
% component) using red dots to represent the projected data).
figure;
C_hat = C(1,:);
Z_hat = V(:,1)*C_hat + repmat(meanCalc,1,N);
scatter(dSet(1,:),dSet(2,:));
hold on;
scatter(Z_hat(1,:),Z_hat(2,:), 'red');