% 2015, Jonathan Strahl, 
% Non-discriminatory ML course, Indrė Žliobaitė, Aalto University
% Implementation of 2012, Kamishima T., Fairness-aware regularizer for
% classification

clear all; close all; clc

%% %%%%%%%%%%%%%%%%%
% Import data
%%%%%%%%%%%%%%%%%%
binData = dlmread('adultd.bindata');
discData = dlmread('adultd.data');

% Pre-process discrete data
colMeans = nanmean(disData);
[row,col] = find(isnan(disData));
disData(isnan(disData)) = colMeans(col);
s_col = 10;
[n, m] = size(disData);
X = disData(:,[1:s_col-1,s_col+1:m-1]);
y = disData(:,m);

% Pre-process binary data
[n, m] = size(binData);
s_col = 81; %Conveniently, second to end, y is end.
X = binData(:,1:m-2);
s = binData(:,m-1);
y = binData(:,m);

%Pre-process common
X = [ones(size(X,1),1), X];


initial_theta_LR = zeros(size(X, 2)+size(s,2), 1);
initial_theta_LR_ns = zeros(size(X, 2), 1);
%initial_theta = randn(length(unique(s)),size(X, 2))*0.1 (Fairer to start all with zero?)
initial_theta_LRPR = zeros(length(unique(s)),size(X, 2)); %Theta for each sensitive value

% LR regularizer parameters 1
lambda = 1;

% Options for 'find minimum of unconstrained multivariable function' (fminunc)
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimise weight parameters with respect to each LR objective function
[theta_LR, cost] = fminunc(@(t)(costFunction(t, [X,s], y)), initial_theta_LR, options);
[theta_LR_ns, J_LR_ns, exit_flag_LR_ns] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta_LR_ns, options);
eta = 1;
[theta_LRPR1, J_LRPR1, exit_flag_LRPR1] = ...
	fminunc(@(t)(costFunctionRegFA(t, X, y, s, lambda, eta)), initial_theta_LRPR, options);
eta = 5;
[theta_LRPR5, J_LRPR5, exit_flag_LRPR5] = ...
	fminunc(@(t)(costFunctionRegFA(t, X, y, s, lambda, eta)), initial_theta_LRPR, options);
eta = 15;
[theta_LRPR15, J_LRPR15, exit_flag_LRPR15] = ...
	fminunc(@(t)(costFunctionRegFA(t, X, y, s, lambda, eta)), initial_theta_LRPR, options);
eta = 30;
[theta_LRPR30, J_LRPR30, exit_flag_LRPR30] = ...
	fminunc(@(t)(costFunctionRegFA(t, X, y, s, lambda, eta)), initial_theta_LRPR, options);
eta = 100;
[theta_LRPR100, J_LRPR100, exit_flag_LRPR100] = ...
	fminunc(@(t)(costFunctionRegFA(t, X, y, s, lambda, eta)), initial_theta_LRPR, options);



y_hat_LR = predict(theta_LR, [X,s]);
y_hat_LR_ns = predict(theta_LR_ns, X);
y_hat_LRPR1 = predictWFA(theta_LRPR1, s, X);
y_hat_LRPR5 = predictWFA(theta_LRPR5, s, X);
y_hat_LRPR15 = predictWFA(theta_LRPR15, s, X);
y_hat_LRPR30 = predictWFA(theta_LRPR30, s, X);
y_hat_LRPR100 = predictWFA(theta_LRPR100, s, X);

Y_hat = [y_hat_LR,y_hat_LR_ns,y_hat_LRPR1,y_hat_LRPR5,y_hat_LRPR15,y_hat_LRPR30,y_hat_LRPR100];

model_results = zeros(size(Y_hat,2),6); % LR,LR_ns,LRPR eta={1,5,15,30,100} x acc., NMI, NPI, UEI, CVS, PI/MI

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Measure accuracy, mutual information (MI), NMI, PI, NPI, UEI, SCVS, ECVS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:size(Y_hat,2)
    % Accuracy = (#truePos + #trueNeg) / (#truePos + #falseP + #trueNeg + #falseN)
    model_results(i,1) = sum(Y_hat(:,i) == y) / length(y);
    % MI = sum p(x,y) log (p(x,y) / (p(x)p(y))), NMI = MI / sqrt(H(X)H(Y)),
    % where H(X) is the entropy of X, that is represented by y_hat
    [NMI,MI] = mutualInformationXY(y,Y_hat(:,i));
    model_results(i,2) = NMI;
    % NPI = sum p(y,s) log(p(y,s)/(p(y)p(s))), NPI = PI / sqrt(H(Y)H(S))
    [NPI,PI] = prejudiceIndex(Y_hat(:,i),s);
    model_results(i,3) = NPI;
    % UEI = sqrt(1 - sqrt(p_hat(y,s)*p_samp(y,s)))
    model_results(i,4) = underEstimationIndex(Y_hat(:,i),y,s);
    % CVS = P(Y=1|S=1) - P(Y=1|S=0)
    model_results(i,5) = CaldersVerwerScore(Y_hat(:,i),s);
    % PI/MI
    model_results(i,6) = PI / MI;
end

%% Results
model_results;

%% OLD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%
% Run LG with 5-fold CV (The paper uses 4 parts for training, learning, and
% 1 part for validation, referred to as test)
%%%%%%%%%%%%%%%%%%%%%%%%%
% indices = crossvalind('Kfold', size(binData,1), 5);

% trData = 1:round(size(binData,1) * 0.8);
% vaData = length(trData)+1:length(trData)+round(size(binData,1) * 0.1)+1;
% tsData = length(trData)+length(vaData)+1:length(binData);

% x_cols = 1:size(binData,2)-1;
% y_col = size(binData,2);


% LG
% [b,dev,stats] = glmfit(binData([trData,vaData],x_cols),binData([trData,vaData],y_col),'binomial','link','logit');
% [y_hat,dy_lo,dy_hi] = glmval(b,binData(tsData,x_cols),'logit',stats);
% y_hat_dis = ~(y_hat<0.5);

% LG without the sensitive feature

%Fairness aware LR
% theta = fairnessAwareLR(discData([trData,vaData],1:size(discData,2)-1),discData([trData,vaData],size(discData,2)));
% p_y_hat_1 = fairnessAwareLRPred(binData(tsData,x_cols),theta);
% y_hat = ~(p_y_hat_1 < 0.5);
% acc = sum(y_hat == binData(tsData,y_col)) / size(binData(tsData,y_col),1);



