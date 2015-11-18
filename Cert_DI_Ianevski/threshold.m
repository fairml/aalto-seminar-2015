function [DI, BER, e_] = threshold(FULL, N)

% in our example males - minorities (X = 0)
c = N; % number of males being hired (X = 0) (Y = 1)
d = 70 - N; % number of females being hired (X = 1) (Y = 1)
a = size(FULL,1)/2 - c; % number of males being not hired (X = 0) (Y = 0)
b = size(FULL,1)/2 - d; % number of females being not hired (X = 1) (Y = 0)

Conf_m = [a b; c d]; %The confusion matrix
sensitivity = d / (b + d); %sensitivity (true positive rate)
specificity = a / (a + c); %specificity  (true negative rate)
LR = sensitivity / (1 - specificity); %LIKELIHOOD RATIO (POSITIVE))
DI = 1 / LR; %DISPARATE IMPACT (if DI < 0.8, a data set has DI)
%any decision exhibiting disparate impact can be predicted with BER lower than threshold
beta_ = c / (a + c); % ?
alpha_ = b / (b + d); % ?
BER = (1 + beta_ - (1 - alpha_)) / 2; %The balanced error rate (BER)

% Therefore we got a threshold, which is equal:
e_ = 1/2 - beta_/8;

end