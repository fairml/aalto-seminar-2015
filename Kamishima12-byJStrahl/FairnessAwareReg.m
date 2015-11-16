% 2012, Kamishima T., Fairness-aware regularizer

% Import data
binaryData = dlmread('adultd.bindata');
discData = dlmread('adultd.data');
s = discData(:,10); %senstitive feature: gender (M=0,F=1)

sum(max(discData(:,1:15)));

% Run LG with 5-fold CV
trData = 1:round(size(binaryData,1) * 0.8);
vaData = length(trData)+1:length(trData)+round(size(binaryData,1) * 0.1)+1;
tsData = length(trData)+length(vaData)+1:length(binaryData);

x_cols = 1:size(binaryData,2)-1;
y_col = size(binaryData,2);

[b,dev,stats] = glmfit(binaryData([trData,vaData],x_cols),binaryData([trData,vaData],y_col),'binomial','link','logit');
[y_hat,dy_lo,dy_hi] = glmval(b,binaryData(tsData,x_cols),'logit',stats);
y_hat_dis = ~(y_hat<0.5);

bin_tst = [1 0 0 0 1 1 1 0]';
bin_pred = [1 0 1 0 0 1 1 0 ]';

sum(bin_tst == bin_pred) / size(bin_tst,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Measure accuracy, mutual information (MI), NMI, PI, NPI, UEI, SCVS, ECVS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Accuracy = (#truePos + #trueNeg) / (#truePos + #falseP + #trueNeg + #falseN)
acc = sum(y_hat_dis == binaryData(tsData,y_col)) / size(binaryData(tsData,y_col),1);

% MI = sum p(x,y) log (p(x,y) / (p(x)p(y))), NMI = MI / sqrt(H(X)H(Y)),
% where H(X) is the entropy of X
J = [sum(~y_hat_dis & ~binaryData(tsData,y_col)),sum(~y_hat_dis & binaryData(tsData,y_col));sum(y_hat_dis & ~binaryData(tsData,y_col)),sum(y_hat_dis & binaryData(tsData,y_col))]/size(binaryData(tsData,y_col),1);
MI = sum(sum(J.*log2(J./(sum(J,2)*sum(J,1))))); 
p_y_hat = sum(y_hat_dis) / length(y_hat_dis);
H_b_y_hat = -p_y_hat * log2(p_y_hat) - (1-p_y_hat) * log2(1-p_y_hat);
p_y_tst = sum(binaryData(tsData,y_col)) / size(binaryData(tsData,y_col),1);
H_b_y_tst = -p_y_tst * log2(p_y_tst) - (1-p_y_tst) * log2(1-p_y_tst);
NMI = MI / sqrt(H_b_y_hat*H_b_y_tst);

% NPI = sum p(y,s) log(p(y,s)/(p(y)p(s))), NPI = PI / sqrt(H(Y)H(S))
s(tsData)
J = [sum(~binaryData(tsData,y_col) & ~s(tsData)),sum(~binaryData(tsData,y_col) & s(tsData));sum(binaryData(tsData,y_col) & ~s(tsData)),sum(binaryData(tsData,y_col) & s(tsData))]/size(binaryData(tsData,y_col),1);
PI = sum(sum(J.*log2(J./(sum(J,2)*sum(J,1))))); 
p_s = sum(s(tsData)) / length(s(tsData));
H_b_s = -p_s * log2(p_s) - (1-p_s) * log2(1-p_s);
NPI = PI / sqrt(H_b_y_tst * H_b_s);

% UEI = sqrt(1 - sqrt(p_hat(y,s)*p_samp(y,s)))
p_hat_y_s = [sum(y_hat_dis & s(tsData)),sum(y_hat_dis & ~s(tsData)),sum(~y_hat_dis & s(tsData)),sum(~y_hat_dis & ~s(tsData))]/length(y_hat_dis);
p_samp_y_s = [sum(binaryData([trData,vaData],y_col) & s([trData,vaData])),sum(binaryData([trData,vaData],y_col) & ~s([trData,vaData])),sum(~binaryData([trData,vaData],y_col) & s([trData,vaData])),sum(~binaryData([trData,vaData],y_col) & ~s([trData,vaData]))]/size(binaryData([trData,vaData],y_col),1);
UEI = sqrt(1-(sum(sqrt(p_hat_y_s.*p_samp_y_s))));

% CVS = P(Y=1|S=1) - P(Y=1|S=0)
p_y_s1 = sum(y_hat_dis & s(tsData)) / sum(s(tsData));
p_y_s0 = sum(y_hat_dis & ~s(tsData)) / sum(~s(tsData));
CVS = p_y_s1 - p_y_s0;

resultTab = {'LR', acc, NMI, NPI, UEI, CVS, PI/MI };
