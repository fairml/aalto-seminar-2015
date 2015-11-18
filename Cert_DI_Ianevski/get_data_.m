function [M_male,S_male,M_female,S_female, Male_, Female_, FULL, test_data] = get_data_()

%some statistic data from paper example
%our test will have a data with the same parameters 
M_male = 400; %mean of Males
S_male = 50; %std of Males

M_female = 550; %mean of females
S_female = 100; %std of females

%create 'FAKE' data according to these statistical
%Results of SAT exam for males and females
% Male_ = round(randn(61,1) * (S_male) + M_male); % create male data
% Female_ = round(randn(61,1) * (S_female) + M_female); % create female data
Data = xlsread('SAT.xlsx');
Male_ = Data(:,1);
Female_ = Data(:,2);
%create FULL variable, which comprises all joined (and shuffled) values of
%males and females
for i = 1:length(Female_)
    Female_full(i,1) = Female_(i,1);
    Female_full(i,2) = 0;
    Male_full(i,1) = Male_(i,1);
    Male_full(i,2) = 1;
end
FULL = cat(1, Male_full(:,:),Female_full(:,:));
FULL = FULL(randperm(size(FULL,1)),:);

%test data - 1/3 of FULL data. therefore: 2/3 are trained data
test_data = FULL(end*(2/3)+1:end,2);

% We can also look at the data (check deviations from normal distribution)
 figure;
 qqplot(Female_); 
 qqplot(Male_);
end
