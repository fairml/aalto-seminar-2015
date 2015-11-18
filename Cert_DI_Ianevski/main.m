clear all;

%function that creates 122 results of SAT exam (61 of both gender)
%which will be classified for (lack of) disparate impact
%and and will be subject to removing DI
%DATA (R) : SCORE(Y), GENDER(X), where X - protected attribute, Y -
%remaining one.

[M_male,S_male,M_female,S_female, Male_, Female_, FULL, test_data] = get_data_();

%Algorithm (A) for hiring 70 people taken exam with the highest score.
%returns number of males out of 70 which were hired. (C = 1, X = 'Male')
%number of females is 70 - N  

N = to_hire(FULL);

%function for getting DI value, BER, and most significantly - BER threshold (e_)

[DI, BER, e_] = threshold(FULL, N);

 
% Certifying DI with SVM

% ALGORITHM:

%1. %RUN A CLASSIFIER, ATTEMPTING TO PREDICT THE PROTECTED 
    %ATTRIBUTES X FROM THE REMAINING ATTRIBUTES Y. 
    %(2/3) - training data, (1/3) - test data.
    
    s = 0;    
    SVMModel = fitcsvm(FULL(1:(2/3)*end,1),FULL(1:(2/3)*end,2),'KernelFunction','rbf','Standardize',true,'ClassNames',{'0','1'});
    [label,score] = predict(SVMModel,FULL(end*(2/3)+1:end,1));
    label = str2num(cell2mat(label));
    
%2  %DETERMINE THE ERROR IN THE PREDICTION
    for i = 1:40
        if (test_data(i) == label(i))
        s = s + 1;
        end
    end
    pr_e = (size(test_data) - s) / size(test_data);
    
%3  %COMPARING THRESHOLD OF E (USING ESTIMATE OF BETA) 
    %AND THE ERROR IN THE PREDICTION
    
    if (e_ < pr_e)
        display('data set free from DI');
    else
        display('DI is certified');
    end


%build probability density functions
[f_m,x_m] = ksdensity(Male_, 'npoints', length(Male_));
[f_f,x_f] = ksdensity(Female_, 'npoints', length(Female_));

%and cumulative distribution functions
%for males and females
Mpd = makedist('Normal','mu',M_male,'sigma',S_male);
cdf_m = cdf(Mpd,x_m);
Fpd = makedist('Normal','mu',M_female,'sigma',S_female);
cdf_f = cdf(Fpd,x_f);

%plot probability density functions and cumulative distribution functions;
plot_functions(Male_, Female_, x_m, x_f, cdf_m, cdf_f, f_m, f_f);

