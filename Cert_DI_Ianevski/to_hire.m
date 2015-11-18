function [N] = to_hire(FULL)
%returns number of males out of 70 which were hired.

%sort the data
[~,d2] = sort(FULL(:,1));
EL_ = FULL(d2,:);

%get 70 with highest exam
Res = EL_(end-69:end,:);

%find number of males. 
%(since they are coded as 1 I used sum())
N = sum(Res(:,2));
