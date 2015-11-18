data = load('credit.csv');
%we fix the protected group according to the age <=25
protected = data(data(:,13) <= 25,:);
%we fix the upprotected group to other than the previous
unprotected = data(data(:,13) > 25,:);
%save class and remove from attributes
class_p = protected(:,21);
class_u = unprotected(:,21);
protected(:,21) = [];
unprotected(:,21) = [];
%for each woman assigned with a BAD class in the protected dataset find diff 
idx_p = knnsearch( protected, protected(class_p == 2, :),'K', 16 );
idx_u = knnsearch( unprotected, protected(class_p == 2, :),'K', 16 );
%to find the proportion p1
n = size(idx_p, 1);
p1 = zeros(1,n);
for i=1:n
    tbl = tabulate(class_p(idx_p(i,:)));
    if isempty (tbl(tbl(:,1) == 2,3))
        p1(i) = 0;
    else
        p1(i) = tbl(2,3);
    end;
end;
%to find the proportion p2
n = size(idx_u, 1);
p2 = zeros(1,n);
for i=1:n
    tbl = tabulate(class_u(idx_u(i,:)));
    if isempty (tbl(tbl(:,1) == 2,3))
        p2(i) = 0;
    else
        p2(i) = tbl(tbl(:,1) == 2,3);
    end;
end;
%to find diff
diff = (p1-p2)/100;
[f,x] = ecdf(diff);
figure(1)
plot(x,f)


%we fix the protected group according to the age 25-60

protected = data((data(:,13) > 25) & (data(:,13) <= 60) ,:);
%protected = data(data(:,13) > 60 ,:);
%we fix the upprotected group to other than the previous
unprotected = data((data(:,13) <= 25) | (data(:,13) > 60),:);
%unprotected = data(data(:,13) <= 60,:);
%save class and remove from attributes
class_p = protected(:,21);
class_u = unprotected(:,21);
protected(:,21) = [];
unprotected(:,21) = [];
%for each woman assigned with a BAD class in the protected dataset find diff 
idx_p = knnsearch( protected, protected(class_p == 2, :),'K', 16 );
idx_u = knnsearch( unprotected, protected(class_p == 2, :),'K', 16 );
%to find the proportion p1
n = size(idx_p, 1);
p1 = zeros(1,n);
for i=1:n
    tbl = tabulate(class_p(idx_p(i,:)));
    if isempty (tbl(tbl(:,1) == 2,3))
        p1(i) = 0;
    else
        p1(i) = tbl(2,3);
    end;
end;
%to find the proportion p2
n = size(idx_u, 1);
p2 = zeros(1,n);
for i=1:n
    tbl = tabulate(class_u(idx_u(i,:)));
    if isempty (tbl(tbl(:,1) == 2,3))
        p2(i) = 0;
    else
        p2(i) = tbl(tbl(:,1) == 2,3);
    end;
end;
%to find diff
diff = (p1-p2)/100;
[f,x] = ecdf(diff);
hold on
plot(x,f)


%we fix the protected group according to the age >60

%protected = data((data(:,13) > 25) & (data(:,13) <= 60) ,:);
protected = data(data(:,13) > 60 ,:);
%we fix the upprotected group to other than the previous
%unprotected = data((data(:,13) <= 25) | (data(:,13) > 60),:);
unprotected = data(data(:,13) <= 60,:);
%save class and remove from attributes
class_p = protected(:,21);
class_u = unprotected(:,21);
protected(:,21) = [];
unprotected(:,21) = [];
%for each woman assigned with a BAD class in the protected dataset find diff 
idx_p = knnsearch( protected, protected(class_p == 2, :),'K', 16 );
idx_u = knnsearch( unprotected, protected(class_p == 2, :),'K', 16 );
%to find the proportion p1
n = size(idx_p, 1);
p1 = zeros(1,n);
for i=1:n
    tbl = tabulate(class_p(idx_p(i,:)));
    if isempty (tbl(tbl(:,1) == 2,3))
        p1(i) = 0;
    else
        p1(i) = tbl(2,3);
    end;
end;
%to find the proportion p2
n = size(idx_u, 1);
p2 = zeros(1,n);
for i=1:n
    tbl = tabulate(class_u(idx_u(i,:)));
    if isempty (tbl(tbl(:,1) == 2,3))
        p2(i) = 0;
    else
        p2(i) = tbl(tbl(:,1) == 2,3);
    end;
end;
%to find diff
diff = (p1-p2)/100;
[f,x] = ecdf(diff);
hold on
plot(x,f)