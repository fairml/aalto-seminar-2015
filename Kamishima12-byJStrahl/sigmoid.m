function g = sigmoid(z, w)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.

if nargin < 2
    w=1;
end

s = dotprod(w,z);
% You need to return the following variables correctly 
g = zeros(size(s));

% Compute the sigmoid of each value of z (where z can be a matrix,
% vector or scalar).

g = 1./(1 + exp(-s));

% This is to avoid log zero!

for e=1:length(g)
    if g(e) == 0.0
        g(e) = 0.00000000001;
    elseif g(e) == 1.0
        g(e) = 0.99999999989999999;
    end

end