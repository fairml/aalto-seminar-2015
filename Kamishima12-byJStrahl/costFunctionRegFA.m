function [costJWithRegularizationAndPrejudiceRegularizer, gradientWithPrejudiceRegularization] = costFunctionRegFA(theta, X, y, s, lambda, eta)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% See pr.py in the fadm/lr/ directory in Kamishima's Python implementation
% for more details.
% etta: prejudice aware regularizer parameter
% s_col: column number for the sensitive feature

% X: predictors without the sensitive feature
% s: the sensitive feature

% Test code
% X = [11,12;21,22;31,32]
% theta = [111,112;221,222]'
% s = [1,0,1]'
% y = [1,1,0]'
% lambda=1
% eta=1

% Initialize some useful values
% y = mx1 column vector
numberOfTrainingExamples = length(y); % = m

% return the following variables correctly  
costJ = 0; % costJ = single number 
gradnt = zeros(size(theta,1),1); % gradient = nx1 column vector (same size as theta)
gradnt_par_q = zeros(size(theta,1),1);

% Compute the costJ of a particular choice of theta

% Extract s for now from x!

% compute cost costJ
% X = mxn matrix
% theta = #s values x #features
hypothesis = zeros(numberOfTrainingExamples,1);  % p(y|x,s) = p from Kamishima's code.
for i = 1:numberOfTrainingExamples
    hypothesis(i) = sigmoid(X(i,:)',theta(s(i)+1,:)); % hypothesis = mx1 column vector
end

s_values = unique(s);

q = zeros(size(s_values));
for i = 1:length(s_values)
    q(i) = sum(hypothesis(s==s_values(i))) / sum(s==s_values(i));
end
r = sum(hypothesis) / size(X,1);

% costJ = likelihood:
%costJ = (-1/numberOfTrainingExamples) * sum( y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis) );
costJ = -1 * sum( y .* log(hypothesis) + (1.0 - y) .* log(1 - hypothesis) );

%% Prejudice remover regularizer

% From Kamishima's Python implementation:
%        # fairness-aware regularizer
%         # \sum_{x,s in D} \
%         #    sigma(x,x)       [log(rho(s))     - log(pi)    ] + \
%         #    (1 - sigma(x,s)) [log(1 - rho(s)) - log(1 - pi)]
%         f = np.sum(p * (np.log(q[s]) - np.log(r))
%              + (1.0 - p) * (np.log(1.0 - q[s]) - np.log(1.0 - r)))
%         # rho(s) = Pr[y=0|s] = \sum_{(xi,si)in D st si=s} sigma(xi,si) / #D[s]
%         q = np.array([np.sum(p[s == si])
%                       for si in xrange(self.n_sfv_)]) / self.c_s_
%         # sigma = Pr[y=0|x,s] = sigmoid(w(s)^T x)
%         p = np.array([sigmoid(X[i, :], coef[s[i], :])
%                       for i in xrange(self.n_samples_)])
%        # pi = Pr[y=0] = \sum_{(xi,si)in D} sigma(xi,si)
%         r = np.sum(p) / self.n_samples_
costPrejudiceRemoverRegularizer = sum(hypothesis .* (log(q(s+1)) - log(r)) + (1.0 - hypothesis) .* (log(1 - q(s+1)) - log(1.0 - r)));

%costRegularizationTerm = lambda/(2*numberOfTrainingExamples) * sum( theta(2:end).^2 );
costRegularizationTerm =  sum(sum( theta.^2 ));

costJWithRegularizationAndPrejudiceRegularizer = costJ + eta * costPrejudiceRemoverRegularizer + lambda/2 * costRegularizationTerm;

% Compute the partial derivatives and set gradiant to the partial
% derivatives of the cost w.r.t. each parameter in theta

% compute the gradient 
% for i = 1:numberOfTrainingExamples
% 	% hypothesis = mx1 column vector
% 	% y = mx1 column vector
% 	% X = mxn matrix
% 	gradnt = gradnt + ( hypothesis(i) - y(i) ) * X(i, :)';
% end

gradHypothesis = bsxfun(@times,(hypothesis .* (1.0 - hypothesis)),X);

gradQ = zeros(length(s_values),size(gradHypothesis,2));
for i = 1:length(s_values)
    gradQ(i,:) = sum(gradHypothesis(s == s_values(i),:),1) / sum(s == s_values(i));
end

gradR = sum(sum(gradHypothesis),1) / size(X,1);

gradL = zeros(length(s_values),size(X,2)); % #s-values by #features
for i = 1:length(s_values)
    s_i = s == s_values(i);
    gradL(i,:) = sum(bsxfun(@times,(y(s_i) - hypothesis(s_i)),X(s_i,:)),1);
end

% fairness-aware regularizer gradient:
%         # differentialy by w(s)
%         # \sum_{x,s in {D st s=si} \
%         #     [(log(rho(si)) - log(pi)) - (log(1 - rho(si)) - log(1 - pi))] \
%         #     * d_sigma
%         # + \sum_{x,s in {D st s=si} \
%         #     [ {sigma(xi, si) - rho(si)} / {rho(si) (1 - rho(si))} ] \
%         #     * d_rho
%         # - \sum_{x,s in {D st s=si} \
%         #     [ {sigma(xi, si) - pi} / {pi (1 - pi)} ] \
%         #     * d_pi
f1 = (log(q(s+1)) - log(r)) - (log(1.0 - q(s+1)) - log(1.0 - r));
f2 = (hypothesis - q(s+1)) ./ (q(s+1) .* (1.0 - q(s+1)));
f3 = (hypothesis - r) / (r * (1 - r));
f4 = bsxfun(@times,f1,gradHypothesis) + bsxfun(@times,f2,gradQ(s+1,:)) - bsxfun(@times,f3,gradR);

f = zeros(length(s_values),size(f4,2));
for i = 1:length(s_values)
    s_i = s == s_values(i);
    f(i,:) = sum(f4(s_i,:),1);
end



%gradientRegularizationTerm = lambda/numberOfTrainingExamples * [0 0; theta(2:end,:)]; 
gradReg = theta; 


% where [0; theta(2:end)] is the same column vector theta beginning with a value of '0' at index
% 1 and then containing the old values from index 2:end of theta

% gradient = nx1 column vector
%gradientWithRegularization = (1/numberOfTrainingExamples) * [gradnt,gradnt] + gradientRegularizationTerm;
gradientWithPrejudiceRegularization = -1 * gradL + eta * f + lambda * gradReg;
end
