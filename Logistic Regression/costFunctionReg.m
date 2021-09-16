function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
linear_hypothesis = X*theta;
hypothesis = 1./(1 + exp(-linear_hypothesis));
regularization_theta = theta.^2;
regularization_theta(1,1) = 0;
reg_sum = sum(regularization_theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%((-1/m)*((y'*(log(hypothesis))) + ((1-y)'*(log(1-hypothesis)))))
%((lambda/(2*m))*reg_sum)
%lambda
%2*m
%reg_sum
J = (((-1/m)*((y'*(log(hypothesis))) + ((1-y)'*(log(1-hypothesis)))))+((lambda/(2*m))*reg_sum));

for i=2:size(theta,1),
     grad(1,1) = (1/m)*((X(:,1)')*(hypothesis-y));
     grad(i,1) = ((1/m)*((X(:,i)')*(hypothesis -y))+((lambda/m)*theta(i)));
end;




% =============================================================

end
