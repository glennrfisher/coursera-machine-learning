function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% compute cost function
predictions = X * theta;
squared_error = (predictions - y) .^ 2;
mean_squared_error = sum(squared_error) / (2*m);
reg_term = lambda * sum(theta(2:end) .^ 2) / (2*m);
J = mean_squared_error + reg_term;

% compute gradient
grad_unreg = ((predictions - y)' * X)' / m;
grad_reg_term = lambda * theta / m;
grad = grad_unreg + [0; grad_reg_term(2:end)];

% =========================================================================

grad = grad(:);

end
