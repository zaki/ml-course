function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

  sxt = sigmoid(X * theta);
  ntheta = theta;
  ntheta(1) = 0;

  J = (1 / m) * sum(-y .* log(sxt) - (1 - y) .* log(1 - sxt)) + (lambda / 2 / m) * sum(ntheta .* ntheta);

  beta = sxt .- y;
  grad = (1 / m) * X' * beta + (lambda / m) * ntheta;

  grad = grad(:);
end
