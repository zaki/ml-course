function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

  % Transform y
  vy = zeros(m, num_labels);
  for i = 1:m
    vy(i, y(i)) = 1;
  end

  % Regularization thetas
  rTheta1 = Theta1(:, 2:size(Theta1, 2));
  rTheta2 = Theta2(:, 2:size(Theta2, 2));

  % Forward propagation
  a1 = [ones(m, 1) X];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2, 1), 1) a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  J = (1 / m) * sum(sum(-vy.*log(a3) - (1 - vy) .* log(1 - a3))) ...
    + lambda / 2 / m * (sum(sum(rTheta1 .* rTheta1)) + sum(sum(rTheta2 .* rTheta2)));

  % Backpropagation
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  for t = 1:m

    % 1. Feedforward for m only
    % column vectors!
    a1 = [1; X(t,:)'];
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % 2. Output layer
    delta3 = a3 - vy(t,:)';

    % 3. Hidden layer
    delta2 = Theta2' * delta3 .* [1; sigmoidGradient(z2)];

    % 4. Accumulation
    Theta2_grad = Theta2_grad + delta3 * a2';
    Theta1_grad = Theta1_grad + delta2(2:end) * a1';
  end

  dTheta2 = zeros(size(Theta2));
  dTheta1 = zeros(size(Theta1));
  dTheta2(:,2:end) = Theta2(:,2:end);
  dTheta1(:,2:end) = Theta1(:,2:end);

  Theta2_grad = 1 / m * Theta2_grad + lambda / m * dTheta2;
  Theta1_grad = 1 / m * Theta1_grad + lambda / m * dTheta1;

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
