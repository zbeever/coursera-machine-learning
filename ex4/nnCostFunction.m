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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

y = eye(num_labels)(y, :); % Change the training answers to vectors classifying inputs by their jth index

a1 = [ones(m, 1) X]; % Add a set of bias units to the training set

z2 = a1 * Theta1'; % Apply the weights of the first set of connections to the training set
a2 = sigmoid(z2); % Run it through the sigmoid function

a2 = [ones(size(a2, 1), 1) a2]; % Add another set of bias units
z3 = a2 * Theta2'; % Apply the weights of the second set of connections
a3 = sigmoid(z3); % Run it through the sigmoid function

logistf = -y .* log(a3) - (1 - y) .* log(1 - a3); % The heart of the cost function is still the logistic regression cost function
J = ((1 / m) .* sum(sum(logistf))) + (lambda / (2 * m)) .* (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))); % Added to the cost function is the regularized addendum focused on minimizing the norm of both theta matrices

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

accum_2 = 0;
accum_1 = 0;

delta_3 = a3 - y; % The outermost layer's error is just the difference between the training answers and our hypotheses
delta_2 = delta_3 * Theta2(:,2:end) .* sigmoidGradient(z2); % Hidden layers have slightly more complex errors - the above layer of error matrix-multiplied by the current layer's theta matrix and then element-wise multiplied by the gradient of the current activation layer

accum_2 = delta_3' * a2; % Although computing the above seems complex, it simplifies the gradient to this: the transpose of the error matrix matrix-multiplied by the current activation layer
accum_1 = delta_2' * a1; % Ditto

Theta2_grad = accum_2 / m; % The final gradient is divided by m, the number of training examples
Theta1_grad = accum_1 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2(:, 1) = 0; % Remove any effect the bias units might have
Theta1(:, 1) = 0;

reg_grad_2 = (lambda / m) .* Theta2; % Element-wise multiply Y/m by our theta matrices
reg_grad_1 = (lambda / m) .* Theta1;

Theta2_grad = Theta2_grad + reg_grad_2; % Add these regularized terms to their respective theta gradients
Theta1_grad = Theta1_grad + reg_grad_1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
