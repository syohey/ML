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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% reshape y to KxM = 10x5000 matrix
Y = zeros(num_labels, m);
for i = 1:m,
  Y(y(i),i) = 1;
endfor


% PART 1
a_1 = [ones(m,1) X];
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
z_3 = [ones(m,1) a_2] * Theta2';
h = sigmoid(z_3); % also a_3 but the last layer so it becomes h ;)


% calculate cost
c = 1/m; % constant multiplier
temp = 0; % temporary storage variabe for J
for i = 1:m,
  temp = temp ...
         + sum( -Y(:,i)' * log(h(i,:)') ...
         - (1-Y(:,i)') * log(1-h(i,:)') );
endfor

J = c * temp;

% regularize the cost function
##theta1_n = size(Theta1,2);
##theta2_n = size(Theta2,2);

J = J + lambda/(2*m) * ...
    ( ...
        sum( sum(Theta1(:,2:end).^2) ) ...
      + sum( sum(Theta2(:,2:end).^2) ) ...
    );

  


% BACKPROPAGATION  
##X_t = X'; % transpose X = 400x5000
##D_1 = zeros(25,401);
##D_2 = zeros(10,26);
##for t = 1:m,
##  % step 1
##  a_1 = [1 ; X_t(:,t)]; % 401x1
##  z_2 = Theta1 * a_1; % 25x401 * 401x1 = 25x1
##  a_2 = sigmoid(z_2); % 25x1
##  z_3 = Theta2 * [1; a_2]; % 10x26 * 26x1 = 10x1
##  a_3 = sigmoid(z_3); % 10x1
##  
##  % step 2
##  delta_3 = a_3 - Y(:,t); % 10x1
##  
##  % step 3
##  % (10x25)' * 10x1 = 25x1 .* 25x1 = 25x1
##  delta_2 = Theta2(:,2:end)' * delta_3 .* sigmoidGradient(z_2);
##  
##  % step 4
##  D_1 = D_1 + delta_2 * a_1'; % 25x401 + 25x1 * 1x401 = 25x401
##  D_2 = D_2 + delta_3 * [1 ; a_2]'; % 10x26 + 10x1 * 1x26 = 10x26
##  
##endfor
##
##Theta1_grad = 1/m * D_1;
##Theta2_grad = 1/m * D_2;


% PART 2
% backpropagation attempt 2
X_t = X'; % 5000x400 -> 400x5000 --- trnspose X
D_1 = zeros(size(Theta1)); % 25x401
D_2 = zeros(size(Theta2)); % 10x26
for t = 1:m,
  % step 1 --- FP
  a_1 = [1 ; X_t(:,t)]; % 401x1
  z_2 = Theta1 * a_1; % 25x401 * 401x1 = 25x1
  temp = sigmoid(z_2); % 25x1
  a_2 = [1 ; temp]; % 26x1
  z_3 = Theta2 * a_2; % 10x26 * 26x1 = 10x1
  a_3 = sigmoid(z_3); % 10x1
 
  
  % step 2 -- not sure if i did it correctly
  delta_3 = a_3 - Y(:,t); % 10x1

  % step 3 --- ???
  % 26x10 * 10x1 .* 25x1 = 25x1
  delta_2 = Theta2(:,2:end)' * delta_3 .* sigmoidGradient(z_2);
  
  % step 4
  % 25x401 = 25x401 + 25x1 * 1x401
  D_1 = D_1 + delta_2 * a_1';
  % 10x26 = 10x26 + 10x1 * 10x26
  D_2 = D_2 + delta_3 * a_2';  
  
endfor

Theta1_grad = 1/m * D_1; % 25x401
Theta2_grad = 1/m * D_2; % 10x26




% PART 3
reg_1 = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end); % 25x400
reg_2 = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end); % 10x25

Theta1_grad = [Theta1_grad(:,1) reg_1];
Theta2_grad = [Theta2_grad(:,1) reg_2];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
