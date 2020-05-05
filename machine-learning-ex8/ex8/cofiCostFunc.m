function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X)); %%% 5x3
Theta_grad = zeros(size(Theta)); %%% 4x3

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

R1 = X*Theta'; %%% 5x3 * 3x4 = 5x4

##J = 1/2 * sum( (R1(R==1) - Y(R==1)).^2 );


J = 1/2 * sum( (R1(R==1) - Y(R==1)).^2 ) + ...
    lambda/2 * sum( sum( Theta.^2 ) ) + ...
    lambda/2 * sum( sum( X.^2 ) );







##for i = 1:num_movies,
##  r1 = X(i,:)*Theta'; % 1x3 * 3x4 = 1x4
##  diff = (r1.*R(i,:) - Y(i,:).*R(i,:)); % 1x4
##  X_grad(i,:) = diff * Theta; % 1x4 * 4x3 = 1x3
##endfor
##
##for j = 1:num_users,
##  r1 = X*Theta(j,:)'; % 5x3 * 3x1 = 5x1
##  diff = (r1.*R(:,j) - Y(:,j).*R(:,j)); % 5x1 .* 5x1 ... -> 5x1
##  Theta_grad(j,:) = diff' * X; % 1x5 * 5x3 = 1x3
##endfor




for i = 1:num_movies,
  r1 = X(i,:)*Theta'; % 1x3 * 3x4 = 1x4
  diff = (r1.*R(i,:) - Y(i,:).*R(i,:)); % 1x4
  X_grad(i,:) = diff * Theta + lambda*X(i,:); % 1x4 * 4x3 = 1x3
endfor

for j = 1:num_users,
  r1 = X*Theta(j,:)'; % 5x3 * 3x1 = 5x1
  diff = (r1.*R(:,j) - Y(:,j).*R(:,j)); % 5x1 .* 5x1 ... -> 5x1
  Theta_grad(j,:) = diff' * X + lambda*Theta(j,:); % 1x5 * 5x3 = 1x3
endfor




% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
