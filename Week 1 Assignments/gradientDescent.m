function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training example
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

   h=X*theta-y;% m X 1
   delta=(1/m)*(h'*X)';
   theta=theta-(alpha*delta);  %using formula theta=theta-alpha*(1/m)*(h(xi)-y)*Xi
 
   J_history(iter) = computeCost(X, y, theta);

end

end
