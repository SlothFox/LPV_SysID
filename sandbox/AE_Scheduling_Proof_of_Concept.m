close all


N = 100;

x = zeros(2,N);
x_lin = zeros(2,N);
x_nl = zeros(2,N);

x_AE = zeros(2,N);

x0 = [0.1;0.1];

x(:,1) = x0;

A0 = [0.1,0; -0.1, -0.1];

u = rand(1,N);

for i=2:N
    
    x_new_lin = A0 * x(:,i-1) + u(:,i-1);
    x_new_nl = [0;0.6*x(1,i-1)+0.2*x(2,i-1)] .* x(:,i-1);
    
    x_new = x_new_lin + x_new_nl;
    
    
     x_AE(:,i) = (x_new-A0*x(:,i-1)) .* x(:,i-1)/norm(x(:,i-1))^2;
    
    x(:,i) = x_new;
    x_lin(:,i) = x_new_lin;
    x_nl(:,i) = x_new_nl;
    
end

   
figure;
hold on
scatter(x(1,:),x(2,:))
scatter(x_lin(1,:),x_lin(2,:))
scatter(x_AE(1,:),x_AE(2,:))
hold off

