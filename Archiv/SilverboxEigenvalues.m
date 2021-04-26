dt = 1/610.35;
a = 2;
d = 0.01;
m = 0.0001;

A = [1,         dt;
    -dt*a/m,    1-dt*d/m];


eig(A)