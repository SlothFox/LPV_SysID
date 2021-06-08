clear 
clc

load hinfsynExData P

% P = c2d(P,0.01);

ncont = 1; 
nmeas = 2; 

opts = hinfsynOptions('Display','on');
gamRange = [1.4 1.6];
[K,CL,gamma,info] = hinfsyn(P,nmeas,ncont,gamRange,opts);


figure;
stepplot(CL)