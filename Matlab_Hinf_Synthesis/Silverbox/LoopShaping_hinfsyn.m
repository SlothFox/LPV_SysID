% Plant defined according to https://de.mathworks.com/help/robust/ref/lti.hinfsyn.html#mw_3e327dc9-7ae1-4f2b-b670-9e271011dd8e

% clear
% clc

load('VertexSystemsSilverbox.mat') 

% Define dimensions of plant
nx = 2;
nw = 1;
ny = 1;
nu = 1;
nq = 1;

% In this control problem, certain matrices are fixed
A = S1{1};
B1  = zeros(nx,nw);
B2 = S1{2};
C1 = S1{3};
D11 = -eye(nq,nw);
D12 = zeros(nq,nu);
C2 = S1{3};
D21 = -eye(ny,nw);
D22 = zeros(ny,nu);

ts=-1;

P1 = ss(A,[B1,B2],[C1;C2],[D11,D12;D21,D22],ts);

% gam_range = [1.021,1.0211];

[K,CL,gamma,info] = hinfsyn(P1,1,1);




