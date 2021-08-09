% Plant defined according to https://de.mathworks.com/help/robust/ref/lti.hinfsyn.html#mw_3e327dc9-7ae1-4f2b-b670-9e271011dd8e

% clear
% clc

% Load identified Vertex Systems
load('VertexSystemsTestLPV.mat') 
% load('VertexController_TestLPV.mat')  

S1 = {double(A1),double(B1),double(C1)};

% Define dimensions of plant
nx = 2;
nw = 1;
ny = 1;
nu = 1;
nq = 1;



% In this control problem, certain matrices are fixed
A_1 = S1{1};
B1  = zeros(nx,nw);
B2_1 = S1{2};
C1_1 = S1{3};
D11 = eye(nq,nw);
D12 = zeros(nq,nu);
C2_1 = S1{3};
D21 = eye(ny,nw);
D22 = zeros(ny,nu);

ts=-1;


P = ltisys(A_1,B2_1,C2_1,D22);


[P,r] = sconnect('r','e=P-r','K:e','P:K',P);

% Paug = smult(P,sdiag(w1,1));

[gopt,k] = dhinflmi(P,[1 1],1,1);

Pcl = slft(P,k);

splot(Pcl,'st')

