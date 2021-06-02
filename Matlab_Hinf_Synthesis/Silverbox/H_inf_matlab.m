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
A_1 = S1{1};
B1  = zeros(nx,nw);
B2_1 = S1{2};
C1_1 = S1{3};
D11 = -eye(nq,nw);
D12 = zeros(nq,nu);
C2_1 = S1{3};
D21 = -eye(ny,nw);
D22 = zeros(ny,nu);

A_2 = S2{1};
A_3 = S3{1};
A_4 = S4{1};

B2_2 = S2{2};
B2_3 = S3{2};
B2_4 = S4{2};

C1_2 = S2{3};
C1_3 = S3{3};
C1_4 = S4{3};

C2_2 = S2{3};
C2_3 = S3{3};
C2_4 = S4{3};

% ts=-1;

P1 = ss(A_1,[B1,B2_1],[C1_1;C2_1],[D11,D12;D21,D22],ts);
P2 = ss(A_1,[B1,B2_2],[C1_2;C2_2],[D11,D12;D21,D22],ts);
P3 = ss(A_1,[B1,B2_3],[C1_3;C2_3],[D11,D12;D21,D22],ts);
P4 = ss(A_1,[B1,B2_4],[C1_4;C2_4],[D11,D12;D21,D22],ts);


pols = psys([P1,P2,P3,P4]);


% [gopt,pdK,R,S] = hinfgs(pdP,r,gmin,tol,tolred)