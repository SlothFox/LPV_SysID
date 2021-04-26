clc
clear

% Load identified Vertex Systems
load('VertexSystemsSilverbox.mat') 


% Specify dimensions of problem
nx = 2;
nw = 1;
ny = 1;
nu = 1;
nq = 1;
k = 2;

% In this control problem, certain matrices are fixed
B1  = zeros(nx,nw);
D11 = eye(nq,nw);
D12 = zeros(nq,nu);
D21 = zeros(ny,nw);
D22 = zeros(ny,nu);

% Define Vertex Systems
A  = S1{1};%+1*eye(nx);
B2 = S1{2};
C1 = S1{3};
C2 = S1{3};
s1 = ltisys(A,B2,C1,D12);

A  = S2{1};%+1*eye(nx);
B2 = S2{2};
C1 = S2{3};
C2 = S2{3};
% s2 = [A,B1,B2;C1,D11,D12;C2,D21,D22];
s2 = ltisys(A,B2,C1,D12);


A  = S3{1};%+1*eye(nx);
B2 = S3{2};
C1 = S3{3};
C2 = S3{3};
% s3 = [A,B1,B2;C1,D11,D12;C2,D21,D22];
s3 = ltisys(A,B2,C1,D12);

A  = S4{1};%+1*eye(nx);
B2 = S4{2};
C1 = S4{3};
C2 = S4{3};
% s4 =[A,B1,B2;C1,D11,D12;C2,D21,D22];
s4 = ltisys(A,B2,C1,D12);

pols = psys([s1,s2,s3,s4]);

[gopt,pdK,R,S] = hinfgs(pols,[1,1])

