%% Now figure how to apply filter in a simple way
clear
clc
close all

% Add YALMIP to path
addpath(genpath('..\YALMIP-master'))  
% Linux:
% addpath '\mosek-linux\9.2\toolbox\r2015a'

% Windows
addpath 'C:\Program Files\Mosek\9.2\toolbox\R2015a'
ops = sdpsettings('solver','mosek');

% Load identified Vertex Systems
load('VertexSystemsSilverbox.mat') 

% Specify dimensions of problem
nx = 2;
nw = 1;
ny = 1;
nu = 1;
nq = 1;
k = 2;


% Define state space matrices of general control structure
B1 = [0;0];
D11 = [1];
D12 = [0.0001];          % Fix cheap control problem
D21 = [1];
D22 = [0];

ts = (2^14)/(10^7); %  -1;    % Discrete system with unspecified sampling time

% Define vertex systems
P1 = ss(S1{1},[B1,S1{2}],[-S1{3};-S1{3}],[D11,D12;D21,D22],ts);
P1.u = {'w','u'};
P1.y = {'e','y'};

P2 = ss(S2{1},[B1,S2{2}],[-S2{3};-S2{3}],[D11,D12;D21,D22],ts);
P2.u = {'w','u'};
P2.y = {'e','y'};

P3 = ss(S3{1},[B1,S1{2}],[-S3{3};-S3{3}],[D11,D12;D21,D22],ts);
P3.u = {'w','u'};
P3.y = {'e','y'};

P4 = ss(S4{1},[B1,S4{2}],[-S4{3};-S4{3}],[D11,D12;D21,D22],ts);
P4.u = {'w','u'};
P4.y = {'e','y'};


% Filter is a first order filter 
% [b,a] = butter(2,1/10); 
% W1 = zpk([],[-0.1,-0.1],0.01);
W1 = tf([1],[10,1]);
W1 = ss(W1);
W1 = c2d(W1,ts);

W1.u = {'e'};
W1.y = {'e_w'};

Paug1 = connect(P1,W1,{'w','u'},{'e_w','y'});
Paug2 = connect(P2,W1,{'w','u'},{'e_w','y'});
Paug3 = connect(P3,W1,{'w','u'},{'e_w','y'});
Paug4 = connect(P4,W1,{'w','u'},{'e_w','y'});

% Paug = c2d(Paug,ts);

VertexSystems = {Paug1,Paug2,Paug3,Paug4};

VertexController = dgshinf(VertexSystems,1,1,ops);

%% Investigate Performance of vertex controllers

Paug = Paug1; 
C = VertexController{1};

C = ss(C(1:3,1:3),C(1:3,4),C(4,1:3),C(4,4),ts);
C.u = {'y'};
C.y = {'u'};

CL = connect(Paug,C,{'w'},{'e_w'});

figure;
stepplot(CL)


figure;
bodeplot(CL)