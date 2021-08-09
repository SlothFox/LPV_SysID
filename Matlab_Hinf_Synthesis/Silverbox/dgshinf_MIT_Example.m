%% Now figure how to apply filter in a simple way
clear
clc
% close all

% Add YALMIP to path
addpath(genpath('..\YALMIP-master'))  
% Linux:
% addpath '\mosek-linux\9.2\toolbox\r2015a'

% Windows
addpath 'C:\Program Files\Mosek\9.2\toolbox\R2015a'
ops = sdpsettings('solver','mosek');

ts = 0.001;

A = [-1];
B = [1];
C = [1];
D = [0];

% Define state space matrices of general control structure

B1 = 0;
B2 = B;
C1 = -C;
C2 = -C;
D11 = [1];
D12 = [0];
D21 = [1];
D22 = [0];

% Define Plant
P1 = ss(A,[B1,B2],[C1;C2],[D11,D12;D21,D22]);
P1.u = {'w','u'};
P1.y = {'e','y'};

% Filter is a first order filter 

W1 = tf([1],[10 ,1]);
W1_ss = ss(W1);


W1_ss.u = {'e'};
W1_ss.y = {'e_w'};


Paug = connect(P1,W1_ss,{'w','u'},{'e_w','y'});

% Fix cheap control problem
Paug.D(1,2)=0.01;

Paug = c2d(Paug,ts);
VertexController = dgshinf({Paug},1,1,ops);

C = VertexController{1};
C = ss(C(1:2,1:2),C(1:2,3),C(3,1:2),C(3,3),ts);
C.u = {'y'};
C.y = {'u'};

CL = connect(Paug,C,{'w'},{'e_w'});

figure;
stepplot(CL)