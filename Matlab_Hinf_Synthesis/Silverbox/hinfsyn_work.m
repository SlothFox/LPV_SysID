clear
clc

%% Define state space matrices of plant

A = [0 1;-1,-1];
B = [0;1];
C = [1,0];
D = [0];

%% Define state space matrices of general control structure

B1 = [0;0];
B2 = B;
C1 = -C;
C2 = -C;
D11 = [1];
D12 = [0];
D21 = [1];
D22 = [0];

%% Define Plant
P = ss(A,[B1,B2],[C1;C2],[D11,D12;D21,D22]);

%% Synthesize controller with LMI

opts = hinfsynOptions('Method','LMI','Display','on');
[K,CL,gamma,info] = hinfsyn(P,1,1,opts);

%% Inspect step response of closed loop system, error e should decrease
stepplot(CL)