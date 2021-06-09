clear
clc
close all
%% Without weighting

% Define state space matrices of plant

% A = [-1];
% B = [1];
% C = [1];
% D = [0];

% Define state space matrices of general control structure

% B1 = 0;
% B2 = B;
% C1 = -C;
% C2 = -C;
% D11 = [1];
% D12 = [0];
% D21 = [1];
% D22 = [0];

% Define Plant
% P1 = ss(A,[B1,B2],[C1;C2],[D11,D12;D21,D22]);

% Synthesize controller with LMI

% opts = hinfsynOptions('Method','LMI','Display','on');
% [K,CL,gamma,info] = hinfsyn(P1,1,1,opts);

% Inspect step response of closed loop system, error e should decrease
% figure;
% bodeplot(CL)

%% Now introduce weighting of e

A = [-1 0;-1 -10];
B1 = [0;1];
B2 = [1;0];
C1 = [0 10];
C2 = [-1 0];
D11 = [0];
D12 = [0];
D21 = [1];
D22 = [0];

P2 = ss(A,[B1,B2],[C1;C2],[D11,D12;D21,D22]);

opts = hinfsynOptions('Method','LMI','Display','on');
[K1,CL1,gamma1,info1] = hinfsyn(P2,1,1,opts);

figure;
bodeplot(CL1)
figure;
stepplot(CL1)

%% Now figure how to apply filter in a simple way

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

W1 = tf([10],[10 ,1]);
W1_ss = ss(W1);
W1_ss.u = {'e'};
W1_ss.y = {'e_w'};


Paug = connect(P1,W1_ss,{'w','u'},{'e_w','y'});


opts = hinfsynOptions('Method','LMI','Display','on');
[K2,CL2,gamma2,info2] = hinfsyn(Paug,1,1,opts);

figure;
bodeplot(CL2)
figure;
stepplot(CL2)

% W1_ss = ltisys(W1_ss.A,W1_ss.B,W1_ss.C,W1_ss.D);
% Paug = smult(P1,sdiag(W1_ss,1));
