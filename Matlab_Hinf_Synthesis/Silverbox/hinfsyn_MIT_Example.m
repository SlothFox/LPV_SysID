clear
clc
% close all

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

W1 = tf([10],[10 ,1]);
W1_ss = ss(W1);
W1_ss.u = {'e'};
W1_ss.y = {'e_w'};


Paug = connect(P1,W1_ss,{'w','u'},{'e_w','y'});

% Fix cheap control problem
Paug.D(1,2)=0.01;
Paug = c2d(Paug,ts);

opts = hinfsynOptions('Method','LMI','Display','on');
[K2,CL2,gamma2,info2] = hinfsyn(Paug,1,1,opts);

figure;
bodeplot(CL2)
figure;
stepplot(CL2)