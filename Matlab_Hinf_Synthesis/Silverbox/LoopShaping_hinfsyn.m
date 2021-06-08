% Plant defined according to https://de.mathworks.com/help/robust/ref/lti.hinfsyn.html#mw_3e327dc9-7ae1-4f2b-b670-9e271011dd8e

clear
clc

A = [0 1;-1,-1];
B = [0;1];
C = [1,0];
D = [0];

cont_sys = ss(A,B,C,D);

% In this control problem, certain matrices are fixed

B1 = [0;0];
B2 = B;
C1 = C;
C2 = C;
D11 = [1];
D12 = [0];
D21 = [1];
D22 = [0];

% ts=0.01;

opts = hinfsynOptions('Method','LMI','Display','on');

P = ss(A,[B1,B2],[-C1;-C2],[D11,D12;D21,D22]);


[K,CL,gamma,info] = hinfsyn(P,1,1,opts);
stepplot(CL)

%% Try full sate measurement
A = [1,0.2;-0.5,0.9];
B1 = [0;0];
B2 = [0;1];
C = [1,0;0,1];
D11 = [1;0];
D12 = [1;0];
D21 = D11;
D22 = D12;


ts=-1;

P1 = ss(A,[B1,B2],[-C;-C],[D11,D12;D21,D22],ts);

opts = hinfsynOptions('Method','LMI','Display','on');

[K,CL,gamma,info] = hinfsyn(P1,2,1,opts);
stepplot(CL)

