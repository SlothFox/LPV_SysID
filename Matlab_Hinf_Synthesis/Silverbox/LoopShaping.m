% clear
% clc

load('VertexSystemsSilverbox.mat') 


%%

G_1 = ss(S1{1},S1{2},S1{3},zeros(1,1));
W1 = makeweight(10,[1 0.1],0.01);
W2 = [];
W3 = [];

% gam_range = [1.021,1.0211];

[K_1,CL_1,gamma_1,info_1] = mixsyn(G_1,W1,W2,W3,gam_range);


S = feedback(1,G_1*K_1);

KS = K_1*S;
T = 1-S;
sigma(S,'b',KS,'r',T,'g',gamma/W1,'b-.')
legend('S','KS','T')
grid

%%

G_2 = ss(S2{1},S2{2},S2{3},zeros(1,1));

W1 = [];
W2 = [];
W3 = [];

gam_range = [1.021,1.0211];

[K_2,CL_2,gamma_2,info_2] = mixsyn(G_2,W1,W2,W3,gam_range);