clc
clear
close all

% Load identified Vertex Systems
load('VertexSystemsTestLPV.mat') 
load('VertexController_TestLPV.mat') 

% Frozen System 1

P1 = ss(A1,B1,C1,[0],-1);

A_K = VertexController{1}(1:2,1:2);
B_K = VertexController{1}(1:2,3);
C_K = VertexController{1}(3,1:2);
D_K = VertexController{1}(3,3);

C1 = ss(A_K,B_K,C_K,D_K,-1);

%%
load('VertexController_TestLPV_dhinflmi.mat')
A_K = K(1:2,1:2);
B_K = K(1:2,3);
C_K = K(3,1:2);
D_K = K(3,3);

C2 = ss(A_K,B_K,C_K,D_K,-1);

%%
figure;
hold on
bode(P1)
hold off

%%
figure
loops = loopsens(P1,C1); 
bode(loops.Si,'r',loops.Ti,'b',loops.Li,'g')
legend('Sensitivity','Complementary Sensitivity','Loop Transfer')

figure
loops = loopsens(P1,C2); 
bode(loops.Si,'r',loops.Ti,'b',loops.Li,'g')
legend('Sensitivity','Complementary Sensitivity','Loop Transfer')