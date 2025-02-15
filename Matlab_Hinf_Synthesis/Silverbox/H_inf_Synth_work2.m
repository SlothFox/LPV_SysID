clear
clc

% Add YALMIP to path
addpath(genpath('..\YALMIP-master'))  
% Linux:
% addpath '\mosek-linux\9.2\toolbox\r2015a'

% Windows
addpath 'C:\Program Files\Mosek\9.2\toolbox\R2015a'
ops = sdpsettings('solver','mosek');

% ops = []

% Load identified Vertex Systems
load('VertexSystemsTestLPV.mat') 

S1 = {A1,B1,C1};

% Specify dimensions of problem
nx = 2;
nw = 1;
ny = 1;
nu = 1;
nq = 1;
k = 2;

% In this control problem, certain matrices are fixed
B1  = zeros(nx,nw);
D11 = -eye(nq,nw);
D12 = zeros(nq,nu);
D21 = -eye(ny,nw);
D22 = zeros(ny,nu);

%% Solvability conditions
r = sdpvar(1,1);

R = sdpvar(nx,nx,'symmetric');
S = sdpvar(nx,nx,'symmetric');

LMI = [[S,eye(nx);eye(nx),R] >= 0];

VertexSystems = {'S1','S2','S3','S4'};

for vertex = [1:1]
    
    system = eval(VertexSystems{vertex});
    
    A  = double(system{1});
    B2 = double(system{2});
    C1 = double(system{3});
    C2 = double(system{3});
    
    NR = null([B2',D12']);
    NS = null([C2,D21]);

    MR = [A*R*A'-R,   A*R*C1',               B1;
          C1*R*A',     -r*eye(nq)+C1*R*C1',  D11;
          B1',        D11',                  -r*eye(nw)];

    MS = [A'*S*A-S,   A'*S*B1,               C1';
         B1'*S*A,     -r*eye(nw)+B1'*S*B1,   D11';
         C1,          D11,                   -r*eye(nq)];
 
    NR = [NR,           zeros(3,1);
          zeros(1,2),   eye(1) ];
    
    NS = [NS,       zeros(3,1);
            zeros(1,2), eye(1) ];
        
    LMI = [LMI,[NR'*MR*NR<=0]];
    LMI = [LMI,[NS'*MS*NS<=0]];
    
end

optimize(LMI,r,ops)

%% Reconstruct controller

r = double(r);
R = double(R);
S = double(S);

% Calculate rank of controller
% [U,Sigma,V] = svd(eye(nx)-R*S);



% Try another way to calculate Xcl

X1 = S;
X2 = (X1-inv(R));
[U,T] = schur(X2);
X2 = U*sqrt(T)*U';

X = [X1,X2;X2',eye(nx)];
% Solve one LMI for each vertex controller

VertexController= {};

for vertex = [1:1]
    
    system = eval(VertexSystems{vertex});
    
    theta = sdpvar(k+nu,k+ny,'full');
%     r = sdpvar(1,1);
    
    A  = double(system{1});
    B2 = double(system{2});
    C1 = double(system{3});
    C2 = double(system{3});

    A0 = [A,zeros(nx,k);zeros(k,nx),zeros(k,k)];
    B0 = [B1;zeros(k,nw)];
    C0 = [C1, zeros(nq,k)];
%     D11 = D11;
    
    BB = [zeros(nx,k) B2;eye(k) zeros(k,nu)];
    CC = [zeros(k,nx), eye(k); C2, zeros(ny,k)];
    DD12 = [zeros(nq,k), D12];
    DD21 = [zeros(k,nw); D21];
   
    Acl = A0 + BB*theta*CC;
    Bcl = B0 + BB*theta*DD21;
    Ccl = C0 + DD12*theta*CC;
    Dcl = D11+ DD12*theta*DD21;
    
    P = [BB' zeros(k+nu,nx+k) zeros(k+nu,nw) DD12'];
    Q = [zeros(k+ny,nx+k) CC DD21 zeros(k+ny,nq)];    
      

    Psi = [-inv(X),        A0,               B0,                 zeros(nx+k,ny);
           A0',           -X,               zeros(nx+k,nw),     C0';
           B0',           zeros(nw,nx+k),   -r*eye(nw)          D11';
           zeros(nq,nx+k),C0,               D11,                -r*eye(ny)];  
    
    LMI = [[Psi+P'*theta*Q + Q'*theta'*P] <= 0];

    optimize(LMI,[],ops)

    theta = double(theta);
    
    VertexController{vertex} = double(theta);
    
end


save('VertexController_TestLPV','VertexController')