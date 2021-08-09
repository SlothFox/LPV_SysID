clear
clc

% Add YALMIP to path
addpath(genpath('../YALMIP-master'))  
% Linux:
% addpath '\mosek-linux\9.2\toolbox\r2015a'

% Windows
addpath 'C:\Program Files\Mosek\9.2\toolbox\R2015a'
ops = sdpsettings('solver','mosek');

% ops = []

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
D11 = -eye(nq,nw);
D12 = zeros(nq,nu);
D21 = -eye(ny,nw);
D22 = zeros(ny,nu);

%% Solvability conditions
r = sdpvar(1,1);

R = sdpvar(nx,nx,'symmetric');
S = sdpvar(nx,nx,'symmetric');

LMI = [[R,eye(nx);eye(nx),S] >= 0];

VertexSystems = {'S1','S2'};

for vertex = [1:2]
    
    system = eval(VertexSystems{vertex});
    
    A  = system{1};
    B2 = system{2};
    C1 = system{3};
    C2 = system{3};
    
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

% Calculate Xcl by Apkarian
% [M,N] = qr(eye(nx)-R*S);
% 
% O=[S, eye(nx);
%  N, zeros(k,nx)];
% 
% P = [eye(nx), R;
%     zeros(k,nx), M'];
% 
% Xcl = O*pinv(P);

% Try another way to calculate Xcl

X1 = S;
X2 = X1-inv(R);
[U,T] = schur(X2);
X2 = U*sqrt(T)*U';

Xcl = [X1,X2;X2',eye(nx)];
% Solve one LMI for each vertex controller

VertexController= {};

for vertex = [1:2]
    
    system = eval(VertexSystems{vertex});
    
    theta = sdpvar(k+nu,k+ny,'full');
%     r = sdpvar(1,1);
    
    A  = system{1};
    B2 = system{2};
    C1 = system{3};
    C2 = system{3};

    A0 = [A,zeros(nx,k);zeros(k,nx),zeros(k,k)];
    B0 = [B1;zeros(k,nw)];
    C0 = [C1, zeros(nq,k)];
%     D11 = D11;
    
    BB = [zeros(nx,k) B2;eye(k) zeros(k,nu)];
    CC = [zeros(k,nx), eye(k); C2, zeros(ny,k)];
    DD12 = [zeros(ny,k), D12];
    DD21 = [zeros(k,nw); D21];
   
    Acl = A0 + BB*theta*CC;
    Bcl = B0 + BB*theta*DD21;
    Ccl = C0 + DD12*theta*CC;
    Dcl = D11+ DD12*theta*DD21;

    %    Conti Time ? : 
%     Mcl = [Acl'*Xcl+Xcl*Acl, Xcl*Bcl,    Ccl';
%            Bcl'*Xcl,         -r*eye(nw),   Dcl';
%            Ccl,               Dcl,       -r*eye(ny)   ];

% Try discrete time
    
%     Mcl = [Acl'*Xcl*Acl-Xcl, Acl'*Xcl*Bcl,              Ccl';
%            Bcl'*Xcl*Acl,     Bcl'*Xcl*Bcl-r*eye(nw),      Dcl';
%            Ccl,              Dcl,                       -r*eye(nq)];       

   Mcl = [-inv(Xcl),        Acl,               Bcl,            zeros(nx+k,ny);
           Acl',            -Xcl,              zeros(nx+k,nw), Ccl';
           Bcl',            zeros(nw,nx+k),   -r*eye(nw)       Dcl';
           zeros(nq,nx+k),  Ccl,               Dcl,             -r*eye(ny)];  
    
    LMI = [[Mcl] <= 0];

    optimize(LMI,[],ops)

    theta = double(theta);
    
    VertexController{vertex} = double(theta);
    
end


save('VertexController_SilverBox_Phys','VertexController')