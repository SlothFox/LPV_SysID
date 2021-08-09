function [VertexController] = dgshinf(VertexSystems,num_meas,num_cont,ops)
%DGSHINF Synthesis for discrete time gain schedlued H_inf controller
%   Detailed explanation goes here

% Extract Problem Dimensions from first Vertex System and check if all 
% Vertex Systems have the same dimensions

VertexSystem = VertexSystems{1};
A  = VertexSystem.A;
B1 = VertexSystem.B(:,1:end-num_cont);
B2 = VertexSystem.B(:,end-num_cont+1:end);
C1 = VertexSystem.C(1:end-num_meas,:);
C2 = VertexSystem.C(end-num_meas+1:end,:);
D11 = VertexSystem.D(1:end-num_meas,1:end-num_cont);
D12 = VertexSystem.D(1:end-num_meas,end-num_cont+1);
D21 = VertexSystem.D(end-num_meas+1:end,1:end-num_cont);
D22 = VertexSystem.D(end-num_meas+1:end,end-num_cont+1);

nx = size(A,1);
nw = size(B1,2);
nu = size(B2,2);
nq = size(C1,1);
ny = size(C2,1);

k = nx; % consider full rank controllers for now

%% Solvability conditions
r = sdpvar(1,1);                % gamma        

R = sdpvar(nx,nx,'symmetric');  % R
S = sdpvar(nx,nx,'symmetric');  % S

LMI = [[R,eye(nx);eye(nx),S] >= 0];

for vertex = 1:length(VertexSystems)
    
    VertexSystem = VertexSystems{vertex};
    A  = VertexSystem.A;
    B1 = VertexSystem.B(:,1:nw);
    B2 = VertexSystem.B(:,nw+1:end);
    C1 = VertexSystem.C(1:1:nq,:);
    C2 = VertexSystem.C(nq+1:end,:);
    D11 = VertexSystem.D(1:nq,1:nw);
    D12 = VertexSystem.D(1:nq,nw+1:end);
    D21 = VertexSystem.D(ny+1:end,1:nw);
    D22 = VertexSystem.D(nq+1:end,nw+1:end);
    
    % Get shorthands for dimensions for convenience


    NR = null([B2',D12']);
    NS = null([C2,D21]);

    MR = [A*R*A'-R,   A*R*C1',               B1;
          C1*R*A',     -r*eye(nq)+C1*R*C1',  D11;
          B1',        D11',                  -r*eye(nw)];

    MS = [A'*S*A-S,   A'*S*B1,               C1';
         B1'*S*A,     -r*eye(nw)+B1'*S*B1,   D11';
         C1,          D11,                   -r*eye(nq)];
 
    NR = [NR,                                       zeros(size(NR,1),size(MR,2)-size(NR,2) );
          zeros(size(MR,1)-size(NR,1),size(NR,2)),  eye(size(MR,1)-size(NR,1),size(MR,2)-size(NR,2))];

    
    NS = [NS,                                       zeros(size(NS,1),size(MS,2)-size(NS,2) );
          zeros(size(MS,1)-size(NS,1),size(NS,2)),  eye(size(MS,1)-size(NS,1),size(MS,2)-size(NS,2))];
        
    LMI = [LMI,[NR'*MR*NR<=0]];
    LMI = [LMI,[NS'*MS*NS<=0]];
    
end

optimize(LMI,r+trace(R)*1e-08+trace(S)*1e-08,ops)

%% Reconstruct Controller from R and S

r = double(r);
R = double(R);
S = double(S);

% Calculate rank of controller
% [U,Sigma,V] = svd(eye(nx)-R*S);

% Calculate Xcl by Pazo 2015
X1 = S;
X2 = X1-inv(R);
[U,T] = schur(X2);
X2 = U*sqrt(T)*U';

Xcl = [X1,X2;X2',eye(nx)];

VertexController= {};

for vertex = 1:length(VertexSystems)
    
    VertexSystem = VertexSystems{vertex};
    A  = VertexSystem.A;
    B1 = VertexSystem.B(:,1:nw);
    B2 = VertexSystem.B(:,nw+1:end);
    C1 = VertexSystem.C(1:1:nq,:);
    C2 = VertexSystem.C(nq+1:end,:);
    D11 = VertexSystem.D(1:nq,1:nw);
    D12 = VertexSystem.D(1:nq,nw+1:end);
    D21 = VertexSystem.D(ny+1:end,1:nw);
    D22 = VertexSystem.D(nq+1:end,nw+1:end);

% % rescale the plant data to keep the conditioning of M,N small
% sclf=1;
% sclx=sqrt(norm(R,1));
% if sclx > 10
%    C1=C1/sclx; D11=D11/sclx; D12=D12/sclx;
%    r=r/sclx;  R=R*sclx; sclf=sclf*sclx;
% end
% scly=sqrt(norm(S,1));
% if scly > 10
%    B1=B1/scly; D11=D11/scly; D21=D21/scly;
%    r=r/scly;  S=S*scly;  sclf=sclf*scly;
% end    
    
    
    
    theta = sdpvar(k+nu,k+ny,'full');

    
%     A0 = [A,zeros(nx,k);zeros(k,nx),zeros(k,k)];
%     B0 = [B1;zeros(k,nw)];
%     C0 = [C1, zeros(nq,k)];
%     
%     BB = [zeros(nx,k) B2;eye(k) zeros(k,nu)];
%     CC = [zeros(k,nx), eye(k); C2, zeros(ny,k)];
%     DD12 = [zeros(ny,k), D12];
%     DD21 = [zeros(k,nw); D21];
%    
%     Acl = A0 + BB*theta*CC;
%     Bcl = B0 + BB*theta*DD21;
%     Ccl = C0 + DD12*theta*CC;
%     Dcl = D11+ DD12*theta*DD21;
% 
%     Mcl = [-inv(Xcl),        Acl,               Bcl,            zeros(nx+k,ny);
%            Acl',            -Xcl,              zeros(nx+k,nw), Ccl';
%            Bcl',            zeros(nw,nx+k),   -r*eye(nw)       Dcl';
%            zeros(nq,nx+k),  Ccl,               Dcl,             -r*eye(ny)];  
%     
%     LMI = [[Mcl] <= 0];
% 
%     optimize(LMI,[],ops)
% 
%     theta = double(theta);
%     
%     VertexController{vertex} = theta;
    
    A0 = [A,zeros(nx,k);zeros(k,nx),zeros(k,k)];
    B0 = [B1;zeros(k,nw)];
    C0 = [C1, zeros(nq,k)];
    
    BB = [zeros(nx,k) B2;eye(k) zeros(k,nu)];
    CC = [zeros(k,nx), eye(k); C2, zeros(ny,k)];
    DD12 = [zeros(nq,k), D12];
    DD21 = [zeros(k,nw); D21];
   
   
    P = [BB' zeros(k+nu,nx+k) zeros(k+nu,nw) DD12'];
    Q = [zeros(k+ny,nx+k) CC DD21 zeros(k+ny,nq)];    
      

    Psi = [-inv(Xcl),        A0,               B0,                 zeros(nx+k,ny);
           A0',           -Xcl,               zeros(nx+k,nw),     C0';
           B0',           zeros(nw,nx+k),   -r*eye(nw)          D11';
           zeros(nq,nx+k),C0,               D11,                -r*eye(ny)];  
    
    LMI = [[Psi+P'*theta*Q + Q'*theta'*P] <= 0];

    optimize(LMI,[],ops)

    theta = double(theta);
    
    VertexController{vertex} = theta;    
    
    
end


end

