% Plant defined according to https://de.mathworks.com/help/robust/ref/lti.hinfsyn.html#mw_3e327dc9-7ae1-4f2b-b670-9e271011dd8e

clear
clc

% Load identified Vertex Systems
load('VertexSystemsTestLPV.mat') 
% load('VertexController_TestLPV.mat')  

S1 = {double(A1),double(B1),double(C1)};


% Define dimensions of plant
nx = 2;
nw = 1;
ny = 1;
nu = 1;
nq = 1;

% In this control problem, certain matrices are fixed
A_1 = S1{1};
B1  = zeros(nx,nw);
B2_1 = S1{2};
C1_1 = S1{3};
D11 = -eye(nq,nw);
D12 = zeros(nq,nu);
C2_1 = S1{3};
D21 = -eye(ny,nw);
D22 = zeros(ny,nu);



P = ltisys(A_1, [B1,B2_1],[C1_1;C2_1],[D11,D12;D21,D22]);

[g,K,x1,x2,y1,y2] = dhinflmi(P,[1,1]);

clsys = slft(K,P);
splot(clsys,'st')

save('VertexController_TestLPV_dhinflmi','K')

%% inside dhinflmi
r = [1,1];
options=[0 0 0 0];
tol=1e-2; 
gmin=0; 
ubo=1; lbo=1;   % default = specified bounds

% tolerances
macheps=mach_eps;
tolsing=10*sqrt(macheps);
toleig=macheps^(2/3);


% retrieve plant data
[a,b1,b2,c1,c2,d11,d12,d21,d22]=hinfpar(P,r);
na=size(a,1); [p1,m1]=size(d11); [p2,m2]=size(d22);
if ~m1, error('D11 is empty according to the dimensions R of D22'); end


% for numerical stability of the controller computation,
% zero the sing. values of D12 s.t  || B2 D12^+ || > 1/tolsing

[u,s,v]=svd(d12);
abstol=max(toleig*norm(b2,1),tolsing*s(1,1));
ratio=max([s;zeros(1,size(s,2))])./...
      max([tolsing*abs(b2*v);abstol*ones(1,m2)]);
ind2=find(ratio < 1); l2=length(ind2);
if l2 > 0, s(:,ind2)=zeros(p1,length(ind2)); d12=u*s*v'; end

[u,s,v]=svd(d21');
abstol=max(toleig*norm(c2,1),tolsing*s(1,1));
ratio=max([s;zeros(1,size(s,2))])./...
      max([tolsing*abs(c2'*v);abstol*ones(1,p2)]);
ind2=find(ratio < 1); l2=length(ind2);
if l2 > 0, s(:,ind2)=zeros(m1,length(ind2)); d21=v*s'*u'; end



% compute the outer factors

NR=lnull([b2;d12],0,tolsing);
cnr=size(NR,2);
NR=[NR,zeros(na+p1,m1);zeros(m1,cnr) eye(m1)];

NS=rnull([c2,d21],0,tolsing);
cns=size(NS,2);
NS=[NS,zeros(na+m1,p1);zeros(p1,cns) eye(p1)];



% LMI setup

setlmis([]);
R=lmivar(1,[na 1]);       % R  Matrix variable Type 1, one diagonal block, na x na, fully symmetric 
S=lmivar(1,[na 1]);       % S
gm=lmivar(1,[1 1]);       % gamma

aux1=[a;c1]; aux2=[eye(na) zeros(na,p1)];
lmiterm([1 0 0 0],NR);
lmiterm([1 1 1 R],aux1,aux1');
lmiterm([1 1 1 R],aux2',-aux2);
lmiterm([1 1 1 gm],[zeros(na,p1);-eye(p1)],[zeros(p1,na) eye(p1)]);
lmiterm([1 2 1 0],[b1' d11']);
lmiterm([1 2 2 gm],-1,1);

aux1=[a b1]; aux2=[eye(na) zeros(na,m1)];
lmiterm([2 0 0 0],NS);
lmiterm([2 1 1 S],aux1',aux1);
lmiterm([2 1 1 S],aux2',-aux2);
lmiterm([2 1 1 gm],[zeros(na,m1);-eye(m1)],[zeros(m1,na) eye(m1)]);
lmiterm([2 2 1 0],[c1 d11]);
lmiterm([2 2 2 gm],-1,1);

lmiterm([-3 1 1 R],1,1);
lmiterm([-3 2 1 0],1);
lmiterm([-3 2 2 S],1,1);

LMIs=getlmis;


% objective
penalty=max(gmin,1)*1e-8;
Rdiag=diag(decinfo(LMIs,R));
Sdiag=diag(decinfo(LMIs,S));
cc=[zeros(na*(na+1),1) ; 1];
cc([Rdiag;Sdiag])=penalty*ones(2*na,1);

options=[tol 0 1e8 0 options(4)];   % fixed feasibility radius

[gopt,xopt]=mincx(LMIs,cc,options,[],gmin);


if isempty(gopt),
  x1=[]; x2=[]; y1=[]; y2=[];
else
  % X = gamma * inv(R)
  x1=dec2mat(LMIs,xopt,R);
  y1=dec2mat(LMIs,xopt,S);
  gopt=gopt-penalty*(trace(x1)+trace(y1));
  x2=gopt*eye(na);
  y2=gopt*eye(na);
end





