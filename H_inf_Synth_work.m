nx = 
nw =
ny =


r = sdpvar(1,1);

R = sdpvar(3,3,'symmetric');
S = sdpvar(3,3,'symmetric');

LMI = [[R,eye(3);eye(3),S] <= 0];

for 
    
    NR = null([B2',D12']);
    NS = null([C2',D21']);

    MR = [A*R*A'-R,   A*R*C1',               B1;
          C1*R*A',     -r*eye(ny)+C1*R*C1',   D1;
          B1',        D1',                    -r*eye(nw)];

    MS = [A'*S*A-S,   A'*S*B1',               C1';
         B1'*S*A,     -r*eye(ny)+B1'*S*B1,   D1';
         C1,          D1,                    -r*eye(nw)];
 
    NR = [NR,       zeros();
            zeros(), eye() ];
    
    NS = [NS,       zeros();
            zeros(), eye() ];
        
        
    LMI = [LMI,[NR'*MR*NR<=0],[[NS'*MS*NS<=0]]];

optimize(F,r)