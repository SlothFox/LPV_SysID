clear
clc

a=0; b1=1; b2=2; c1=1; d11=0; d12=0; c2= 1; d21=1; d22=0;
P=ltisys(a,[b1 b2],[c1;c2],[d11 d12;d21 d22])

[g,K,x1,x2,y1,y2] = hinflmi(P,[1,1])