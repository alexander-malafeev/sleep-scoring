function [ stagesSym ] = stagesNum2Sym( stages, labels )
% converts numeric representation of the stages to
% symbolic


stagesSym = [];


if ~exist('labels','var') || isempty(labels)
    labels= ['3' '2' '1' 'R' 'W' 'U'];
end

st1=labels(3);
st2=labels(2);
st34=labels(1);
rem=labels(4);
W=labels(5);
art = labels(6);


stagesSym( find(stages==0) ) = W;

stagesSym( find(stages==1) ) = st1;
stagesSym( find(stages==2) ) = st2;
stagesSym( find(stages==3) ) = st34;



stagesSym( find(stages==4) ) = rem;

stagesSym = char(stagesSym);
end