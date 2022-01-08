function  body = initCellGrid(n, probHIV)
%INIT returns an n-by-n grid of values?
% infected A1, infected A2, healthy, dead
% prohHIV, probability a cell has HIV
% probHIV is the probability that a cell has HIV
global A11  healthy  % healthy = 0; A1 = 1; A2 = 2; 
%%% initialize forest
per = rand(n);
healthyorNot = (per < probHIV); % =1 if A1, 0 otherwise
healthies = 1 - healthyorNot;
body  =  healthies * healthy + A11*healthyorNot;
end