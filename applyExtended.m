function newLat = applyExtended(latExt,probReplace, probInfect, therapy,t, type,rankLevel)
% APPLYEXTENDED - Function to apply Moore's Neighborhood
n = size(latExt, 1) - 2;
newLat = zeros(n);
for j = 2:(n + 1)
for i = 2:(n + 1)
site = latExt (i, j);
N = latExt (i - 1, j);
E = latExt (i, j + 1);
S = latExt (i + 1, j);
W = latExt (i, j - 1);
NE = latExt (i - 1, j+1);
SE = latExt (i+1, j + 1);
SW = latExt (i + 1, j-1);
NW = latExt (i-1, j - 1);
if therapy == false
    newLat(i - 1, j - 1) = spread(site, N, E, S, W,NE,SE, NW, SW, probReplace, probInfect);
else 
    newLat(i - 1, j - 1) = spreadTherapy(site, N, E, S, W,NE,SE, NW, SW, probReplace, probInfect,t,type,rankLevel);
end
end
end