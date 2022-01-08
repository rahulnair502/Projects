function grids = infection(n, probReplace, probInfect,probHIV,therapy,type,rankLevel, t)
%infection simulation
global A11 A12 A13 A14 A2 healthy dead
A11 = 2; A12 = 3; A13 =4; A14 = 5; A2 =6; dead = 1; healthy = 0;
body = initCellGrid( n,probHIV );
grids = zeros(n, n, t + 1); % initialize the 3-D grids to zero
grids(:, :, 1) = body; % initialize the first 2-D lattice of grids to initial forest lattice
for i = 2:(t + 1)
bodyExtended = extCellGrid(body);
body = applyExtended(bodyExtended, probReplace, probInfect,therapy, i,type,rankLevel);
grids(:, :, i) = body; % updating the ith lattice of 3-D grids
end