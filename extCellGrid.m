function extlat = extCellGrid(lat)
% extCellGrid returns extended lattice
extendRows = [lat(end, :); lat; lat(1, :)];
extlat = [extendRows(:, end) extendRows extendRows(:, 1)];