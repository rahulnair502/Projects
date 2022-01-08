function M = showGraphs(graphList)
% SHOWGRAPHS - Function to return movie visualization

map = [1 1 0; % healthy -> yellow
0.1 0.75 0.2; % dead
0.6 0.2 0.1 %infected
];  
colormap(map);
m = size(graphList,3 );
M = moviein(m);
for k = 1:m
    g = graphList(:, :, k);
    image(g + 1)
    axis off
    axis square
    M(k) = getframe;
    
end