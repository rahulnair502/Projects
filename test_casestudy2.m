t = 12*20;
k = 60;
%represent constant probRespond, stepwise, and linearly decreasing
a = 'a';
b= 'b';
c = 'c';

%counts the number of occurences and plots them
%last inputs are therapy status, type, and Ranklevel, this can be
%zero i therapy staus is false. Last input is time
grids = infection(k,.99, 10^-5, .05,false, 'c' ,1,t);
r = showGraphs(grids);
movie(r)
