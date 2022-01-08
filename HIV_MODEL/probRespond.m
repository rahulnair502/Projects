function p = probRespond(t, type)
%this code determines the probabilty of cellullar response based off of inputs
if type == 'a'
    p = .8;
elseif  type == 'b'
    if t <= 400
        p =.8;
    else 
        p =.4;
    end
    
else
   if t <= 300
       p =1;
   else
   p = interp1(300:(12*52), 1:1/(300 - 12*52):0, t);
   end
end
end
