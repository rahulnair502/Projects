function newSite = spreadTherapy(site, N, E, S, W,NE,SE, NW, SW, probReplace, probInfect ,t,type, rankLevel)
% spread - Function to return the value of a site at the next time step based on values at site, N, E, S, W
% This is the function with the effect of drug therapy
global  A11 A12 A13 A14 A2 healthy dead 
if site == A11
    newSite = A12;
elseif site == A12
    newSite = A13;
elseif site == A13
    newSite = A14;
elseif site == A14
    newSite = A2;
elseif site == A2
    newSite = dead;
elseif site == dead 
    if rand() < probReplace
        newSite = healthy;
    else
        newSite = dead;
    end
elseif site == healthy
   
    if (N== A11 || N== A12 || N== A13 ||N== A14 || E== A11 || E== A12 ||...
            E== A13 || E== A14 ||  S== A11 || S== A12 || S== A13 || S== A14...
            || W== A11 || W== A12 || W== A13 ||W== A14 ||...
            NE== A11 || NE== A12 || NE== A13 ||NE== A14 || SE== A11 || SE== A12 ||...
            SE== A13 || SE== A14 ||  SW== A11 || SW== A12 || SW== A13 || SW== A14...
            || NW== A11 || NW== A12 || NW== A13 ||NW== A14) && (rand()< (1 - probRespond(t,type)) * rankLevel/8) && (t >300)
        
        newSite = A11;
    
   
       
    elseif (N== A11 || N== A12 || N== A13 ||N== A14 || E== A11 || E== A12 ||...
            E== A13 || E== A14 ||  S== A11 || S== A12 || S== A13 || S== A14...
            || W== A11 || W== A12 || W== A13 ||W== A14 ||...
            NE== A11 || NE== A12 || NE== A13 ||NE== A14 || SE== A11 || SE== A12 ||...
            SE== A13 || SE== A14 ||  SW== A11 || SW== A12 || SW== A13 || SW== A14...
            || NW== A11 || NW== A12 || NW== A13 ||NW== A14) && (t <=300)
        
        newSite = A11;
       
    else
        
        counter = 0;
        if N == A2
            counter = counter +1;
            
        end
         if E == A2
            counter = counter +1;
            
         end
         if S == A2
            counter = counter +1;
         end
         if W == A2
            counter = counter +1;
            
         end
        
         if NE == A2
            counter = counter +1;
        end
         if SE == A2
            counter = counter +1;
         end
         if SW == A2
            counter = counter +1;
            
         end
         if NW == A2
            counter = counter +1;
         end
         
    if (counter >= 3) || (rand <= probInfect)
        newSite = A11;
        else
        newSite = healthy;
        
    end
    end
    else
        if rand <= probInfect
            newSite = A11;
        elseif rand < probReplace
        newSite = healthy;
        else 
            newSite = dead;
        end
end
  
end