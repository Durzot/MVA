## Author: Yoann Pradat <pradaty@Yoanns-MacBook-Pro.local>
## Created: 2018-10-18

function v = gradn (u)
   [M,N] = size(u);
   v = zeros(M-1, N-1);
   for k=1:(M-1)
        for l=1:(N-1)
            v(k,l) = sqrt((u(k+1,l)-u(k,l))^2+(u(k, l+1)-u(k,l))^2);
        end
    end
endfunction
