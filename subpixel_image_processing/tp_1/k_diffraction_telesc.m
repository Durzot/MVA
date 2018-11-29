function [y] = k_diffraction_telesc (r, C, ep)
    jr = besselj(1, r);
    jr_ep = besselj(1,ep .* r);
    y = C * (2 .* jr ./ r) .** 2 -  C * (2 * ep .* jr_ep ./ r) .** 2 - 8*C*(ep.* jr .* jr_ep) ./ r^2; 
endfunction
