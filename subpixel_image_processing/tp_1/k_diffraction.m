function [y] = k_diffraction (r, C)
    jr = besselj(1, r);
    y = C * (2 .* jr ./ r) .** 2;
endfunction
