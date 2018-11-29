function retval = count_zeros (x)
    retval = 0;
    for i=1:(length(x)-1)
        if x(i)*x(i+1) < 0
            retval = retval +1;
        endif
    endfor
endfunction
