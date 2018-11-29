function retval = ftm (x)
    rho=abs(x);
    if (rho >= 1)
        retval = 0;
    else    
        retval=acos(rho)-rho*sqrt(1-rho^2);
endfunction

