function [p,s] = perdecomp(u)
    [ny, nx] = size(u);
    u = double(u);
    X = 1:nx; Y=1:ny;
    v = zeros(nx, ny);
    v(1,X) = u(1,X)-u(ny,X);
    v(ny,X) = -v(1,X);
    v(Y,1) = v(Y,1) + u(Y,1) - u(Y,nx);
    v(Y,nx) = v(Y,nx) -u(Y,1) + u(Y,nx);
    fx = repmat(cos(2.*pi*(X-1)/nx),ny,1);
    fy = repmat(cos(2.*pi*(Y'-1)/ny),1,nx);
    fx(1,1) = 0.;
    s = real(ifft2(fft2(v)*0.5./(2.-fx-fy)));
    p = u-s;
 
