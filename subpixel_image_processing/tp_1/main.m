# Question 1
lr = linspace(0.01, 10, 100);
lf = arrayfun(@(r) k_diffraction(r, 1), lr);
plot(lr, min(0.1, lf));

# Question 2 
[x, y] = meshgrid(linspace(-10, 10, 100), linspace(-10, 10, 100));
r = sqrt(x .^ 2 + y .^ 2);
C = 1;
imshow(arrayfun(@(er) k_diffraction(er, C), r));
imshow(arrayfun(@(er) k_diffraction(er, C), r), [0,0.01]);

# Question 3
# Profil radial
lr = linspace(0.01, 2, 100);
lf = arrayfun(@(r) ftm(r), lr);
plot(lr,lf);

# Image bidimensionnelle
[x, y] = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100));
r = sqrt(x .^ 2 + y .^ 2);
imshow(arrayfun(@(er) ftm(er), r));
imshow(arrayfun(@(er) k_diffraction(er, 1), r), [0,0.01]);

[x, y] = meshgrid(linspace(-100, 100, 400), linspace(-100, 100, 400));
r = sqrt(x .^ 2 + y .^ 2);
coor = arrayfun(@(er) k_diffraction(er, 1), r);
v = fftshift(abs(fft2(coor)));
imshow(v, [0,max(max(v))]);

# Question 4
n_pts = 100;
n_pts_search = 100;
lr = linspace(0.01, 10, n_pts);

 We localised optimal distance in [1,3]
d_search=linspace(3,1, n_pts_search); 
i=0;
n_zeros_x = 4;

# First point spread function
K1 = k_diffraction(lr,1);
[minval, argmin] = min(K1(1:n_pts/2));
ra = lr(argmin);

fprintf("Rayon d'airy:f\n", ra);
plot(lr, K1)

while n_zeros_x>=2
    i=i+1;
    d=d_search(i);
    rd=abs(lr-d);
    K2=k_diffraction(rd,1);

     Sum PFS distant from d
    K=K1+K2;
    diff_K = diff(K);
    
     Count number of zero crossings in the range 0 to d
     If this is higher than 1, there are at least 2 extrema
    
    [minval, argmin] = min(abs(lr-d));
    f_idx = 1;
    l_idx = argmin-1;
    n_zeros_x = count_zeros(diff_K(f_idx:l_idx));
    hold on
    plot(lr,K)
end
set(gca, "fontsize", 12)
text(1, 1.2,["Critical dist. frac of ra: "  num2str(d/ra)], "fontsize", 24, "linewidth", 4)
fprintf("Critical distance as fraction of ra: f\n", d/ra)

# Question 6
# Image bidimensionnelle
[x, y] = meshgrid(linspace(-100, 100, 400), linspace(-100, 100, 400));
r = sqrt(x .^ 2 + y .^ 2);
coor = arrayfun(@(er) k_diffraction_telesc(er, 1, 1/4), r);
v = fftshift(abs(fft2(coor)));
imshow(v, [0,max(max(v))]);

