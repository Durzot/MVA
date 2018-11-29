############
# Exercice 1
############

############
# Question 1
############

u = double(imread("room.pgm"))/255;
imshow(u);

lambda=3;
v = u(1:lambda:end, 1:lambda:end);
w = kron(v, ones(lambda));
[ny, nx] = size(u);
imshow([u, w(1:ny, 1:nx)]);

############
# Question 2
############

lambda=4;
v = u(1:lambda:end, 1:lambda:end);
w = kron(v, ones(lambda));
[ny, nx] = size(u);
imshow([u, w(1:ny, 1:nx)]);

############
# Question 3
############
f = zeros(512);
f(190, 50) = 2;
onde = real(ifft2(f));
imshow(onde, []);

onde_ft = abs(fft2(onde));
imshow(fftshift(onde_ft), [0,max(max(onde_ft))], 'xdata', 1:512, 'ydata', 1:512);
axis on;

 Sous échantillonnage facteur 2 onde
lambda=2;
onde_ech = onde(1:lambda:end, 1:lambda:end);
onde_ech_zoom = kron(onde_ech, ones(lambda));
[ny, nx] = size(onde);

 Onde et onde échantillonée
imshow([onde, onde_ech_zoom(1:ny, 1:nx)], []);

 Onde échantillonée et transfo fourier
onde_ech_ft = abs(fft2(onde_ech));
imshow(fftshift(onde_ech_ft), [0,max(max(onde_ech_ft))], 'xdata', 1:512, 'ydata', 1:512);
axis on;


[m, pos] = max(onde_ech_ft(:));
[x,y] = ind2sub(size(onde_ech_ft), pos)

############
# Exercice 2
############

############
# Question 1
############

f = zeros(512);
f(190, 50) = 2;
onde = real(ifft2(f));

 Onde et son carré
imshow([onde./max(max(onde)), onde.^2./max(max(onde.^2))], []);

imshow(fftzoom(onde,2), []);

############
# Question 2
############

g = zeros(1024, 1024);
g(66, 206) = 2;
onde_fftzoom = real(ifft2(g));
imshow(onde_fftzoom, []);

############
# Question 3
############

u = double(imread("nimes.pgm"))/255;
imshow(u);
imshow(gradn(u));



