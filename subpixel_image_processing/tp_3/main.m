############
# Exercice 7
############

# Question 1

u = double(imread('lena.pgm'));
f = fft2(u);
imshow2(f);

imshow2(abs(f));
imshow2(abs(f), []);
imshow2(normsat(abs(f), 1),);
imshow2(normsat(fftshift(abs(f)),1));


############
# Exercice 8
############

# Question 3
u = double(imread('lena.pgm'));
imshow2(u, []);
v = fshift(u, -30, -30);
figure();
imshow2(v, []);
