#############
# Exercice 11
#############

# Question 1
u = double(imread("lena.pgm"));
[p,s] = perdecomp(u);

subplot(2,2,1), imshow(u, [])
title("Original image")
subplot(2,2,2), imshow(p+s, [])
title("p+s")
subplot(2,2,3), imshow(p, [])
title("p")
subplot(2,2,4), imshow(s, [])
title("s")

# Question 2

subplot(2,2,1), imshow(kron(ones(2,2),u), []); 
title("u")
subplot(2,2,2), imshow(kron(ones(2,2),p), []); 
title("p")
subplot(2,2,3), imshow(kron(ones(2,2),s), []); 
title("s")

# Question 3

fu = fft2(u);
fp = fft2(p);
fs = fft2(s);

subplot(1,2,1), imshow(normsat(fftshift(log(abs(fu))), 1));
title("log-module fft2(u)")
subplot(1,2,2), imshow(normsat(fftshift(angle(fu)), 1));
title("phase fft2(u)")

subplot(1,2,1), imshow(normsat(fftshift(log(abs(fp))), 1));
title("log-module fft2(p)")
subplot(1,2,2), imshow(normsat(fftshift(angle(fp)), 1));
title("phase fft2(p)")

subplot(1,2,1), imshow(normsat(fftshift(log(abs(fs))), 1));
title("log-module fft2(s)")
subplot(1,2,2), imshow(normsat(fftshift(angle(fs)), 1));
title("phase fft2(s)")

# Question 4

imshow(fsym2(u), []);
ffsym2u = fft2(fsym2(u));

subplot(1,2,1), imshow(normsat(fftshift(log(abs(ffsym2u))), 1));
title("log-module fft2(fsym2(u))")
subplot(1,2,2), imshow(normsat(fftshift(angle(ffsym2u)), 1));
title("phase fft2(fsym2(u))")





