#############
# Exercise 16
#############

x0 = 50.5;
y0 = 50.5;

u = double(imread("bouc.pgm"));
v = ffttrans(u, x0, y0);

subplot(1, 2, 1), imshow(u, []);
title("u")
imshow(v(1:60, 1:60), []);
title("trans u 50.5")
subplot(1, 1, 1), imshow(u-v, []);
title("u-v")

# Avoid artefacts on borders
# Shift p, shift s by closest integer
[p, s] = perdecomp(u);
transp = ffttrans(p, x0, y0);
transs = ffttrans(s, x0, y0);
transs_r = ffttrans(s, round(x0), round(y0));

subplot(1,2,1), imshow((transp+transs)(1:60,1:60), []);
title("trans p 50.5 + trans s 50.5");
subplot(1,2,2), imshow((transp+transs_r)(1:60, 1:60), []);
title("trans p 50.5 + trans s 50");

