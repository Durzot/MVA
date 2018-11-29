#############
# Exercice 16
#############

fig = figure;
orders = [-3, 0,1,3,5,7,9];

u = double(imread("crop_bouc.pgm"));

for n = orders    
    v = fzoom(u, 16, n);
    
    subplot(1,2,1), imshow(u, []);
    title("u");
    subplot(1,2,2), imshow(v, []);
    title(['v order ',num2str(n)]);
    drawnow;
    frame = getframe(fig);
    str = ["v_bouc_order", num2str(n), ".png"];
    imwrite(frame2im(frame), str);
endfor

u = double(imread("crop_cameraman.pgm"));

for n = orders    
    v = fzoom(u, 16, n);
    
    subplot(1,2,1), imshow(u, []);
    title("u");
    subplot(1,2,2), imshow(v, []);
    title(['v order ',num2str(n)]);
    drawnow;
    frame = getframe(fig);
    str = ["v_cameraman_order", num2str(n), ".png"];
    imwrite(frame2im(frame), str);
endfor


