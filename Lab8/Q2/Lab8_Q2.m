clc
close all
clear all

%% Q2
f = imread('t2.jpg');

figure;
subplot(1,3,1);
imshow(f);
title("Main image");

%%%%% Gaussian filter
sigma = 0.6;
h = Gaussian(sigma,[256,256]);

%%%%% Producing blured signal
H = fft2(h);
F = fft2(f(:,:,1));
G = F.*H;
g = fftshift(ifft2(F.*H));
g = g/max(max(g));

subplot(1,3,2);
imshow(abs(g));
title("Filtered image");

%%%%% Reverse filter
F1 = G./H;
f1 = ifft2(F1);
f1 = f1/max(max(f1));

subplot(1,3,3);
imshow(abs(f1),[]);
title("Reverse filtered image");

%%%%% Image + Noise
sigma = 0.001^(0.5);
gaussian_noise = sigma*randn(256,256);
g_new = g + gaussian_noise;
G_new = fft2(g_new);
g_new = g_new/max(max(g_new));

F_new = G_new./H;
f_new = fftshift(ifft2(F_new));
f_new = f_new/max(max(f_new));

figure;
subplot(1,3,1);
imshow(f);
title("Main image");

subplot(1,3,2);
imshow(g_new);
title("Noisy image");

subplot(1,3,3);
imshow(abs(f_new),[]);
title("Reverse filtered noisy image");
