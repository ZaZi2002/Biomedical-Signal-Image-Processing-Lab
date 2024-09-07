clc
close all
clear all

%% Q1
image1 = imread('t2.jpg');

%%%%% Gaussian noise
sigma = 15^(0.5);
gaussian_noise = sigma*randn(256,256,3);

noisy_img1 = double(image1) + gaussian_noise;

figure('Name',"Gaussian noise");
subplot(2,2,1);
imshow(image1(:,:,1));
title("Main image");

subplot(2,2,2);
imshow(noisy_img1(:,:,1),[]);
title("Signal + Gaussian noise");

%%%%% Normalized kernel
kernel1 = zeros(256,256);
kernel1(127:130,127:130) = 1/16;

filtered_image1 = fftshift(ifft2(fft2(noisy_img1(:,:,1)).*fft2(kernel1)));
filtered_image1 = filtered_image1/max(max(filtered_image1));

subplot(2,2,3);
imshow(abs(filtered_image1));
title("Kernel1 filter");

%%%%% Imgaussfilter
filtered_image2 = imgaussfilt(noisy_img1(:,:,1),1);

subplot(2,2,4);
imshow(abs(filtered_image2),[]);
title("imgaussfilt");

