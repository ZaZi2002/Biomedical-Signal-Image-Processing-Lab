%% Q1
clc;
clear;
close all;

im = imread("S1_Q1_utils\t1.jpg"); %Read the image

figure;
imshow(im);
title("Main Image");


im = squeeze(im(:,:,1));
row = 128; %Selected row
midRow_fft = fft(im(row, :)); %FFT for row 128
midRow_fft = fftshift(midRow_fft); %Shifting zero frequency to the center

n = size(im,2);
f_arr = (-n/2:n/2-1) * (2*pi/n);

%Plotting FFT:
figure;
subplot(211);
plot(f_arr, abs(midRow_fft)); title("FFT magnitude for row "+row);
subplot(212);
plot(f_arr, angle(midRow_fft)); title("FFT phase for row "+row);



%FFT for the whole image:
im_double = double(im);
im_fft = fft2(im_double);
im_fft_mag = log10(abs(im_fft));
im_fft_mag = im_fft_mag/max(max(im_fft_mag));


%Plotting FFT:
figure;
subplot(131);
imshow(im);
title("Image");

subplot(132);
imshow(im_fft_mag);
title("Image FFT");

subplot(133);
im_fft_shift = fftshift(im_fft);
im_fft_mag_shift = log10(abs(im_fft_shift));
im_fft_mag_shift = im_fft_mag_shift/max(max(im_fft_mag_shift));


imshow(im_fft_mag_shift);
title("Image FFT after fftshift");



%% Q2

clc;
clear;
close all;

%Creating Image G:
size = 256; %Size of the window
r = 15; %Radius of the circle

[X,Y] = ndgrid(1:size, 1:size); %Creating grid for the window

G = ((X-size/2).^2 + (Y-size/2).^2) <= r^2; %Creating a circle at the center

%Creating Image F:
F = zeros(size, size);
F(100, 50) = 1;
F(120, 48) = 2;

F_conv_G = ifftshift(ifft2(fft2(G) .* fft2(F))); %F * G


figure;
subplot(131);
imshow(G);
title("Image G");

subplot(132);
imshow(F./max(max(F)));
title("Image F");

subplot(133);
imshow(F_conv_G./max(max(F_conv_G)));
title("F*G");



%Reading Image:
im = imread("S1_Q2_utils\pd.jpg");
im = squeeze(im(:,:,1));
im_double = double(im);


figure;
subplot(121);
imshow(im);
title("Main Image");

im_conv_G = ifftshift(ifft2(fft2(G) .* fft2(im_double)));% Image * G

subplot(122);
imshow(im_conv_G./max(max(im_conv_G)));
title("MainImage * G");


%% Q4: 1

clc;
% clear;
% close all;


%Reading Image:
im = imread("S1_Q4_utils\ct.jpg");
im = squeeze(im(:,:,1));
im_double = double(im);
size = size(im,1);

[Y, X] = ndgrid(-size/2+1:size/2, -size/2+1:size/2); %Creating Grid

x0 = 20; %Horizontal shift
y0 = 40; %Vertical shift

kernel = exp(-1i * 2*pi * (x0*X + y0*Y) / size); %Creating shift kernel

%Creating shifted image in Fourier Domain:
im_fft = fftshift(fft2(im_double));
shift_im_fft = im_fft .* kernel;
shift_im = abs(ifft2(shift_im_fft));

%Plotting required results:
figure;

subplot(221);
imshow(im);
title("Main Image");

subplot(222);
imshow(shift_im ./ max(max(shift_im)));
title("Shifted Image");

subplot(212);
plot(abs(kernel));
title("Kernel Magnitude");



%% Q4: 2

clc; clear;
close all;

theta = 30; %Rotation degree

%Loadin Image:
im = imread("S1_Q4_utils\ct.jpg");
im = squeeze(im(:,:,1));
im_double = double(im);
size = size(im,1);


%****************Rotating the Image directly:**********************
rotate_im = imrotate(im, theta); 

figure;

subplot(121);
imshow(im);
title("Main Image");

subplot(122);
imshow(rotate_im);
title("Rotated Image");
%*******************************************************

%***************Comparing Fourier Transforms:******************
figure;

im_fft = fftshift(fft2(im_double));
rotate_im_fft = fftshift(fft2(rotate_im));

subplot(121);
imshow(log10(abs(im_fft)) ./ max(max(log10(abs(im_fft)))));
title("Main Image FFT");

subplot(122);
imshow(log10(abs(rotate_im_fft)) ./ max(max(log10(abs(rotate_im_fft)))));
title("Rotated Image FFT");
%**************************************************************************


%***************Rotating the image in the Fourier Domain:******************
figure;

im_fft = fftshift(fft2(ifftshift(im_double)));
rotate_im_fft = imrotate(im_fft, theta);
rotate_im = abs(fftshift(ifft2(ifftshift(rotate_im_fft))));

subplot(121);
imshow(abs(im));
title("Main Image");

subplot(122);
imshow(abs(rotate_im) ./ max(max(abs(rotate_im))));
title("Rotated Image");
%**************************************************************************




%% Q5

clc;
clear;
close all;

%Loading the image
im = imread("S1_Q5_utils\t1.jpg"); 
im = double(im);

im_ver_diff = (circshift(im, -1, 1) - circshift(im, 1, 1))/2; %Vertical differentiating
im_hor_diff = (circshift(im, -1, 2) - circshift(im, 1, 2))/2; %Horizontal differentiating
im_grad_abs = sqrt(double(im_ver_diff.^2 + im_hor_diff.^2)); %Gradient vector magnitude


%Plotting Images:
figure;

subplot(141);
imshow(im ./ max(max(im)));
title("Main Image");

subplot(142);
imshow(abs(im_ver_diff) ./ max(max(abs(im_ver_diff))));
title("Vertically Differentiated Image");

subplot(143);
imshow(abs(im_hor_diff) ./ max(max(abs(im_hor_diff))));
title("Horiznotally Differentiated Image");

subplot(144);
imshow(im_grad_abs ./ max(max(im_grad_abs)));
title("Gradient Vector Magnitude");



%% Q6
clc; clear;
close all;

%Loading the image
im = imread("S1_Q5_utils\t1.jpg");
im_double = double(im(:,:,1));

%Using Canny & Sobel methods to detect edges:
edge_canny = edge(im_double, 'canny');
edge_sobel = edge(im_double);

%Plotting Images:
figure;
subplot(131);
imshow(im);
title("Main Image");

subplot(132);
imshow(edge_sobel);
title("Sobel edge detection Image");

subplot(133);
imshow(edge_canny);
title("Canny edge detection Image");


