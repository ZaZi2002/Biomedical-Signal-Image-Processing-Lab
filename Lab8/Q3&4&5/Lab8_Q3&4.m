clc
close all
clear all

%% Q3
%%%%% Resizing image
f = imread('S2_Q3_utils\t2.jpg');
f_resized = imresize(f(:,:,1),[64,64],"cubic");

figure;
subplot(1,3,1);
imshow(f_resized);
title("Main image");

%%%%% Producing D
h = [0 1 0;1 2 1;0 1 0];
K = zeros(64,64);
K([1:3],[1:3]) = h;

n = 1;
for c = 1:64
    for r = 1:64
        K_shifetd = circshift(K,[r-1,c-1]);
        D(n,:) = reshape(K_shifetd,1,64*64);
        n = n + 1;
    end
end

%%%%% g = Df
f_new = double(reshape(f_resized,64*64,1));
g_unreshaped = D*f_new;
g = reshape(g_unreshaped,64,64);

%%%%% noisy image
sigma = 0;
gaussian_noise = sigma*randn(64,64);
g_noisy = g + gaussian_noise;
g_noisy = g_noisy/max(max(g_noisy));

%%%%% f_hat
f_hat_unreshaped = pinv(D)*reshape(g_noisy,64*64,1);
f_hat = reshape(f_hat_unreshaped,64,64);
f_hat = f_hat/max(max(f_hat));

%%%%% Plots
figure;
subplot(1,3,1);
imshow(f_resized);
title("Main image");

subplot(1,3,2);
imshow(g_noisy);
title("Noisy image");

subplot(1,3,3);
imshow(f_hat);
title("Reproduced image");



%% Q4
clc
betha = 0.01;
f_k = zeros(64*64,1);
f_k_new = ones(64*64,1);
dif = norm(sum(f_k_new - f_k))/(64*64);
g_noisy_reshaped = reshape(g_noisy,64*64,1);
n = 0;
thresh = 0.1e-9;
while (dif > thresh)
    f_k_new = f_k + betha*transpose(D)*(g_noisy_reshaped - D*f_k);
    dif = norm(sum(f_k_new - f_k))/(64*64);
    f_k = f_k_new;
    n = n+1;
    %dif
end
display("Number of repeats = " + n + " : with thresh = " + thresh);
f_hat = reshape(f_k,64,64);
f_hat = f_hat/max(max(f_hat));

figure;
subplot(1,3,1);
imshow(f_resized);
title("Main image");

subplot(1,3,2);
imshow(g_noisy);
title("Noisy image");

subplot(1,3,3);
imshow(f_hat);
title("Reproduced image");