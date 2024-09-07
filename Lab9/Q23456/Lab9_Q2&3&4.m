clc
clear all
close all

%% Q2
%%%%% Reading images %%%%%
pd = imread('S3_Q2_utils\pd.jpg');
t1 = imread("S3_Q2_utils\t1.jpg");
t2 = imread('S3_Q2_utils\t2.jpg');
pd = double(pd(:,:,1));
t1 = double(t1(:,:,1));
t2 = double(t2(:,:,1));

%%%%% Producing feature vectors %%%%%
features = zeros(249*213,3);
features(:,1) = reshape(pd,249*213,1);
features(:,2) = reshape(t1,249*213,1);
features(:,3) = reshape(t2,249*213,1);

%%%%% K-means %%%%%
clustered_im = kmeans(features,6);
clustered_im = reshape(clustered_im,249,213);
clusteres = zeros(249,213,6);
for i = 1:249
    for j = 1:213
        clusteres(i,j,clustered_im(i,j)) = 1;
    end
end

%%%%% Plotting clusters %%%%%
figure('Name',"clusters");
for i = 1:6
    subplot(2,3,i);
    imshow(clusteres(:,:,i));
    title("Cluster " + i);
end

%% Q3
clc
%%%%% First random centeres %%%%%
k = 6;
C = zeros(6,3);
C_old = zeros(6,3);
for i = 1:k
    n = randi([1,249*213]);
    C(i,:) = features(n,:);
    C_old(i,:) = features(n,:) + 10;
end

%%%%% K-means cycle %%%%%
N = 0;
while (abs(C_old-C) > 10e-9)
    N = N + 1;
    %%%%% Clustering %%%%%
    clustering = zeros(249*13,1);
    for i = 1:249*213
        MIN = 256*256*3 + 1;
        n = 0;
        for j = 1:k
            dist = (features(i,1)-C(j,1))^2 + (features(i,2)-C(j,2))^2 + (features(i,3)-C(j,3))^2;
            if (dist<MIN)
                MIN = dist;
                n = j;
            end
        end
        clustering(i) = n;
    end
    
    %%%%% Finding new centeres %%%%%
    MEAN = zeros(6,3);
    amounts = ones(6,1);
    for i = 1:249*213
        MEAN(clustering(i),:) = MEAN(clustering(i),:) + features(i,:);
        amounts(clustering(i)) = amounts(clustering(i)) + 1;
    end
    C_old = C;
    for i = 1:k
        C(i,:) = MEAN(i,:)/amounts(i);
    end
end

disp("Convergence cycles = " + N)

%%%%% Finding clusters %%%%%
clustering = reshape(clustering,249,213);
clusteres = zeros(249,213,6);
for i = 1:249
    for j = 1:213
        clusteres(i,j,clustered_im(i,j)) = 1;
    end
end

%%%%% Plotting clusters %%%%%
figure('Name',"clusters");
for i = 1:6
    subplot(2,3,i);
    imshow(clusteres(:,:,i));
    title("Cluster " + i);
end

%% Q4
%%%%% FCM %%%%%
[C,clustered_fcm] = fcm(features,6);
clustered_fcm_reshaped = reshape(clustered_fcm.',249,213,6);

%%%%% Plotting clusters %%%%%
figure('Name',"clusters");
for i = 1:6
    subplot(2,3,i);
    imshow(clustered_fcm_reshaped(:,:,i));
    title("Cluster " + i);
end

%%%%% Certain clustering using max meow %%%%%
MAX_meow = max(clustered_fcm);
clusteres = zeros(249*213,6);
for i = 1:249*213
    for j = 1:6
        if clustered_fcm(j,i) == MAX_meow(i)
            clusteres(i,j) = 1;
        end
    end
end
clustered_MAX_reshaped = reshape(clusteres,249,213,6);

%%%%% Plotting clusters %%%%%
figure('Name',"clusters");
for i = 1:6
    subplot(2,3,i);
    imshow(clustered_MAX_reshaped(:,:,i));
    title("Cluster " + i);
end
