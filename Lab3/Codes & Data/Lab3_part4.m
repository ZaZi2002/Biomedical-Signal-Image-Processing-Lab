%% PART4: Q1
clc; 
clear;
close all;

fs = 256;
data = load('X.dat'); %Loading data

plot3ch(data, fs); %Plotting scatter for the data (Ignore Figure1 !)


%********************For SVD method:****************************
denoised_data_svd = cell2mat(struct2cell(load('denoised_data_svd.mat'))); %Loading denoised data

%Loading SVD matrices:
% U = cell2mat(struct2cell(load('U.mat')));
S = cell2mat(struct2cell(load('S.mat')));
V = cell2mat(struct2cell(load('V.mat')));
denoised_S = cell2mat(struct2cell(load('denoised_S.mat'))); %S matrix after removing unwanted components

%Plotting the directions of columns of V:
for i=1:size(V,2)
    plot3dv(V(:,i),S(i,i), 'BLUE');
end

%Denoised data scatter (Ignore the first figure)
plot3ch(denoised_data_svd, fs);
for i=1:size(V,2)
    plot3dv(V(:,i),S(i,i), 'BLUE');
end

%Norm for each column in V:
norm_V1 = norm(V(:,1))
norm_V2 = norm(V(:,2))
norm_V3 = norm(V(:,3))

%Angles between column1&2, column1&3, column2&3 in V, respectively:
vectsAngle_SVD = zeros(3,1);
vectsAngle_SVD(1) = rad2deg(acos(dot(V(:,1), V(:,2))/(norm_V1*norm_V2))) %between column1&2
vectsAngle_SVD(2) = rad2deg(acos(dot(V(:,1), V(:,3))/(norm_V1*norm_V3))) %between column1&3
vectsAngle_SVD(3) = rad2deg(acos(dot(V(:,2), V(:,3))/(norm_V2*norm_V3))) %between column2&3
%*************************************************************

%**********************For ICA method:*******************
W_inv = load('W_inv.mat');
ZHAT = load('ZHAT.mat');
ICA_X = load('ICA_X.mat');
plot3ch(ICA_X.new_X,fs,'ICA-X');
plot3dv(W_inv.W_inv(:,1));
plot3dv(W_inv.W_inv(:,2));
plot3dv(W_inv.W_inv(:,3));
grid minor

%Norm for each column in W_inv:
norm_W1 = norm(W_inv.W_inv(:,1))
norm_W2 = norm(W_inv.W_inv(:,2))
norm_W3 = norm(W_inv.W_inv(:,3))

vectsAngle_ICA(1) = rad2deg(acos(dot(W_inv.W_inv(:,1), W_inv.W_inv(:,2))/(norm_W1*norm_W2)))
vectsAngle_ICA(2) = rad2deg(acos(dot(W_inv.W_inv(:,1), W_inv.W_inv(:,3))/(norm_W1*norm_W3)))
vectsAngle_ICA(3) = rad2deg(acos(dot(W_inv.W_inv(:,2), W_inv.W_inv(:,3))/(norm_W2*norm_W3)))

%********************************************************


%% PART4: Q2

cleanData = load('fecg2.dat'); %Loading clean data
t = 1/fs : 1/fs : size(data,1)/fs; 


%*********************For clean data:***************
figure;
plot(t, cleanData);
title("Clean data signal");
xlabel('Time [sec]');
ylabel('Potential [mV]');
%**************************************************


%**********************For SVD method:*******************
figure;
plot(t, denoised_data_svd(:,3));
title("SVD denoised signal (Channel3)");
xlabel('Time [sec]');
ylabel('Potential [mV]');
%********************************************************


%**********************For ICA method:*******************
figure('Name','Part3');

% plot(t,ICA_X.new_X(:,1));
% title("first channel")
% xlim('tight');
% xlabel('Time (s)')
% grid minor
% 
% subplot(3,1,2);
% plot(t,ICA_X.new_X(:,2));
% title("second channel")
% xlim('tight');
% xlabel('Time (s)')
% grid minor

% subplot(3,1,3);
plot(t,ICA_X.new_X(:,3));
title("third channel")
xlim('tight');
xlabel('Time (s)')
grid minor
%********************************************************

%% PART4: Q3

%**********************For SVD method:*******************
corrcoef(denoised_data_svd(:,3) , cleanData)
%********************************************************

%**********************For ICA method:*******************
corrcoef(ICA_X.new_X(:,3) , cleanData)
%********************************************************



