clc
close all
clear all

fs = 256;

%% Part1
load('X.dat');
[W,ZHAT] = ica(transpose(X));
W_inv = inv(W);
save('W.mat',"W");
save('W_inv.mat',"W_inv");
save('ZHAT.mat',"ZHAT");

%% Part2
plot3ch(X,fs,'X');

plot3dv(W_inv(:,1));
plot3dv(W_inv(:,2));
plot3dv(W_inv(:,3));
grid minor

%% Part3
figure('Name','Part3');
t = 0.001 : 1/fs : 10; 

subplot(3,1,1);
plot(t,ZHAT(1,:));
title("first row")
xlim('tight');
xlabel('Time (s)')
grid minor

subplot(3,1,2);
plot(t,ZHAT(2,:));
title("second row")
xlim('tight');
xlabel('Time (s)')
grid minor

subplot(3,1,3);
plot(t,ZHAT(3,:));
title("third row")
xlim('tight');
xlabel('Time (s)')
grid minor

% removing mother and noise sources
W_inv(:,1) = 0; 
W_inv(:,2) = 0;

% producing new observed signals
new_X = W_inv*ZHAT;
new_X = transpose(new_X);

%% Part4
plot3ch(new_X,fs,'X');
grid minor
save('ICA_X.mat',"new_X");
