clc
close all
clear all

%% Part1
load("mecg1.dat");
load("fecg1.dat");
load("noise1.dat");
fs = 256;  % Sampling freq
t = 0.001 : 1/fs : 10;  % ECG time length
mixSignal1 = mecg1 + fecg1 + noise1; % mixing all 3 signals

figure('Name','Part1');
subplot(4,1,1);
plot(t,mecg1);
title("M ECG 1")
xlim('tight');
xlabel('Time (s)')
grid minor

subplot(4,1,2);
plot(t,fecg1);
title("F ECG 1")
xlim('tight');
xlabel('Time (s)')
grid minor

subplot(4,1,3);
plot(t,noise1);
title("Noise")
xlim('tight');
xlabel('Time (s)')
grid minor

subplot(4,1,4);
plot(t,mixSignal1);
title("Mixed Signal")
xlim('tight');
xlabel('Time (s)')
grid minor

%% Part2
% pwelch
mecg1_pwelch = pwelch(mecg1); 
mecg1_pwelch = fftshift(mecg1_pwelch);
fecg1_pwelch = pwelch(fecg1);
fecg1_pwelch = fftshift(fecg1_pwelch);
noise1_pwelch = pwelch(noise1);
noise1_pwelch = fftshift(noise1_pwelch);
mixSignal1_pwelch = pwelch(mixSignal1);
mixSignal1_pwelch = fftshift(mixSignal1_pwelch);

N = length(mecg1_pwelch);
f = fs*(-N/2:N/2-1)/N;

figure('Name','Part2');
subplot(4,1,1);
plot(f,mecg1_pwelch);
title("M ECG 1")
xlim([0,50]);
xlabel('Frequency (Hz)')
grid minor

subplot(4,1,2);
plot(f,fecg1_pwelch);
title("F ECG 1")
xlim([0,100]);
xlabel('Frequency (Hz)')
grid minor

subplot(4,1,3);
plot(f,noise1_pwelch);
title("Noise")
xlim([0,50]);
xlabel('Frequency (Hz)')
grid minor

subplot(4,1,4);
plot(f,mixSignal1_pwelch);
title("Mixed Signal")
xlim([0,100]);
xlabel('Frequency (Hz)')
grid minor

%% Part3
mean_mecg1 = mean(mecg1)
var_mecg1 = var(mecg1)
mean_fecg1 = mean(fecg1)
var_fecg1 = var(fecg1)
mean_noise1 = mean(noise1)
var_noise1 = var(noise1)
mean_mixSignal1 = mean(mixSignal1)
var_mixSignal1 = var(mixSignal1)

%% Part4
n = 5000; % number of bins for histogram
figure('Name','Part4');
subplot(4,1,1);
hist(mecg1,n);
title("M ECG 1")
xlim('tight');
xlabel('Data Values')
grid minor

subplot(4,1,2);
hist(fecg1,n);
title("F ECG 1")
xlim('tight');
xlabel('Data Values')
grid minor

subplot(4,1,3);
hist(noise1,n);
title("Noise")
xlim('tight');
xlabel('Data Values')
grid minor

subplot(4,1,4);
hist(mixSignal1,n);
title("Mixed Signal")
xlim('tight');
xlabel('Data Values')
grid minor

kurt_mecg1 = kurtosis(mecg1)
kurt_fecg1 = kurtosis(fecg1)
kurt_noise1 = kurtosis(noise1)
kurt_mixedSignal1 = kurtosis(mixSignal1)