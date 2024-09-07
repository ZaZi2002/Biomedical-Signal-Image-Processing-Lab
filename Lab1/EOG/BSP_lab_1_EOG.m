clc
close all
clear all
load("EOG_sig.mat");
fs = 256;  % Sampling freq
t = 0.001 : 1/fs : 30;  % EEG time length

%% Part 1
figure("Name","Part1");
subplot(2,1,1);
plot(t,Sig(1,:));
grid on;
xlim('tight');
ylim([-15000,15000])
title(Labels(1))
xlabel('Time (s)')

subplot(2,1,2);
plot(t,Sig(2,:));
grid on;
xlim('tight');
ylim([-15000,15000])
title(Labels(2))
xlabel('Time (s)')

%% Part 2
Sig_fft(1,:) = fft(Sig(1,:));
Sig_fft(1,:) = fftshift(Sig_fft(1,:));
Sig_fft(2,:) = fft(Sig(2,:));
Sig_fft(2,:) = fftshift(Sig_fft(2,:));
N = length(Sig(1,:));
f = fs*(-N/2:N/2-1)/N;

figure("Name","Part2_fft");
subplot(2,1,1);
plot(f,abs(Sig_fft(1,:)));
grid on;
xlim([-20,20]);
title(Labels(1))
xlabel('Frequency (Hz)')

subplot(2,1,2);
plot(f,abs(Sig_fft(1,:)));
grid on;
xlim([-20,20]);
title(Labels(2))
xlabel('Frequency (Hz)')

figure("Name","Part2_spectrogram");
window = hamming(128);
subplot(2,1,1)
spectrogram(Sig(1,:),window,64,128,fs,'yaxis');
subplot(2,1,2)
spectrogram(Sig(2,:),window,64,128,fs,'yaxis');