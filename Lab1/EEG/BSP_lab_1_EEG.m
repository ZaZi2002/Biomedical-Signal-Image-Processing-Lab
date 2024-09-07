clc
close all
clear all
load("EEG_sig.mat");
fs = 256;  % Sampling freq
t = 0.001 : 1/fs : 64;  % EEG time length

%% Part 1
figure("Name","Part1");
plot(t,Z(5,:)); % Plotting channel 5 of Z
grid on;
xlim('tight');
ylim([-100,100]);
title('5th Channel Signal')
xlabel('Time (s)')

%% Part 3
figure("Name","Part3");
subplot(2,1,1)
plot(t,Z(5,:)); % Plotting channel 5 of Z
grid on;
xlim('tight');
ylim([-200,200]);
title('5th Channel Signal')
xlabel('Time (s)')

subplot(2,1,2)
plot(t,Z(23,:)); % Plotting channel 23 of Z
grid on;
xlim('tight');
ylim([-200,200]);
title('23th Channel Signal')
xlabel('Time (s)')

%% Part 4
offset = max(max(abs(Z)))/3 ; % distance between channels in the plot
ElecName = des.channelnames ; % names of 32 channels
disp_eeg(Z,offset,fs,ElecName);
xlim('tight');
grid minor

%% Part 6
epoch(1,:) = Z(5,2*fs+1:7*fs); % seconds of 2 to 7
epoch(2,:) = Z(5,30*fs+1:35*fs); % seconds of 30 to 35
epoch(3,:) = Z(5,42*fs+1:47*fs); % seconds of 42 to 47
epoch(4,:) = Z(5,50*fs+1:55*fs); % seconds of 50 to 55

t2 = 0.001 : 1/fs : 5;

% time plots
figure("Name","Part6_time");
subplot(4,1,1)
plot(t2,epoch(1,:));
grid on;
xlim('tight');
ylim([-100,100]);
title('2 to 7 sec epoch')
xlabel('Time (s)')

subplot(4,1,2)
plot(t2,epoch(2,:));
grid on;
xlim('tight');
ylim([-100,100]);
title('30 to 35 sec epoch')
xlabel('Time (s)')

subplot(4,1,3)
plot(t2,epoch(3,:));
grid on;
xlim('tight');
ylim([-100,100]);
title('42 to 47 sec epoch')
xlabel('Time (s)')

subplot(4,1,4)
plot(t2,epoch(4,:));
grid on;
xlim('tight');
ylim([-100,100]);
title('50 to 55 sec epoch')
xlabel('Time (s)')

% fft plots
N = length(epoch(1,:));
f = fs*(-N/2:N/2-1)/N; % defining freq range

figure("Name","Part6_fft");

epoch_fft(1,:) = fft(epoch(1,:)); 
epoch_fft(1,:) = fftshift(epoch_fft(1,:));
subplot(2,2,1)
plot(f,abs(epoch_fft(1,:)));
grid on;
xlim([-100,100]);
title('2 to 7 sec epoch')
xlabel('Frequency (Hz)')

epoch_fft(2,:) = fft(epoch(2,:)); 
epoch_fft(2,:) = fftshift(epoch_fft(2,:));
subplot(2,2,2)
plot(f,abs(epoch_fft(2,:)));
grid on;
xlim([-100,100]);
title('30 to 35 sec epoch')
xlabel('Frequency (Hz)')

epoch_fft(3,:) = fft(epoch(3,:)); 
epoch_fft(3,:) = fftshift(epoch_fft(3,:));
subplot(2,2,3)
plot(f,abs(epoch_fft(3,:)));
grid on;
xlim([-100,100]);
title('42 to 47 sec epoch')
xlabel('Frequency (Hz)')

epoch_fft(4,:) = fft(epoch(4,:)); 
epoch_fft(4,:) = fftshift(epoch_fft(4,:));
subplot(2,2,4)
plot(f,abs(epoch_fft(4,:)));
grid on;
xlim([-100,100]);
title('50 to 55 sec epoch')
xlabel('Frequency (Hz)')

%% Part 7

figure("Name","Part7_pwelch");

epoch_pwelch(1,:) = pwelch(epoch(1,:));
epoch_pwelch(1,:) = fftshift(epoch_pwelch(1,:));
N = length(epoch_pwelch(1,:));
f = fs*(-N/2:N/2-1)/N; % defining freq range
subplot(2,2,1)
plot(f,epoch_pwelch(1,:));
grid on;
xlim([0,100]);
title('2 to 7 sec epoch')
xlabel('Frequency (Hz)')

epoch_pwelch(2,:) = pwelch(epoch(2,:));
epoch_pwelch(2,:) = fftshift(epoch_pwelch(2,:));
subplot(2,2,2)
plot(f,epoch_pwelch(2,:));
grid on;
xlim([0,100]);
title('30 to 35 sec epoch')
xlabel('Frequency (Hz)')

epoch_pwelch(3,:) = pwelch(epoch(3,:));
epoch_pwelch(3,:) = fftshift(epoch_pwelch(3,:));
subplot(2,2,3)
plot(f,epoch_pwelch(3,:));
grid on;
xlim([0,100]);
title('42 to 47 sec epoch')
xlabel('Frequency (Hz)')

epoch_pwelch(4,:) = pwelch(epoch(4,:));
epoch_pwelch(4,:) = fftshift(epoch_pwelch(4,:));
subplot(2,2,4)
plot(f,epoch_pwelch(4,:));
grid on;
xlim([0,100]);
title('50 to 55 sec epoch')
xlabel('Frequency (Hz)')

%% Part 8
figure("Name","Part8_spectrogram");
window = hamming(128);
subplot(2,2,1)
spectrogram(epoch(1,:),window,64,128,fs,'yaxis');
title('2 to 7 sec epoch')
subplot(2,2,2)
spectrogram(epoch(2,:),window,64,128,fs,'yaxis');
title('30 to 35 sec epoch')
subplot(2,2,3)
spectrogram(epoch(3,:),window,64,128,fs,'yaxis');
title('42 to 47 sec epoch')
subplot(2,2,4)
spectrogram(epoch(4,:),window,64,128,fs,'yaxis');
title('50 to 55 sec epoch')

%% Part 9
[b,a] = butter(6,60/(fs/2),'low'); % Butterworth filter of order 6
filtered_epoch = filter(b,a,epoch(2,:));
downsampled_epoch = downsample(filtered_epoch,2); % down sampling
fs2 = 128; % new sampling freq
t3 = 0.001 : 1/fs2 : 5;
figure("Name","Part9_time");

subplot(3,1,1)
plot(t3,downsampled_epoch);
grid on;
xlim('tight');
ylim([-100,100]);
title('30 to 35 sec filtered epoch')
xlabel('Time (s)')

epoch_fft_2 = fft(downsampled_epoch); 
epoch_fft_2 = fftshift(epoch_fft_2);
N = length(downsampled_epoch);
f = fs2*(-N/2:N/2-1)/N;
subplot(3,1,2)
plot(f,abs(epoch_fft_2));
grid on;
xlim([-100,100]);
title('30 to 35 sec fft of filtered epoch')
xlabel('Frequency (Hz)')

window = hamming(64);
subplot(3,1,3)
spectrogram(downsampled_epoch,window,32,64,fs2,'yaxis');
title('30 to 35 sec epoch')