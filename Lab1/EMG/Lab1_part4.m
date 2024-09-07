clc;
close all;
clear;
load("EMG_sig.mat");
fs = 4000;  % Sampling freq

%% Part 4, Q1
figure("Name","Part1");
subplot(3,1,1);
t_healthym = 1/fs: 1/fs: (length(emg_healthym))/fs; %Defining time vector
plot(t_healthym, emg_healthym);
title('EMG signal for healthy subject');
grid on;
xlim('tight');
xlabel('Time (s)');
ylabel('EMG [uV]');

subplot(3,1,2);
t_myopathym = 1/fs: 1/fs: length(emg_myopathym)/fs; %Defining time vector
plot(t_myopathym, emg_myopathym);
title('EMG signal for myopathy subject');
grid on;
xlim('tight');
xlabel('Time (s)')
ylabel('EMG [uV]');

subplot(3,1,3);
t_neuropathym = 1/fs: 1/fs: length(emg_neuropathym)/fs; %Defining time vector
plot(t_neuropathym, emg_neuropathym);
title('EMG signal for neuropathy subject');
grid on;
xlim('tight');
xlabel('Time (s)')
ylabel('EMG [uV]');

%% part 4, Q2
clc;
% close all;

%*******************FFT plots***********************
xlim_arr = [-1500, 1500]; %Frequency interval to be shown on FFT plot

figure("Name","Part4_Q2_fft");

%For healthy subject:
fft_healthym = fft(emg_healthym); 
fft_healthym = fftshift(fft_healthym);
N = length(emg_healthym);
f = fs*(-N/2:N/2-1)/N;

subplot(3,1,1)
plot(f,abs(fft_healthym));
grid on;
xlim(xlim_arr);
title('FFT for healthy subject')
xlabel('Frequency (Hz)');
ylabel('FFT');


%For myopathy subject
fft_myopathym = fft(emg_myopathym); 
fft_myopathym = fftshift(fft_myopathym);
N = length(emg_myopathym);
f = fs*(-N/2:N/2-1)/N;

subplot(3,1,2)
plot(f,abs(fft_myopathym));
grid on;
xlim(xlim_arr);
title('FFT for myopathy subject')
xlabel('Frequency (Hz)');
ylabel('FFT');



%For neuropathy subject:
fft_neuropathym = fft(emg_neuropathym); 
fft_neuropathym = fftshift(fft_neuropathym);
N = length(emg_neuropathym);
f = fs*(-N/2:N/2-1)/N;

subplot(3,1,3)
plot(f,abs(fft_neuropathym));
grid on;
xlim(xlim_arr);
title('FFT for neuropathy subject')
xlabel('Frequency (Hz)');
ylabel('FFT');
%*******************************************************




%*******************Spectrograms**************************
L = 128; %Using The number used in EEG analysis
n_overlap = L/2;
n_fft = L;

figure("Name","Part4_Q2_spectrogram");
window = hamming(L); %Defining the window for computing STFT
subplot(3,1,1);
spectrogram(emg_healthym,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for healthy subject')

subplot(3,1,2);
spectrogram(emg_myopathym,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for myopathy subject')

subplot(3,1,3);
spectrogram(emg_neuropathym,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for neuropathy subject')

%*************************************************************
