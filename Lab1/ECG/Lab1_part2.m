%% Part2 (ECG), Q1
clc; clear;
dataset = (load('ECG_sig.mat')); %Loading ECG dataset
sig = dataset.Sig; %Choosing the main signal. Dimension = samples*channels
fs = dataset.sfreq; %Sampling frequency
duration = size(sig,1)/fs; %Duration of the signal [sec]
t = 1/fs:1/fs:duration; %Defining Time vector

%**************Plotting time-domain signals********************
figure;
subplot(211);
plot(t, sig(1:duration*fs, 1));
title('Channel 1');
xlabel('Time[sec]');
ylabel('ECG signal [mV]');

subplot(212);
plot(t, sig(1:duration*fs, 2));
title('Channel 2');
xlabel('Time[sec]');
ylabel('ECG signal [mV]');
%************************************************
%% Part2, Q2
clc;
slct_duration = [530, 540]; %Defining the beginning and the end of the interval to plot [sec]
R_times = dataset.ATRTIMED; %single column vector
R_types = dataset.ANNOTD; %Single column vector
R_labels = strings(length(R_types),1); %Defining the vector of labels, corresponding to arrhythmia times
for i = 1:length(R_types)
    R_labels(i,1) = num2label(R_types(i)); %Labeling each arrhythmia
end

%Choosing arrhythmia which exist inside the selected duration:
slct_R_times = R_times(R_times >= slct_duration(1) & R_times<=slct_duration(2));
slct_R_types = R_types(R_times >= slct_duration(1) & R_times<=slct_duration(2));
slct_R_labels = R_labels(R_times >= slct_duration(1) & R_times<=slct_duration(2));

%Time vector for selected duration:
slct_t = slct_duration(1)+1/fs:1/fs:slct_duration(2);
slct_samps = round(slct_t*fs);

figure;
subplot(211);
plot(slct_t, sig(slct_samps, 1)); %Plotting channel 1

%Adding arrhythmia labels to the plot:
for i = 1:length(slct_R_times)
    text(slct_R_times(i), -0.3, slct_R_labels(i), 'FontSize',5, 'HorizontalAlignment', 'center');
end
title('Channel 1');
xlabel('Time[sec]');
ylabel('ECG signal [mV]');

subplot(212);
plot(slct_t, sig(slct_samps, 2)); %Plotting channel 2

%Adding arrhythmia labels to the plot:
for i = 1:length(slct_R_times)
    text(slct_R_times(i), -1.3, slct_R_labels(i), 'FontSize',5, 'HorizontalAlignment', 'center');
end
title('Channel 2');
xlabel('Time[sec]');
ylabel('ECG signal [mV]');

%% Part2, Q4
% clc; 
% close all;

slct_duration_normal = [1, 3]; %Selecting a duration to plot normal heartbeats [sec].
slct_duration_disorder = [477, 479]; %Selecting a duration to plot arrhythmias [sec]

R_times = dataset.ATRTIMED; %single column vector
R_types = dataset.ANNOTD; %Single column vector
R_labels = strings(length(R_types),1);
for i = 1:length(R_types)
    R_labels(i,1) = num2label(R_types(i));
end

%********************Plotting signal for normal heartbeats:****************
%Choosing arrhythmia which exist inside the selected duration:
slct_R_times = R_times(R_times >= slct_duration_normal(1) & R_times<=slct_duration_normal(2));
slct_R_types = R_types(R_times >= slct_duration_normal(1) & R_times<=slct_duration_normal(2));
slct_R_labels = R_labels(R_times >= slct_duration_normal(1) & R_times<=slct_duration_normal(2));

slct_t = slct_duration_normal(1)+1/fs:1/fs:slct_duration_normal(2);
slct_samps = round(slct_t*fs);

figure;
subplot(211);
normalSig_ch1 = sig(slct_samps, 1);
plot(slct_t, normalSig_ch1);
for i = 1:length(slct_R_times)
    text(slct_R_times(i), -0.4, slct_R_labels(i), 'FontSize',5, 'HorizontalAlignment', 'center');
end
title('Channel 1');
xlabel('Time[sec]');
ylabel('ECG signal [mV]');

subplot(212);
normalSig_ch2 = sig(slct_samps, 2);
plot(slct_t, normalSig_ch2);
for i = 1:length(slct_R_times)
    text(slct_R_times(i), -1, slct_R_labels(i), 'FontSize',5, 'HorizontalAlignment', 'center');
end
title('Channel 2');
xlabel('Time[sec]');
ylabel('ECG signal [mV]');
%***********************************************************************



%********************Plotting signal for arrhythmias:****************
%Choosing arrhythmia which exist inside the selected duration:
slct_R_times = R_times(R_times >= slct_duration_disorder(1) & R_times<=slct_duration_disorder(2));
slct_R_types = R_types(R_times >= slct_duration_disorder(1) & R_times<=slct_duration_disorder(2));
slct_R_labels = R_labels(R_times >= slct_duration_disorder(1) & R_times<=slct_duration_disorder(2));

slct_t = slct_duration_disorder(1)+1/fs:1/fs:slct_duration_disorder(2);
slct_samps = round(slct_t*fs);

figure;
subplot(211);
disorderSig_ch1 = sig(slct_samps, 1);
plot(slct_t, disorderSig_ch1);
for i = 1:length(slct_R_times)
    text(slct_R_times(i), -0.4, slct_R_labels(i), 'FontSize',5, 'HorizontalAlignment', 'center');
end
title('Channel 1');
xlabel('Time[sec]');
ylabel('ECG signal [mV]');

subplot(212);
disorderSig_ch2 = sig(slct_samps, 2);
plot(slct_t, disorderSig_ch2);
for i = 1:length(slct_R_times)
    text(slct_R_times(i), -1, slct_R_labels(i), 'FontSize',5, 'HorizontalAlignment', 'center');
end
title('Channel 2');
xlabel('Time[sec]');
ylabel('ECG signal [mV]');
%***********************************************************************


%% part2, Q4: analysis
clc;
% close all;

%*******************FFT plots***********************

%***********For normal beats:
N = length(normalSig_ch1); %total length of the signal
f = fs*(-N/2:N/2-1)/N; %Defining frequency array
xlim_arr = [-60, 60]; %Frequency interval to be shown in FFT plots

figure("Name","Part2_Q4_fft");

%Calculating FFT for channel1:
fft_normalSig_ch1 = fft(normalSig_ch1); 
fft_normalSig_ch1 = fftshift(fft_normalSig_ch1);

subplot(2,2,1)
plot(f,abs(fft_normalSig_ch1));
grid on;
xlim(xlim_arr);
title('FFT for normal beats signal, channel 1')
xlabel('Frequency (Hz)');
ylabel('FFT');

%Calculating FFT for channel2:
fft_normalSig_ch2 = fft(normalSig_ch2); 
fft_normalSig_ch2 = fftshift(fft_normalSig_ch2);

subplot(2,2,2)
plot(f,abs(fft_normalSig_ch2));
grid on;
xlim(xlim_arr);
title('FFT for normal beats signal, channel 2')
xlabel('Frequency (Hz)');
ylabel('FFT');

%***********for arrhythmias:
N = length(disorderSig_ch1);
f = fs*(-N/2:N/2-1)/N;

%Calculating FFT for channel1:
fft_disorderSig_ch1 = fft(disorderSig_ch1); 
fft_disorderSig_ch1 = fftshift(fft_disorderSig_ch1);

subplot(2,2,3)
plot(f,abs(fft_disorderSig_ch1));
grid on;
xlim(xlim_arr);
title('FFT for disorder beats signal, channel 1')
xlabel('Frequency (Hz)');
ylabel('FFT');

%Calculating FFT for channel2:
fft_disorderSig_ch2 = fft(disorderSig_ch2); 
fft_disorderSig_ch2 = fftshift(fft_disorderSig_ch2);

subplot(2,2,4)
plot(f,abs(fft_disorderSig_ch2));
grid on;
xlim(xlim_arr);
title('FFT for disorder beats signal, channel 2')
xlabel('Frequency (Hz)');
ylabel('FFT');
%*******************************************************




%**************************Spectrograms*********************************

%***********Using paramters used in EEG anaysis for Part 1:
L = 128; %Using The number used in EEG analysis
n_overlap = L/2; %Number of overlapping samples (50%)
n_fft = L; %Number of FFT points
window = hamming(L); %Defining the window used for computing STFT

figure("Name","Part2_Q4_spectrogram");

subplot(2,2,1);
spectrogram(normalSig_ch1,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for normal beats signal, channel 1')

subplot(2,2,2);
spectrogram(normalSig_ch2,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for normal beats signal, channel 2')

subplot(2,2,3);
spectrogram(disorderSig_ch1,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for disorder beats signal, channel 1')

subplot(2,2,4);
spectrogram(disorderSig_ch2,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for disorder beats signal, channel 2');





%***********Using another value for L (for better time resolution):
L = 64; %Using another number for better time resolution
n_overlap = L/2;
n_fft = L;

figure("Name","Part2_Q4_spectrogram");
window = hamming(L);
subplot(2,2,1);
spectrogram(normalSig_ch1,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for normal beats signal, channel 1')

subplot(2,2,2);
spectrogram(normalSig_ch2,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for normal beats signal, channel 2')

subplot(2,2,3);
spectrogram(disorderSig_ch1,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for disorder beats signal, channel 1')

subplot(2,2,4);
spectrogram(disorderSig_ch2,window,n_overlap,n_fft,fs,'yaxis');
title('Spectrogram for disorder beats signal, channel 2');
%*************************************************************




%% functions

function name = num2label(number_arr)
%This function converts the number of disorder into its label
        switch number_arr
            case 0
                name = "NOTQRS";
            case 1
                name = "NORMAL";
            case 2
                name = "LBBB";
            case 3
                name = "RBBB";
            case 4
                name = "ABERR";
            case 5
                name = "PVC";
            case 6
                name = "FUSION";
            case 7
                name = "NPC";
            case 8
                name = "APC";
            case 9
                name = "SVPB";
            case 10
                name = "VESC";
            case 11
                name = "NESC";
            case 12
                name = "PACE";
            case 13
                name = "UNKNOWN";
            case 14
                name = "NOISE";
            case 16
                name = "ARFCT";
            case 18
                name = "STCH";
            case 19
                name = "TCH";
            case 20
                name = "SYSTOLE";
            case 21
                name = "DIASTOLE";
            case 22
                name = "NOTE";
            case 23
                name = "MEASURE";
            case 24
                name = "PWAVE";
            case 25
                name = "BBB";
            case 26
                name = "PACESP";
            case 27
                name = "TWAVE";
            case 28
                name = "RHYTHM";
            case 29
                name = "UWAVE";
            case 30
                name = "LEARN";
            case 31
                name = "FLWAV";
            case 32
                name = "VFON";
            case 33
                name = "VFOFF";
            case 34
                name = "AESC";
            case 35
                name = "SVESC";
            case 36
                name = "LINK";
            case 37
                name = "NAPC";
            case 38
                name = "PFUS";
            case 39
                name = "WFON";
            case 40
                name = "WFOFF";
            case 41
                name = "RONT";
        end
    end







