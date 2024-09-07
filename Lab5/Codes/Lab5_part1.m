
%% Part1 : a
clc; close all;
clear;

normal = cell2mat(struct2cell(load('normal.mat'))); %Loading data
Fs = 250; %Sampling Frequency

cleanSeg = normal(1:10*Fs,:); %A segment of the clean signal
noisySeg = normal(end-10*Fs:end,:); %A segment of the noisy signal

winLen = 200; %Window Lenght for pwelch
[cleanSeg_spect, f] = pwelch(cleanSeg(:, 2), winLen, [], [], Fs); %Computing pwelch
[noisySeg_spect, f] = pwelch(noisySeg(:, 2), winLen, [], [], Fs); %Computing pwelch


%*********************Plotting pwelch***********************
figure;
subplot(211);
plot(f, db(cleanSeg_spect, 'power'), 'LineWidth',1);
title("Clean segment (First 10 seconds)");
xlabel("F (Hz)");
ylabel("Spectrum (dB)");

subplot(212);
plot(f, db(noisySeg_spect, 'power'), 'LineWidth',1);
title("Noisy segment (Last 10 seconds)");
xlabel("F (Hz)");
ylabel("Spectrum (dB)");
%***********************************************************


%% part1 : b
clc; close all;



Fpass1 = 2; %Passband frequency 1 for the bandpass filter
Fpass2 = 120; %Passband frequency 2 for the bandpass filter

order = 2; %The order of the filter


 %Removing Baseline using a high-pass filter (in order to compute the total
 %energy of the signal):
filt = designfilt('bandpassiir','FilterOrder',order,...
        'HalfPowerFrequency1',Fpass1,'HalfPowerFrequency2',Fpass2, ...
        'SampleRate',Fs);

filt_data = filter(filt,cleanSeg(:, 2)); %Removing baseline
total_en = sum(filt_data.^2); %Computing the total energy of the signal, without considering baseline

filt_data_en = 0;
Fpass2 = 15; %Passband frequency 2 for the bandpass filter (Initial value)

%Increasing Fpass2 until the energy of the filtered data exceeds 90 percent of the total energy:
while filt_data_en < 0.9 * total_en 
    disp("Fpass2 = " + Fpass2);
    Fpass2 = Fpass2 + 1; %Increase Fpass2

    %Designing the new filter:
    filt = designfilt('bandpassiir','FilterOrder',order,...
        'HalfPowerFrequency1',Fpass1,'HalfPowerFrequency2',Fpass2, ...
        'SampleRate',Fs);

    
    filt_data = filter(filt,cleanSeg(:, 2)); %Applying the filter to the clean sginal
    filt_data_en = sum(filt_data.^2); %Computing The new energy
    disp("Filtered data en = " + filt_data_en);
end
disp("Final FPass2 value = " + Fpass2);

freqz(filt, 512, Fs); %Plotting frequency response of the final Filter

[h,t] = impz(filt);

figure;
stem(t,h);
xlabel('Sample');
ylabel('Amplitude');
title('Impulse Response of the Filter');

%% part1: c
clc;


filt_cleanSeg = filter(filt,cleanSeg(:, 2)); %Filtering clean segment
filt_cleanSeg = [cleanSeg(:,1), filt_cleanSeg]; %Appending time vector. dimension=samples*1
filt_noisySeg = filter(filt,noisySeg(:, 2)); %Filtering noisy segment.
filt_noisySeg = [noisySeg(:,1), filt_noisySeg]; % %Appending time vector. dimension=samples*1


%****************Plotting unfiltered data*****************
figure;
subplot(211);
plot(cleanSeg(:, 1), cleanSeg(:, 2), 'LineWidth',1);
title("Raw Clean segment (First 10 seconds)");
xlabel("Time (s)");
ylabel("Voltage (V)");

subplot(212);
plot(noisySeg(:, 1), noisySeg(:, 2), 'LineWidth',1);
title("Raw Noisy segment (Last 10 seconds)");
xlabel("Time (s)");
ylabel("Voltage (V)");
%*******************************************************



%****************Plotting filtered data*****************
figure;
subplot(211);
plot(filt_cleanSeg(:, 1), filt_cleanSeg(:, 2), 'LineWidth',1);
title("Filtered Clean segment (First 10 seconds)");
xlabel("Time (s)");
ylabel("Voltage (V)");

subplot(212);
plot(filt_noisySeg(:, 1), filt_noisySeg(:, 2), 'LineWidth',1);
title("Filtered Noisy segment (Last 10 seconds)");
xlabel("Time (s)");
ylabel("Voltage (V)");
%*******************************************************








