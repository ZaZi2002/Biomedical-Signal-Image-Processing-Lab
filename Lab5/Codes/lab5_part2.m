clc
close all
clear;

%% Part1
clc;
load('n_422.mat');
load('n_424.mat');
fs = 250;
t = 0.0001:1/fs:10;

%%%%% Normal and arrhythmian epochs
normal_n22 = n_422(1 : 10*fs);
normal_n24 = n_424(1 : 10*fs);
arrhythmia_n22 = n_422(61288 : 61287 + 10*fs);
arrhythmia_n24 = n_424(27249 : 27248 + 10*fs);

%%%%% Pwelch of the signals
pwelch_norm_n22 = fftshift(pwelch(normal_n22));
pwelch_norm_n24 = fftshift(pwelch(normal_n24));
pwelch_arrhythm_n22 = fftshift(pwelch(arrhythmia_n22));
pwelch_arrhythm_n24 = fftshift(pwelch(arrhythmia_n24));

%%%%% Plotting pwelches
N = length(pwelch_norm_n22);
f = fs*(-N/2:N/2-1)/N;

figure('Name',"Part1_n22");
subplot(2,1,1);
plot(f,pwelch_norm_n22);
title("Normal epoch");
xlabel("Frequency(Hz)");
ylabel("Amplitude");
xlim([0,30]);
grid minor
subplot(2,1,2);
plot(f,pwelch_arrhythm_n22);
title("Arrythmia epoch");
xlabel("Frequency(Hz)");
ylabel("Amplitude");
xlim([0,30]);
grid minor
saveas(gcf,"Part1_n22.png");

figure('Name',"Part1_n24");
subplot(2,1,1);
plot(f,pwelch_norm_n24);
title("Normal epoch");
xlabel("Frequency(Hz)");
ylabel("Amplitude");
xlim([0,30]);
grid minor
subplot(2,1,2);
plot(f,pwelch_arrhythm_n24);
title("Arrythmia epoch");
xlabel("Frequency(Hz)");
ylabel("Amplitude");
xlim([0,30]);
grid minor
saveas(gcf,"Part1_n24.png");

%% Part2
%%%%% ffts of signals
normal_n22_fft = fftshift(fft(normal_n22));
normal_n24_fft = fftshift(fft(normal_n24));
arrhythmia_n22_fft = fftshift(fft(arrhythmia_n22));
arrhythmia_n24_fft = fftshift(fft(arrhythmia_n24));

N = length(normal_n22_fft);
f = fs*(-N/2:N/2-1)/N;

%%%%% Time and fourier plots

% n_22
figure('Name',"Part2_n22");
subplot(2,2,1);
plot(t,normal_n22);
title("Normal epoch");
xlabel("Time(s)");
ylabel("Amplitude");
grid minor
subplot(2,2,2);
plot(f,abs(normal_n22_fft));
title("Normal epoch");
xlabel("Frequency(Hz)");
ylabel("Amplitude");
xlim([0,30]);
grid minor
subplot(2,2,3);
plot(t,arrhythmia_n22);
title("Arrythmia epoch");
xlabel("Time(s)");
ylabel("Amplitude");
grid minor
subplot(2,2,4);
plot(f,abs(arrhythmia_n22_fft));
title("Arrythmia epoch");
xlabel("Frequency(Hz)");
ylabel("Amplitude");
xlim([0,30]);
grid minor
saveas(gcf,"Part2_n22.png");

%n_24
figure('Name',"Part2_n24");
subplot(2,2,1);
plot(t,normal_n24);
title("Normal epoch");
xlabel("Time(s)");
ylabel("Amplitude");
grid minor
subplot(2,2,2);
plot(f,abs(normal_n24_fft));
title("Normal epoch");
xlabel("Frequency(Hz)");
ylabel("Amplitude");
xlim([0,30]);
grid minor
subplot(2,2,3);
plot(t,arrhythmia_n24);
title("Arrythmia epoch");
xlabel("Time(s)");
ylabel("Amplitude");
grid minor
subplot(2,2,4);
plot(f,abs(arrhythmia_n24_fft));
title("Arrythmia epoch");
xlabel("Frequency(Hz)");
ylabel("Amplitude");
xlim([0,30]);
grid minor
saveas(gcf,"Part2_n24.png");

%% Part3
clc;
n_422_eventFirstSamps = [1, 10711, 11211, 11442, 59711, 61288]; %Samples in n422 in which the events started
n_422_eventLastSamps = [10710, 11210, 11441, 59710, 61287, 75000]; %Samples in n422 in which the events ended

n_422_events = ["N","VT","N","VT","NOISE","VFIB"]; %Array of events happened in n422


winLen = 10*fs; %The length of each window [samples].
overlap = 50; %Windows overlap, as a percentage of winLen.
totalLen = length(n_422); %Total length of the signal

windows = round((totalLen-winLen)/(winLen*(1-overlap/100)))+1; %Total number of windows
windowsArr = zeros(windows,1);

labelsArr = zeros(windows,1); %Time array for final image
lastSampArr = zeros(windows, 1); %The number of the last sample in each window
firstSampArr =  zeros(windows, 1); %The number of the first sample in each window

for win = 1:windows
    %Computing the array of samples in window number win:
    T = (win-1)*winLen*(1-overlap/100)+1 : ...
        winLen+(win-1)*winLen*(1-overlap/100);
    T = round(T);

    lastSampArr(win) = T(end);
    firstSampArr(win) = T(1);

    windowsArr(win) = mean(T)/fs;

    for ev = 1:length(n_422_eventFirstSamps)
        if firstSampArr(win) >= n_422_eventFirstSamps(ev) && lastSampArr(win) <= n_422_eventLastSamps(ev)
            labelsArr(win) = numLabel(n_422_events(ev)); %Find the corresponding number for the label
        end
    end
end


%% Part4
clc;

bp_arr1 = zeros(windows,1); %Bandpower array.
bp_arr2 = zeros(windows,1); %Bandpower array.

freqRange1 = [0, 10];
freqRange2 = [10, 30];

meanfreq_arr =  zeros(windows,1); %meanfreq array.
medianfreq_arr =  zeros(windows,1); %medianfreq array.

for win = 1:windows
    %Computing the array of samples in window number win:
    T = (win-1)*winLen*(1-overlap/100)+1 : ...
        winLen+(win-1)*winLen*(1-overlap/100);
    T = round(T);

    bp_arr1(win) = bandpower(n_422(T),fs,freqRange1);
    bp_arr2(win) = bandpower(n_422(T),fs,freqRange2);

    meanfreq_arr(win) = meanfreq(n_422(T),fs);
    medianfreq_arr(win) = medfreq(n_422(T),fs);
end

figure('Name', 'Bandpower');
subplot(211)
plot(windowsArr, bp_arr1);
title("Bandpower for Frequency range ["+freqRange1(1)+" , "+freqRange1(2)+"]");
xlabel('Time [sec]');
ylabel('Bandpower');
subplot(212)
plot(windowsArr, bp_arr2);
title("Bandpower for Frequency range ["+freqRange2(1)+" , "+freqRange2(2)+"]");
xlabel('Time [sec]');
ylabel('Bandpower');

figure('Name', 'Meanfreq & Medianfreq');
subplot(211);
plot(windowsArr, meanfreq_arr);
title("Mean frequeancy");
xlabel('Time [sec]');
ylabel('Mean frequnecy (Hz)');

subplot(212);
plot(windowsArr, medianfreq_arr);
title("Median frequeancy");
xlabel('Time [sec]');
ylabel('Median frequency (Hz)');


%% part5

clc;

nbins = 15;
figure;
subplot(221);
histogram(bp_arr1(labelsArr == 1), nbins);
hold on; histogram(bp_arr1(labelsArr == 2), nbins);
title("Bandpower histogram for  Frequency range ["+freqRange1(1)+" , "+freqRange1(2)+"]");
xlabel("Bandpower"); ylabel("Count");
legend("Normal", "VFIB");

subplot(222);
histogram(bp_arr2(labelsArr == 1), nbins);
hold on; histogram(bp_arr2(labelsArr == 2), nbins);
title("Bandpower histogram for  Frequency range ["+freqRange2(1)+" , "+freqRange2(2)+"]");
xlabel("Bandpower"); ylabel("Count");
legend("Normal", "VFIB");

subplot(223);
histogram(meanfreq_arr(labelsArr == 1), nbins);
hold on; histogram(meanfreq_arr(labelsArr == 2), nbins);
title("Mean Frequency histogram");
xlabel("Mean Frequency"); ylabel("Count");
legend("Normal", "VFIB");

subplot(224);
histogram(medianfreq_arr(labelsArr == 1), nbins);
hold on; histogram(medianfreq_arr(labelsArr == 2), nbins);
title("Median Frequency histogram");
xlabel("Median Frequency"); ylabel("Count");
legend("Normal", "VFIB");


%% part 6
clc;

[alarm_medfreq,t] = va_detect_medfreq(n_422,fs);
alarm_medfreq(alarm_medfreq == 1) = 2; %Same as VFIB label
alarm_medfreq(alarm_medfreq == 0) = 1; %Same as Normal label

[alarm_bp,t] = va_detect_bandpower(n_422,fs);
alarm_bp(alarm_bp == 1) = 2; %Same as VFIB label
alarm_bp(alarm_bp == 0) = 1; %Same as Normal label


m_medfreq = confusionmat(labelsArr, alarm_medfreq);
m_bp = confusionmat(labelsArr, alarm_bp);

%****************PLotting Confusion matrix*********************
eventNames = ["None","Normal","VFIB","VT"];
pred_eventNames = ["Pred. None","Pred. Normal","Pred. VFIB","Pred. VT"];
figure;
heatmap(pred_eventNames, eventNames, m_medfreq);
title("Confusion Matrix using Median Frequency");

figure;
heatmap(pred_eventNames, eventNames, m_bp);
title("Confusion Matrix using Bandpower Frequency");
%**************************************************************

%Sensitivity: TP/P
sensitivity_medfreq = m_medfreq(3,3)/(m_medfreq(3,3)+m_medfreq(3,2))
sensitivity_bp = m_bp(3,3)/(m_bp(3,3)+m_bp(3,2))

%Specificity: TN/N
specificity_medfreq = m_medfreq(2,2)/(m_medfreq(2,2)+m_medfreq(2,3))
specificity_bp = m_bp(2,2)/(m_bp(2,2)+m_bp(2,3))

%Accuracy: (TP+TN)/TP+TN+FP+FN)
accuracy_medfreq = (m_medfreq(2,2) + m_medfreq(3,3)) / sum(sum(m_medfreq(2:3,2:3)))
accuracy_bp = (m_bp(2,2) + m_bp(3,3)) / sum(sum(m_bp(2:3,2:3)))


%% Part 7
clc;

max_arr = zeros(windows,1); %Max Amplitude array.
min_arr = zeros(windows,1); %Min Amplitude array.
p2p_arr = zeros(windows,1); %P2P Amplitude array.
locPeaks_arr = zeros(windows,1); %Local Amplitude peaks array.
numberOfzeroes_arr = zeros(windows,1); % Zero passings
var_arr =  zeros(windows,1); %Variance of Amp array.
zeroes_arr =  zeros(windows,1); %Number of zeroes array.

for win = 1:windows
    %Computing the array of samples in window number win:
    T = (win-1)*winLen*(1-overlap/100)+1 : ...
        winLen+(win-1)*winLen*(1-overlap/100);
    T = round(T);

    max_arr(win) = max(n_422(T));
    min_arr(win) = min(n_422(T));
    p2p_arr(win) = max_arr(win) - min_arr(win);
    locPeaks_arr(win) = mean(findpeaks(n_422(T)));
    numberOfzeroes_arr(win) = length(n_422(T)) - nnz(n_422(T));
    var_arr(win) = var(n_422(T));
end

figure('Name', 'Amplitude features1');
subplot(311)
plot(windowsArr, max_arr);
title("Maximum Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

subplot(312)
plot(windowsArr, min_arr);
title("Minimum Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

subplot(313)
plot(windowsArr, p2p_arr);
title("Peak to peak Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

figure('Name', 'Amplitude features1');
subplot(311);
plot(windowsArr, locPeaks_arr);
title("Mean of local peaks Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

subplot(312);
plot(windowsArr, numberOfzeroes_arr);
title("Number of zero passings");
xlabel('Time [sec]');
ylabel('Occurance');
grid minor

subplot(313);
plot(windowsArr, var_arr);
title("Variance of Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

%% Part 8
clc;

nbins = 10;
figure('Name', 'Histograms');
subplot(321);
histogram(max_arr(labelsArr == 1), nbins);
hold on; histogram(max_arr(labelsArr == 2), nbins);
title("Maximum Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(322);
histogram(min_arr(labelsArr == 1), nbins);
hold on; histogram(min_arr(labelsArr == 2), nbins);
title("Minimum Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(323);
histogram(p2p_arr(labelsArr == 1), nbins);
hold on; histogram(p2p_arr(labelsArr == 2), nbins);
title("Peak to peak Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(324);
histogram(locPeaks_arr(labelsArr == 1), nbins);
hold on; histogram(locPeaks_arr(labelsArr == 2), nbins);
title("Mean of local peaks Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(325);
histogram(numberOfzeroes_arr(labelsArr == 1), nbins);
hold on; histogram(numberOfzeroes_arr(labelsArr == 2), nbins);
title("Number of zero passings Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(326);
histogram(var_arr(labelsArr == 1), nbins);
hold on; histogram(var_arr(labelsArr == 2), nbins);
title("Variance of Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

%% Part 9&10
clc;

[alarm_mlp,t] = va_detect_mlp(n_422,fs);
alarm_mlp(alarm_mlp == 1) = 2; %Same as VFIB label
alarm_mlp(alarm_mlp == 0) = 1; %Same as Normal label

[alarm_max,t] = va_detect_max(n_422,fs);
alarm_max(alarm_max == 1) = 2; %Same as VFIB label
alarm_max(alarm_max == 0) = 1; %Same as Normal label


m_mlp = confusionmat(labelsArr, alarm_mlp);
m_max = confusionmat(labelsArr, alarm_max);

%****************PLotting Confusion matrix*********************
eventNames = ["None","Normal","VFIB","VT"];
pred_eventNames = ["Pred. None","Pred. Normal","Pred. VFIB","Pred. VT"];
figure('Name', 'Confusion_MLP');
heatmap(pred_eventNames, eventNames, m_mlp);
title("Confusion Matrix using Mean Local Peaks ");

figure('Name', 'Confusion_MAX');
heatmap(pred_eventNames, eventNames, m_max);
title("Confusion Matrix using Maximum Amplitude");
%**************************************************************

%Sensitivity: TP/P
sensitivity_mlp = m_mlp(3,3)/(m_mlp(3,3)+m_mlp(3,2))
sensitivity_max = m_max(3,3)/(m_max(3,3)+m_max(3,2))

%Specificity: TN/N
Specificity_mlp = m_mlp(2,2)/(m_mlp(2,2)+m_mlp(2,3))
Specificity_max = m_max(2,2)/(m_max(2,2)+m_max(2,3))

%Accuracy: (TP+TN)/TP+TN+FP+FN)
Accuracy_mlp = (m_mlp(2,2) + m_mlp(3,3)) / sum(sum(m_mlp(2:3,2:3)))
Accuracy_max = (m_max(2,2) + m_max(3,3)) / sum(sum(m_max(2:3,2:3)))


%% Part 11-3
clc;
n_424_eventFirstSamps = [1, 27249, 53673, 55134, 58288]; %Samples in n424 in which the events started
n_424_eventLastSamps = [27248, 53672, 55133, 58287, 75000]; %Samples in n424 in which the events ended

n_424_events = ["N","VFIB","NOISE","ASYS","NOD"]; %Array of events happened in n424


winLen = 10*fs; %The length of each window [samples].
overlap = 50; %Windows overlap, as a percentage of winLen.
totalLen = length(n_424); %Total length of the signal

windows = round((totalLen-winLen)/(winLen*(1-overlap/100)))+1; %Total number of windows
windowsArr = zeros(windows,1);

labelsArr = zeros(windows,1); %Time array for final image
lastSampArr = zeros(windows, 1); %The number of the last sample in each window
firstSampArr =  zeros(windows, 1); %The number of the first sample in each window

for win = 1:windows
    %Computing the array of samples in window number win:
    T = (win-1)*winLen*(1-overlap/100)+1 : ...
        winLen+(win-1)*winLen*(1-overlap/100);
    T = round(T);

    lastSampArr(win) = T(end);
    firstSampArr(win) = T(1);

    windowsArr(win) = mean(T)/fs;

    for ev = 1:length(n_424_eventFirstSamps)
        if firstSampArr(win) >= n_424_eventFirstSamps(ev) && lastSampArr(win) <= n_424_eventLastSamps(ev)
            labelsArr(win) = numLabel(n_424_events(ev)); %Find the corresponding number for the label
        end
    end
end


%% Part 11-4
clc;

bp_arr1 = zeros(windows,1); %Bandpower array.
bp_arr2 = zeros(windows,1); %Bandpower array.

freqRange1 = [0, 10];
freqRange2 = [10, 30];

meanfreq_arr =  zeros(windows,1); %meanfreq array.
medianfreq_arr =  zeros(windows,1); %medianfreq array.

for win = 1:windows
    %Computing the array of samples in window number win:
    T = (win-1)*winLen*(1-overlap/100)+1 : ...
        winLen+(win-1)*winLen*(1-overlap/100);
    T = round(T);

    bp_arr1(win) = bandpower(n_424(T),fs,freqRange1);
    bp_arr2(win) = bandpower(n_424(T),fs,freqRange2);

    meanfreq_arr(win) = meanfreq(n_424(T),fs);
    medianfreq_arr(win) = medfreq(n_424(T),fs);
end

figure('Name', 'Bandpower');
subplot(211)
plot(windowsArr, bp_arr1);
title("Bandpower for Frequency range ["+freqRange1(1)+" , "+freqRange1(2)+"]");
xlabel('Time [sec]');
ylabel('Bandpower');
subplot(212)
plot(windowsArr, bp_arr2);
title("Bandpower for Frequency range ["+freqRange2(1)+" , "+freqRange2(2)+"]");
xlabel('Time [sec]');
ylabel('Bandpower');

figure('Name', 'Meanfreq & Medianfreq');
subplot(211);
plot(windowsArr, meanfreq_arr);
title("Mean frequeancy");
xlabel('Time [sec]');
ylabel('Mean frequnecy (Hz)');

subplot(212);
plot(windowsArr, medianfreq_arr);
title("Median frequeancy");
xlabel('Time [sec]');
ylabel('Median frequency (Hz)');


%% part 11-5

clc;

nbins = 15;
figure;
subplot(221);
histogram(bp_arr1(labelsArr == 1), nbins);
hold on; histogram(bp_arr1(labelsArr == 2), nbins);
title("Bandpower histogram for  Frequency range ["+freqRange1(1)+" , "+freqRange1(2)+"]");
xlabel("Bandpower"); ylabel("Count");
legend("Normal", "VFIB");

subplot(222);
histogram(bp_arr2(labelsArr == 1), nbins);
hold on; histogram(bp_arr2(labelsArr == 2), nbins);
title("Bandpower histogram for  Frequency range ["+freqRange2(1)+" , "+freqRange2(2)+"]");
xlabel("Bandpower"); ylabel("Count");
legend("Normal", "VFIB");

subplot(223);
histogram(meanfreq_arr(labelsArr == 1), nbins);
hold on; histogram(meanfreq_arr(labelsArr == 2), nbins);
title("Mean Frequency histogram");
xlabel("Mean Frequency"); ylabel("Count");
legend("Normal", "VFIB");

subplot(224);
histogram(medianfreq_arr(labelsArr == 1), nbins);
hold on; histogram(medianfreq_arr(labelsArr == 2), nbins);
title("Median Frequency histogram");
xlabel("Median Frequency"); ylabel("Count");
legend("Normal", "VFIB");


%% part 11-6
clc;

[alarm_medfreq,t] = va_detect_medfreq2(n_424,fs);
alarm_medfreq(alarm_medfreq == 1) = 2; %Same as VFIB label
alarm_medfreq(alarm_medfreq == 0) = 1; %Same as Normal label

[alarm_meanfreq,t] = va_detect_meanfreq(n_424,fs);
alarm_meanfreq(alarm_meanfreq == 1) = 2; %Same as VFIB label
alarm_meanfreq(alarm_meanfreq == 0) = 1; %Same as Normal label


m_medfreq = confusionmat(labelsArr, alarm_medfreq);
m_meanfreq = confusionmat(labelsArr, alarm_meanfreq);

%****************PLotting Confusion matrix*********************
eventNames = ["None","Normal","VFIB"];
pred_eventNames = ["Pred. None","Pred. Normal","Pred. VFIB"];
figure;
heatmap(pred_eventNames, eventNames, m_medfreq);
title("Confusion Matrix using Median Frequency");

figure;
heatmap(pred_eventNames, eventNames, m_meanfreq);
title("Confusion Matrix using Mean Frequency");
%**************************************************************

%Sensitivity: TP/P
sensitivity_medfreq = m_medfreq(3,3)/(m_medfreq(3,3)+m_medfreq(3,2))
sensitivity_meanfrq = m_meanfreq(3,3)/(m_meanfreq(3,3)+m_meanfreq(3,2))

%Specificity: TN/N
specificity_medfreq = m_medfreq(2,2)/(m_medfreq(2,2)+m_medfreq(2,3))
specificity_meanfrq = m_meanfreq(2,2)/(m_meanfreq(2,2)+m_meanfreq(2,3))

%Accuracy: (TP+TN)/TP+TN+FP+FN)
accuracy_medfreq = (m_medfreq(2,2) + m_medfreq(3,3)) / sum(sum(m_medfreq(2:3,2:3)))
accuracy_meanfrq = (m_meanfreq(2,2) + m_meanfreq(3,3)) / sum(sum(m_meanfreq(2:3,2:3)))


%% Part 11-7
clc;

max_arr = zeros(windows,1); %Max Amplitude array.
min_arr = zeros(windows,1); %Min Amplitude array.
p2p_arr = zeros(windows,1); %P2P Amplitude array.
locPeaks_arr = zeros(windows,1); %Local Amplitude peaks array.
numberOfzeroes_arr = zeros(windows,1); % Zero passings
var_arr =  zeros(windows,1); %Variance of Amp array.
zeroes_arr =  zeros(windows,1); %Number of zeroes array.

for win = 1:windows
    %Computing the array of samples in window number win:
    T = (win-1)*winLen*(1-overlap/100)+1 : ...
        winLen+(win-1)*winLen*(1-overlap/100);
    T = round(T);

    max_arr(win) = max(n_424(T));
    min_arr(win) = min(n_424(T));
    p2p_arr(win) = max_arr(win) - min_arr(win);
    locPeaks_arr(win) = mean(findpeaks(n_424(T)));
    numberOfzeroes_arr(win) = length(n_424(T)) - nnz(n_424(T));
    var_arr(win) = var(n_424(T));
end

figure('Name', 'Amplitude features1');
subplot(311)
plot(windowsArr, max_arr);
title("Maximum Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

subplot(312)
plot(windowsArr, min_arr);
title("Minimum Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

subplot(313)
plot(windowsArr, p2p_arr);
title("Peak to peak Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

figure('Name', 'Amplitude features1');
subplot(311);
plot(windowsArr, locPeaks_arr);
title("Mean of local peaks Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

subplot(312);
plot(windowsArr, numberOfzeroes_arr);
title("Number of zero passings");
xlabel('Time [sec]');
ylabel('Occurance');
grid minor

subplot(313);
plot(windowsArr, var_arr);
title("Variance of Amplitude");
xlabel('Time [sec]');
ylabel('Amplitude');
grid minor

%% Part 11-8
clc;

nbins = 10;
figure('Name', 'Histograms');
subplot(321);
histogram(max_arr(labelsArr == 1), nbins);
hold on; histogram(max_arr(labelsArr == 2), nbins);
title("Maximum Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(322);
histogram(min_arr(labelsArr == 1), nbins);
hold on; histogram(min_arr(labelsArr == 2), nbins);
title("Minimum Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(323);
histogram(p2p_arr(labelsArr == 1), nbins);
hold on; histogram(p2p_arr(labelsArr == 2), nbins);
title("Peak to peak Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(324);
histogram(locPeaks_arr(labelsArr == 1), nbins);
hold on; histogram(locPeaks_arr(labelsArr == 2), nbins);
title("Mean of local peaks Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(325);
histogram(numberOfzeroes_arr(labelsArr == 1), nbins);
hold on; histogram(numberOfzeroes_arr(labelsArr == 2), nbins);
title("Number of zero passings Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

subplot(326);
histogram(var_arr(labelsArr == 1), nbins);
hold on; histogram(var_arr(labelsArr == 2), nbins);
title("Variance of Amplitude Histogram");
xlabel("Median Amplitude"); ylabel("Count");
legend("Normal", "VFIB");
grid minor

%% Part 11-9
clc;

[alarm_zero,t] = va_detect_zero(n_424,fs);
alarm_zero(alarm_zero == 1) = 2; %Same as VFIB label
alarm_zero(alarm_zero == 0) = 1; %Same as Normal label

[alarm_var,t] = va_detect_var(n_424,fs);
alarm_var(alarm_var == 1) = 2; %Same as VFIB label
alarm_var(alarm_var == 0) = 1; %Same as Normal label


m_zero = confusionmat(labelsArr, alarm_zero);
m_var = confusionmat(labelsArr, alarm_var);

%****************PLotting Confusion matrix*********************
eventNames = ["None","Normal","VFIB"];
pred_eventNames = ["Pred. None","Pred. Normal","Pred. VFIB"];
figure;
heatmap(pred_eventNames, eventNames, m_zero);
title("Confusion Matrix using number of zeroes passing ");

figure;
heatmap(pred_eventNames, eventNames, m_var);
title("Confusion Matrix using Variance of Amplitude");
%**************************************************************

%Sensitivity: TP/P
sensitivity_zero = m_zero(3,3)/(m_zero(3,3)+m_zero(3,2))
sensitivity_var = m_var(3,3)/(m_var(3,3)+m_var(3,2))

%Specificity: TN/N
specificity_zero = m_zero(2,2)/(m_zero(2,2)+m_zero(2,3))
specificity_var = m_var(2,2)/(m_var(2,2)+m_var(2,3))

%Accuracy: (TP+TN)/TP+TN+FP+FN)
accuracy_zero = (m_zero(2,2) + m_zero(3,3)) / sum(sum(m_zero(2:3,2:3)))
accuracy_var = (m_var(2,2) + m_var(3,3)) / sum(sum(m_var(2:3,2:3)))

%% Part 12
clc;

[alarm_zero,t] = va_detect_zero(n_424,fs);
alarm_zero(alarm_zero == 1) = 2; %Same as VFIB label
alarm_zero(alarm_zero == 0) = 1; %Same as Normal label

[alarm_mlp,t] = va_detect_mlp(n_424,fs);
alarm_mlp(alarm_mlp == 1) = 2; %Same as VFIB label
alarm_mlp(alarm_mlp == 0) = 1; %Same as Normal label


m_zero = confusionmat(labelsArr, alarm_zero);
m_mlp = confusionmat(labelsArr, alarm_mlp);

%****************PLotting Confusion matrix*********************
eventNames = ["None","Normal","VFIB"];
pred_eventNames = ["Pred. None","Pred. Normal","Pred. VFIB"];
figure;
heatmap(pred_eventNames, eventNames, m_zero);
title("Confusion Matrix using number of zeroes passing ");

figure('Name', 'Confusion_MLP');
heatmap(pred_eventNames, eventNames, m_mlp);
title("Confusion Matrix using Mean Local Peaks ");
%**************************************************************

%Sensitivity: TP/P
sensitivity_zero = m_zero(3,3)/(m_zero(3,3)+m_zero(3,2))
sensitivity_mlp = m_mlp(3,3)/(m_mlp(3,3)+m_mlp(3,2))

%Specificity: TN/N
specificity_zero = m_zero(2,2)/(m_zero(2,2)+m_zero(2,3))
specificity_mlp = m_mlp(2,2)/(m_mlp(2,2)+m_mlp(2,3))

%Accuracy: (TP+TN)/TP+TN+FP+FN)
accuracy_zero = (m_zero(2,2) + m_zero(3,3)) / sum(sum(m_zero(2:3,2:3)))
accuracy_mlp = (m_mlp(2,2) + m_mlp(3,3)) / sum(sum(m_mlp(2:3,2:3)))

%% Part 13
clc;

[alarm_zero,t] = va_detect_zero(n_422,fs);
alarm_zero(alarm_zero == 1) = 2; %Same as VFIB label
alarm_zero(alarm_zero == 0) = 1; %Same as Normal label

[alarm_mlp,t] = va_detect_mlp(n_424,fs);
alarm_mlp(alarm_mlp == 1) = 2; %Same as VFIB label
alarm_mlp(alarm_mlp == 0) = 1; %Same as Normal label


m_zero = confusionmat(labelsArr, alarm_zero);
m_mlp = confusionmat(labelsArr, alarm_mlp);

%****************PLotting Confusion matrix*********************
eventNames = ["None","Normal","VFIB"];
pred_eventNames = ["Pred. None","Pred. Normal","Pred. VFIB"];
figure;
heatmap(pred_eventNames, eventNames, m_zero);
title("Confusion Matrix using number of zeroes passing ");

figure('Name', 'Confusion_MLP');
heatmap(pred_eventNames, eventNames, m_mlp);
title("Confusion Matrix using Mean Local Peaks ");
%**************************************************************

%Sensitivity: TP/P
sensitivity_zero = m_zero(3,3)/(m_zero(3,3)+m_zero(3,2))
sensitivity_mlp = m_mlp(3,3)/(m_mlp(3,3)+m_mlp(3,2))

%Specificity: TN/N
specificity_zero = m_zero(2,2)/(m_zero(2,2)+m_zero(2,3))
specificity_mlp = m_mlp(2,2)/(m_mlp(2,2)+m_mlp(2,3))

%Accuracy: (TP+TN)/TP+TN+FP+FN)
accuracy_zero = (m_zero(2,2) + m_zero(3,3)) / sum(sum(m_zero(2:3,2:3)))
accuracy_mlp = (m_mlp(2,2) + m_mlp(3,3)) / sum(sum(m_mlp(2:3,2:3)))

%% Part 14
clc;
load('n_426.mat');
n_426_eventFirstSamps = [1, 26432]; %Samples in n426 in which the events started
n_426_eventLastSamps = [26431, 75000]; %Samples in n426 in which the events ended

n_426_events = ["N","VF"]; %Array of events happened in n426


winLen = 10*fs; %The length of each window [samples].
overlap = 50; %Windows overlap, as a percentage of winLen.
totalLen = length(n_426); %Total length of the signal

windows = round((totalLen-winLen)/(winLen*(1-overlap/100)))+1; %Total number of windows
windowsArr = zeros(windows,1);

labelsArr = zeros(windows,1); %Time array for final image
lastSampArr = zeros(windows, 1); %The number of the last sample in each window
firstSampArr =  zeros(windows, 1); %The number of the first sample in each window

for win = 1:windows
    %Computing the array of samples in window number win:
    T = (win-1)*winLen*(1-overlap/100)+1 : ...
        winLen+(win-1)*winLen*(1-overlap/100);
    T = round(T);

    lastSampArr(win) = T(end);
    firstSampArr(win) = T(1);

    windowsArr(win) = mean(T)/fs;

    for ev = 1:length(n_426_eventFirstSamps)
        if firstSampArr(win) >= n_426_eventFirstSamps(ev) && lastSampArr(win) <= n_426_eventLastSamps(ev)
            labelsArr(win) = numLabel(n_426_events(ev)); %Find the corresponding number for the label
        end
    end
end

numberOfzeroes_arr = zeros(windows,1); % Zero passings

for win = 1:windows
    %Computing the array of samples in window number win:
    T = (win-1)*winLen*(1-overlap/100)+1 : ...
        winLen+(win-1)*winLen*(1-overlap/100);
    T = round(T);

    numberOfzeroes_arr(win) = length(n_424(T)) - nnz(n_424(T));
end


% figure
% histogram(numberOfzeroes_arr(labelsArr == 1), nbins);
% hold on; histogram(numberOfzeroes_arr(labelsArr == 2), nbins);
% title("Number of zero passings Histogram");
% xlabel("Median Amplitude"); ylabel("Count");
% legend("Normal", "VFIB");
% grid minor


[alarm_zero,t] = va_detect_zero2(n_426,fs);
alarm_zero(alarm_zero == 1) = 2; %Same as VF label
alarm_zero(alarm_zero == 0) = 1; %Same as Normal label

m_zero = confusionmat(labelsArr, alarm_zero);

%****************PLotting Confusion matrix*********************
eventNames = ["None","Normal","VF"];
pred_eventNames = ["Pred. None","Pred. Normal","Pred. VF"];
figure;
heatmap(pred_eventNames, eventNames, m_zero);
title("Confusion Matrix using number of zeroes passing ");

%**************************************************************

%Sensitivity: TP/P
sensitivity_zero = m_zero(3,3)/(m_zero(3,3)+m_zero(3,2))

%Specificity: TN/N
specificity_zero = m_zero(2,2)/(m_zero(2,2)+m_zero(2,3))

%Accuracy: (TP+TN)/TP+TN+FP+FN)
accuracy_zero = (m_zero(2,2) + m_zero(3,3)) / sum(sum(m_zero(2:3,2:3)))

%% functions

function num = numLabel(label)
%This function converts each arrhithmia label from a string into a number
    switch label
        case 'N'
            num = 1;
        case 'VFIB'
            num = 2;
        case 'VF'
            num = 2;
        case 'VT'
            num = 3;
        case 'NOISE'
            num = 4;
        otherwise
            num = 0; %NONE
    end
end













