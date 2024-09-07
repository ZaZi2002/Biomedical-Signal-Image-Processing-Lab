clc
close all
clear all

%% Part1
load('SSVEP_EEG.mat');

% Bandpass filter between 1 - 40
fs = 250; % Sampling Frequency

[b1,a1] = butter(30,40/(fs/2),'low'); % Butterworth lowpass filter of order 30
[b2,a2] = butter(6,1/(fs/2),'high'); % Butterworth highpass filter of order 6
figure('Name','Part1');
t = 0.001 : 1/fs : length(SSVEP_Signal(1,:))/fs;  % EEG time length

for i = 1:6
    filtered_SSVEP_Signal(i,:) = filter(b1,a1,SSVEP_Signal(i,:));
    filtered_SSVEP_Signal(i,:) = filter(b2,a2,filtered_SSVEP_Signal(i,:));

    subplot(6,2,(2*i-1));
    plot(t,SSVEP_Signal(i,:));
    grid on;
    xlim('tight');
    title(" SSVEP channels " +i);
    xlabel('Time (s)')

    subplot(6,2,(2*i));
    plot(t,filtered_SSVEP_Signal(i,:));
    grid on;
    xlim('tight');
    title(" SSVEP filtered channels " +i);
    xlabel('Time (s)')
end

%% Part2
for i = 1:6
    for j = 1:15
        events_channels(i,j,:) = filtered_SSVEP_Signal(i,Event_samples(j) + 1:Event_samples(j) + 5*fs);
    end
end

%% Part3
t = 0.001 : 1/fs : 5;  % event lengths
for i = 1:15
    figure('Name',"Part3_" +i)
    for j = 1:6
        events_pwelch(j,i,:) = fftshift(pwelch(squeeze(events_channels(j,i,:))));
        N = length(events_pwelch(j,i,:));
        f = fs*(-N/2:N/2-1)/N;
        plot(f,squeeze(events_pwelch(j,i,:)));
        hold on;
        xlim([0,40]);
        xlabel('Frequency (Hz)')
        title("Event" + i);
    end
    legend('Channel1','Channel2','Channel3','Channel4','Channel5','Channel6');
    grid minor;
    saveas(gcf,"Part3_Event" + i + ".png");
end

