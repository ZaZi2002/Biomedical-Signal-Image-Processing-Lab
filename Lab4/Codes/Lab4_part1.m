%% Part1: a
clc; close all;

erp_data = cell2mat(struct2cell(load('ERP_EEG.mat'))); %Loading data
fs = 240; %Sampling frequency
N = 100:100:2500; %The number of trials for averaging
t = 1/fs:1/fs:size(erp_data,1)/fs; %Time vector (For plots)

%Plotting ERP for each value of N:
for i = 1:length(N)
    figure;
    plot(t, mean(erp_data(:, 1:N(i)),2));
    title("ERP. N = "+N(i));
    xlabel('Time [s]');
    ylabel('Potential [uV]');
    savefig("Q1_a_N="+N(i)+".fig");
end


%% Part1: b
close all;

N = 1:2550; %The number of trials for averaging

figure;

sample = zeros(size(N)); %Array of maximum absolute values 

%Computing maximum absolute value for each N:
for i = 1:length(N)
    sample(i) = max(abs(mean(erp_data(:, 1:N(i)), 2)));
end

plot(N, sample, 'LineWidth',1);
title('ERP maximum abs vs N');
xlabel('N');
ylabel('ERP maximum abs');
xlim([min(N)-1, max(N)]);


%% Part1: c
close all;

N = 1:2550;
 
rms_arr = zeros(1, length(N)-1); %Array of RMS values
for i=2:length(N)
    erp_sig1 = mean(erp_data(:, 1:N(i)), 2); %ERP signal using N trials
    erp_sig2 = mean(erp_data(:, 1:N(i-1)), 2); %ERP signal using N-1 trials
    
    rms_arr(i-1) = rms(erp_sig1 - erp_sig2); %Computing RMS
end

figure;
plot(N(2:end), rms_arr, 'LineWidth',1);
title('RMS vs Number of averaged Trials (i)');
xlabel('i');
ylabel('RMS Value');
xlim([min(N), max(N)]);


%% Part1: d & e
clc;
close all;

N0 = 600;
N = 2550;

%**************Plotting ERP for different number of trials:**************
figure;
plot(t, mean(erp_data(:, 1:N0),2), 'LineWidth',1); %For N = N0
legend_str = "N = "+N0; %legend_str: The array of strings which are going to be shown as legend

hold on;
plot(t, mean(erp_data(:, 1:N),2), 'LineWidth',1); %For N = 2550
legend_str = [legend_str, "N = "+N];

plot(t, mean(erp_data(:, 1:round(N0/3)),2), 'LineWidth',1); %For N = N0/3
legend_str = [legend_str, "N = "+round(N0/3)];

N0_arr = randperm(N, N0);
plot(t, mean(erp_data(:, N0_arr),2), 'LineWidth',1);  %For N = N0 (randomly chosen)
legend_str = [legend_str, "N = "+N0 + " (random)"];

N0_arr = randperm(N, round(N0/3));
plot(t, mean(erp_data(:, N0_arr),2), 'LineWidth',1);  %For N = N0/3 (randomly chosen)
legend_str = [legend_str, "N = "+round(N0/3) + " (random)"];

legend(legend_str); %Setting legend for different values of N

title("ERP for different values of N");
xlabel('Time [s]');
ylabel('Potential [uV]');
savefig("Q1_e.fig");
%***********************************************************************

