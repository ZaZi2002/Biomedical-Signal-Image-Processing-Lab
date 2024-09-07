%% Part3: a
clc;
close all;
data = load('FiveClass_EEG.mat'); %Loading data
X = data.X; %Main data
y = data.y; %Trials Onset
trial = data.trial; %Trials type

fs = 256; %Sampling frequency

order = 6; %Filters order
Apass = 1; % Passband Ripple (dB)

%*********Delta bandpass**********
Fpass1 = 1; %Cutoff frequency1
Fpass2 = 4; %Cutoff frequency2

delta_X = BP_Filt(X, order, Fpass1, Fpass2, Apass, fs); %Filtering the data
%*********************************

%*********Theta bandpass**********
Fpass1 = 4;
Fpass2 = 8;

theta_X = BP_Filt(X, order, Fpass1, Fpass2, Apass, fs);
%*********************************

%*********Alpha bandpass**********
Fpass1 = 8;
Fpass2 = 13;

alpha_X = BP_Filt(X, order, Fpass1, Fpass2, Apass, fs);
%*********************************

%*********Beta bandpass***********
Fpass1 = 13;
Fpass2 = 30;

beta_X = BP_Filt(X, order, Fpass1, Fpass2, Apass, fs);
%*********************************




%*****************Testing Filters******************
chan = 1; %Selected channel to plot
T = 1 : (5*fs); %Selected Time interval to plot

t = 1/fs:1/fs:length(T)/fs; %Defining time vector


figure;

subplot(511);
plot(t, X(T, chan));
title("Signal X (Raw). Channel "+chan);

subplot(512);
plot(t, delta_X(T, chan));
title("Delta Signal. Channel "+chan);

subplot(513);
plot(t, theta_X(T, chan));
title("Theta Signal. Channel "+chan);

subplot(514);
plot(t, alpha_X(T, chan));
title("Alpha Signal. Channel "+chan);

subplot(515);
plot(t, beta_X(T, chan));
title("Beta Signal. Channel "+chan);
%***************************************************




%% Part3: b
clc;

T = 10; %TIme interval for each epoch [sec]

%Raw data epoching
epoched_X = epoching(X, trial, fs, T);

%Delta epoching
epoched_delta_X = epoching(delta_X, trial, fs, T);

%Theta epoching
epoched_theta_X = epoching(theta_X, trial, fs, T);

%Alpha epoching
epoched_alpha_X = epoching(alpha_X, trial, fs, T);

%Beta epoching
epoched_beta_X = epoching(beta_X, trial, fs, T);


%% Part3: c
clc;


%Raw data Power
epoched_X_pow = epoched_X.^2;

%Delta Power
epoched_delta_X_pow = epoched_delta_X.^2;

%Theta Power
epoched_theta_X_pow = epoched_theta_X.^2;

%Alpha Power
epoched_alpha_X_pow = epoched_alpha_X.^2;

%Beta Power
epoched_beta_X_pow = epoched_beta_X.^2;




%% Part3: d
clc;
close all;


%Initializing desired signals:
delta_X_avg = zeros(size(epoched_X,1), size(epoched_X,2), 5);
theta_X_avg = zeros(size(epoched_X,1), size(epoched_X,2), 5);
alpha_X_avg = zeros(size(epoched_X,1), size(epoched_X,2), 5);
beta_X_avg = zeros(size(epoched_X,1), size(epoched_X,2), 5);

for i = 1:5 %For each class
    class = find(y == i); %Finding the trials corresponding to the current class

    %Averaging over se;ected trials, and saving the data in the output matrices:
    delta_X_avg(:,:,i) = squeeze(mean(epoched_delta_X_pow(:, :, class), 3));
    theta_X_avg(:,:,i) = squeeze(mean(epoched_theta_X_pow(:, :, class), 3));
    alpha_X_avg(:,:,i) = squeeze(mean(epoched_alpha_X_pow(:, :, class), 3));
    beta_X_avg(:,:,i) = squeeze(mean(epoched_beta_X_pow(:, :, class), 3));

end

%% Part3: e
newWin = ones(1,200)/sqrt(200);

%Producing zero matrixes
delta_X_avg_filtered = zeros(2560,30,5);
theta_X_avg_filtered = zeros(2560,30,5);
alpha_X_avg_filtered = zeros(2560,30,5);
beta_X_avg_filtered = zeros(2560,30,5);

for a = 1:5
    for b = 1:30
        %Convolving newWin with signals
        delta_X_avg_filtered(:,b,a) = conv(delta_X_avg(:,b,a),newWin,'same');
        theta_X_avg_filtered(:,b,a) = conv(theta_X_avg(:,b,a),newWin,'same');
        alpha_X_avg_filtered(:,b,a) = conv(alpha_X_avg(:,b,a),newWin,'same');
        beta_X_avg_filtered(:,b,a) = conv(beta_X_avg(:,b,a),newWin,'same');
    end
end

%% Part3: f
fs = 256;
t = 0.0001:1/fs:10;
n = 16; %Channel number

figure('Name',"Part3_f_delta");
for j = 1:5
    plot(t,delta_X_avg_filtered(:,n,j));
    hold on
    grid minor
    title('Delta band');
    xlabel('Time(s)');
end
legend('Class1','Class2','Class3','Class4','Class5');
saveas(gcf,"Part3_Delta_Band.png");

figure('Name',"Part3_f_theta");
for j = 1:5
    plot(t,theta_X_avg_filtered(:,n,j));
    hold on
    grid minor
    title('Theta band');
    xlabel('Time(s)');
end
legend('Class1','Class2','Class3','Class4','Class5');
saveas(gcf,"Part3_Theta_Band.png");

figure('Name',"Part3_f_alpha");
for j = 1:5
    plot(t,alpha_X_avg_filtered(:,n,j));
    hold on
    grid minor
    title('Alpha band');
    xlabel('Time(s)');
end
legend('Class1','Class2','Class3','Class4','Class5');
saveas(gcf,"Part3_Alpha_Band.png");

figure('Name',"Part3_f_betha");
for j = 1:5
    plot(t,beta_X_avg_filtered(:,n,j));
    hold on
    grid minor
    title('Betha band');
    xlabel('Time(s)');
end
legend('Class1','Class2','Class3','Class4','Class5');
saveas(gcf,"Part3_Betha_Band.png");

%% functions

function filt_data = BP_Filt(X, order, Fpass1, Fpass2, Apass, fs)
%This function Filters the data matrix X using input parameters.
%Assumption: dim(X) = samples*channels

    % Construct an FDESIGN object and call its CHEBY1 method.
    h = fdesign.bandpass('N,Fp1,Fp2,Ap', order, Fpass1, Fpass2,...
    Apass, fs);
    Hd = design(h, 'cheby1');

    filt_data = zeros(size(X)); %Output signal
    for c=1:size(X,2) %For each channel
        filt_data(:,c) = filter(Hd,X(:,c)); 
    end
    
end





function epoched_data = epoching(filt_X, trial, fs, T)
%This function epochs the data using trial onsets defined in trial.

epoched_data = zeros(T*fs, size(filt_X,2), length(trial)); %Output epoched signal

    for i=1:length(trial) %For each trial
        epoched_data(:,:,i) = filt_X(trial(i)+1: trial(i)+fs*T,:); %Keep T seconds after trial onset
    end
end




