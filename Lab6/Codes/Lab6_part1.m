%% part1
clear, clc; close all;

load('ElecPosXYZ'); %Loading electrode Positions and names

%Forward Matrix
ModelParams.R = [8 8.5 9.2] ; % Radius of diffetent layers
ModelParams.Sigma = [3.3e-3 8.25e-5 3.3e-3]; 
ModelParams.Lambda = [.5979 .2037 .0237];
ModelParams.Mu = [.6342 .9364 1.0362];

Resolution = 1; %[cm]
[LocMat,GainMat] = ForwardModel_3shell(Resolution, ModelParams) ;
save('GainMat.mat', 'GainMat'); %Saving gain matrix

%Scattering the location of the dipoles:
f = figure;
scatter3(LocMat(1,:), LocMat(2,:), LocMat(3,:));
title('Dipoles Location');
xlabel('X [cm]'); ylabel('Y [cm]'); zlabel('Z [cm]');

%% part2
clc;
ElecPos_x = zeros(size(ElecPos)); %Electrodes x coordinate
ElecPos_y = zeros(size(ElecPos)); %Electrodes y coordinate
ElecPos_z = zeros(size(ElecPos)); %Electrodes z coordinate

ElecName = string(size(ElecPos)); %Electrodes names

for Elec = 1:length(ElecPos)
    ElecPos_x(Elec) = ElecPos{Elec}.XYZ(1) * ModelParams.R(3);
    ElecPos_y(Elec) = ElecPos{Elec}.XYZ(2) * ModelParams.R(3);
    ElecPos_z(Elec) = ElecPos{Elec}.XYZ(3) * ModelParams.R(3);
    ElecName(Elec) = ElecPos{Elec}.Name;
end

%Adding the location of the electrodes to the previous figure:
hold on;
scatter3(ElecPos_x, ElecPos_y, ElecPos_z, 'filled');
text(ElecPos_x, ElecPos_y, ElecPos_z, ElecName);


%% part3
clc;

%Choosing a random dipole:
% % dip = randi(size(LocMat,2))
dip = 894;

dip_r = sqrt(sum(abs(LocMat).^2, 1)); %Dipoles radius
% %************For the last part of the questions, uncomment on of the following lines*************
dip = find(dip_r == max(dip_r), 1); %On the cortex surface

% dip = find(LocMat(1,:) == 0 & LocMat(2,:) == min(LocMat(2,:)) & ...
%     LocMat(3,:) == min(LocMat(3,:))); %On the cortex and temporal
% 
% dip = find(LocMat(1,:) == 0 & LocMat(2,:) == 0 & ...
%     LocMat(3,:) >= 3 & LocMat(3,:) <= 4); %In the deep
% %*********************************************************************************************

hold on;

%Setting the location of the dipole:
x = LocMat(1,dip);
y = LocMat(2,dip);
z = LocMat(3,dip);

pointLen = sqrt(x^2+y^2+z^2); %Dipole radius

%Setting the final point of the momentum, considering unit length for the vector:
x_dst = x/pointLen;
y_dst = y/pointLen;
z_dst = z/pointLen;

%Adding the momentum vector to the previous model:
quiver3(x, y, z, x_dst, y_dst, z_dst, 'LineWidth',2);


%% part4
load('Interictal.mat');

%Randomly choosing one source for the selected dipole:
% % src = randi(size(Interictal,1))
src = 16;

%*********For the last part, you can uncomment this part, or just use the previous src=16*************
% src = randi(size(Interictal,1));
%*********************************************************************

slct_src = Interictal(src, :);

%Creating Momentum matrix for the selected dipole and source:
Q = zeros(3, size(Interictal,2)); 
Q(1,:) = slct_src .* x/pointLen;
Q(2,:) = slct_src .* y/pointLen;
Q(3,:) = slct_src .* z/pointLen;

slct_GainMat = GainMat(:,(dip-1)*3+1:dip*3); %Selecting part of the gain matrix which corresponds with the selected dipole

M = slct_GainMat*Q; %Creating Potential matrix.

%**************Plotting potential of the electrodes:*************
figure;
plots_offset = max(max(M)); 
for i = 1:size(M,1)
    plot(M(i, :) + (i-1)*plots_offset, 'LineWidth',1);
    hold on;
    text(0,(i-1)*plots_offset, ElecName(i));
end

title("Electrodes signal. Dipole number "+dip+" and source number "+src);
xlabel("Sample");
%***************************************************************
%% part5: Averaging over spikes

peak_samps = [];
peak_th = ones(size(M,1),1) .* (mean(M,2)+3*std(M,[],2)); %Peak detection threshold for electrodes

%Computing peak samples for each electrode:
for i = 1:size(M,1)
    [peaks, locs] = findpeaks(M(i,:), 'MinPeakHeight', peak_th(i));
    peak_samps(i,:) = locs;
end

ElecPot = zeros(size(M,1), 1); %Final vector. Dim = electrodes*1

for i = 1:size(peak_samps,1) %For each electrode
    for j = 1:size(peak_samps, 2) %For each peak
        ElecPot(i) = mean(M(i, peak_samps(i,:)-3:peak_samps(i,:)+3));
    end
end

figure;
Display_Potential_3D(ModelParams.R(3),ElecPot);
title("3D potential Display at spike times. Dipole number "+dip+" and source number "+src);
xlabel("X [cm]"); ylabel("Y [cm]"); zlabel("Z [cm]");
%% Part6: Predicting matrix Q
clc;

%Setting initial parameters:
alpha = 0.5;
N = size(M,1);

%Computing predicted momentum matrix:
Q_pred = GainMat.' * inv(GainMat * (GainMat.') + alpha*eye(N)) * M;


%% Part7: Predicting Dipole
clc;

dipNum = size(Q_pred,1)/3; %The number of dipoles
square_Q_pred = Q_pred.^2;

amp_arr = zeros(1, dipNum); %Array of momentum amplitudes

for i = 0:dipNum-1
    amp_arr(i+1) = sum(sum(square_Q_pred((3*i+1):(3*(i+1)), :)));
end

pred_dip = find(amp_arr == max(amp_arr)); %Predict the dipole (which maximizes the momentum amplitude)


%Setting the location of the dipole:
pred_x = LocMat(1,pred_dip);
pred_y = LocMat(2,pred_dip);
pred_z = LocMat(3,pred_dip);

[pred_fi, pred_thetaPrime, pred_r] = cart2sph(pred_x, pred_y, pred_z); %Computing radius and angle of the predicted dipole

pred_pointLen = sqrt(pred_x^2+pred_y^2+pred_z^2); %Dipole radius

%Setting the final point of the momentum, considering unit length for the vector:
pred_x_dst = pred_x/pointLen;
pred_y_dst = pred_y/pointLen;
pred_z_dst = pred_z/pointLen;

%Adding the momentum vector to the previous model:
figure(f);
hold on;
quiver3(pred_x, pred_y, pred_z, pred_x_dst, pred_y_dst, pred_z_dst, 'LineWidth',2);


%Displaying The location of the dipole (fi, pi-theta, r):
disp("Predicted dipole r = "+pred_r);
disp("Predicted dipole fi = "+pred_fi);
disp("Predicted dipole thetaPrime = "+pred_thetaPrime);


%% Part8: computing error

[fi, thetaPrime, r] = cart2sph(x, y, z); %Computing radius and angle of the main dipole

fi_err = pred_fi - fi;
thetaPrime_err = pred_thetaPrime - thetaPrime;
dist_err = sqrt((x-pred_x)^2 + (y-pred_y)^2 + (z-pred_z)^2);


%Displaying errors:
disp("Dipole distance error = "+dist_err);
disp("Dipole fi error = "+fi_err);
disp("Dipole thetaPrime error = "+thetaPrime_err);




