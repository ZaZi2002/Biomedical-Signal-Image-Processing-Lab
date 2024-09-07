%% Q1
clc; 
clear;
close all;

img = imread('S3_Q1_utils\thorax_t1.jpg'); %Loading Image

img = double(img(:,:,1));

%Choosing the central points of the organs (visually)
lung1_center = [91; 95]; 
lung2_center = [90; 175];
liver_center = [151; 90];


%Finding the values of the central points:
lung1_val = img(lung1_center(1), lung1_center(2));
lung2_val = img(lung2_center(1), lung2_center(2));
liver_val = img(liver_center(1), liver_center(2));


%Setting a threshold for comparing the intensity of adjacent points with the center:
lung1_thr = 0.5*lung1_val;
lung2_thr = 0.5*lung2_val;
liver_thr = 0.25*liver_val;


figure;
imshow(uint8(img));
title("Original Image");

%Setting a mask based on defined thresholds:
lung1_mask = abs(img - lung1_val) <= lung1_thr;
lung2_mask = abs(img - lung2_val) <= lung2_thr;
liver_mask = abs(double(liver_val) - double(img)) <= liver_thr;

%Points that are considered as each organ (1 if it is considered and 0 if it is not)
lung1_points = zeros(size(img));
lung2_points = zeros(size(img));
liver_points = zeros(size(img));

lung1_points(lung1_center(1), lung1_center(2)) = 1;
lung2_points(lung2_center(1), lung2_center(2)) = 1;
liver_points(liver_center(1), liver_center(2)) = 1;


%**********************For right lung:***************************
tmp = 0;
while tmp < length(find(lung1_points))
    tmp = length(find(lung1_points)); %Previous number of points for the organ
    [lung1_points_y, lung1_points_x] = find(lung1_points == 1); %Currently selected points
    current_points = [lung1_points_y.';lung1_points_x.'];

    up_points = [lung1_points_y.'-1;lung1_points_x.']; %one pixel upper than current points
    down_points = [lung1_points_y.'+1;lung1_points_x.']; %one pixel lower than current points
    left_points = [lung1_points_y.';lung1_points_x.'-1]; %on the left of the current points
    right_points = [lung1_points_y.';lung1_points_x.'+1]; %on the right of the current points

    for j = 1:size(lung1_points_y)

        %Setting all adjacent points to 1:
        lung1_points(up_points(1,j), up_points(2,j)) = 1;
        lung1_points(down_points(1,j), down_points(2,j)) = 1;
        lung1_points(left_points(1,j), left_points(2,j)) = 1;
        lung1_points(right_points(1,j), right_points(2,j)) = 1;
    end

    %Comparing current points with the defined mask:
    lung1_points = lung1_points.*lung1_mask;
end
%******************************************************************


%************************For left lung:***************************
tmp = 0;
i = 0;
while tmp < length(find(lung2_points))
    i = i+1;
    tmp = length(find(lung2_points));
    [lung2_points_y, lung2_points_x] = find(lung2_points == 1); %Currently selected points
    current_points = [lung2_points_y.';lung2_points_x.'];

    up_points = [lung2_points_y.'-1;lung2_points_x.'];
    down_points = [lung2_points_y.'+1;lung2_points_x.'];
    left_points = [lung2_points_y.';lung2_points_x.'-1];
    right_points = [lung2_points_y.';lung2_points_x.'+1];

for j = 1:size(lung2_points_y)
    lung2_points(up_points(1,j), up_points(2,j)) = 1;
    lung2_points(down_points(1,j), down_points(2,j)) = 1;
    lung2_points(left_points(1,j), left_points(2,j)) = 1;
    lung2_points(right_points(1,j), right_points(2,j)) = 1;
                    
end

    lung2_points = lung2_points.*lung2_mask;
end
%******************************************************************


%************************For liver:********************************
r_thr = 125;

tmp = 0;
while tmp < length(find(liver_points))
    tmp = length(find(liver_points));
    disp(tmp);
    [liver_points_y, liver_points_x] = find(liver_points == 1); %Currently selected points
    current_points = [liver_points_y.';liver_points_x.'];


    up_points = [liver_points_y.'-1;liver_points_x.'];
    down_points = [liver_points_y.'+1;liver_points_x.'];
    left_points = [liver_points_y.';liver_points_x.'-1];
    right_points = [liver_points_y.';liver_points_x.'+1];

    down_points(:, down_points(1,:)>255) = [];
    for j = 1:size(liver_points_y)
        liver_points(up_points(1,j), up_points(2,j)) = 1;
        liver_points(down_points(1,j), down_points(2,j)) = 1;
        liver_points(left_points(1,j), left_points(2,j)) = 1;
        liver_points(right_points(1,j), right_points(2,j)) = 1;
    end

    liver_points = liver_points.*liver_mask;

    %If we're going too far away, stop the exploration
    out_of_range_points = find(abs(current_points(1,:) - liver_center(1)) + abs(current_points(2,:) - liver_center(2)) > r_thr);
    if ~isempty(out_of_range_points)
        liver_points(out_of_range_points) = 0;
        break;
    end
end
%******************************************************************


figure;

subplot(121);
final_img1 = imread('S3_Q1_utils\thorax_t1.jpg');

added_lung1 = uint8(zeros(size(final_img1)));
added_lung1(:,:,2) = 255*uint8(lung1_points);

added_lung2 = uint8(zeros(size(final_img1)));
added_lung2(:,:,2) = 255*uint8(lung2_points);

final_img1 = final_img1 + added_lung1 + added_lung2;
imshow(final_img1);
title("Lungs Detected");


subplot(122);
final_img2 = imread('S3_Q1_utils\thorax_t1.jpg');

added_liver = uint8(zeros(size(final_img2)));
added_liver(:,:,2) = 255*uint8(liver_points);

final_img2 = final_img2 + added_liver;
imshow(final_img2);
title("Liver Detected");





%% functions

function newPoints = checkAdjacents(centerP, lung1_center, lung1_val, lung1_thr, img, checked_points)
%Output Dim = 2*N
    if (centerP(1) == lung1_center(1) && centerP(2) == lung1_center(2))
        newPoints = [];
        disp(newPoints)
    else
        newPoints = checked_points;
    end

    leftP = [centerP(1); centerP(2)-1];
    rightP = [centerP(1); centerP(2)+1];
    upP = [centerP(1)-1; centerP(2)];
    downP = [centerP(1)+1; centerP(2)];

    disp(centerP)

    find(checked_points(1, :) == leftP(1));

    if abs(img(leftP(2), leftP(1)) - lung1_val) <= lung1_thr
        newPoints = [newPoints, leftP];
        checked_points = newPoints;
        newPoints = [newPoints, checkAdjacents(leftP, lung1_center, lung1_val, lung1_thr, img, checked_points)];
    end
    if abs(img(rightP(2), rightP(1)) - lung1_val) <= lung1_thr
        newPoints = [newPoints, rightP];
        checked_points = newPoints;
        newPoints = [newPoints, checkAdjacents(rightP, lung1_center, lung1_val, lung1_thr, img, checked_points)];
    end
    if abs(img(upP(2), upP(1)) - lung1_val) <= lung1_thr
        newPoints = [newPoints, upP];
        checked_points = newPoints;
        newPoints = [newPoints, checkAdjacents(upP, lung1_center, lung1_val, lung1_thr, img, checked_points)];
    end
    if abs(img(downP(2), downP(1)) - lung1_val) <= lung1_thr
        newPoints = [newPoints, downP];
        checked_points = newPoints;
        newPoints = [newPoints, checkAdjacents(downP, lung1_center, lung1_val, lung1_thr, img, checked_points)];
    end
end

