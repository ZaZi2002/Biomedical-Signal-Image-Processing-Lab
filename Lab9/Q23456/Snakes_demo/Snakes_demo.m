%% Snakes Demo: 
clear; close all; clc;

%% Read in an image
f = double(imread('brain.jpg'));
f = f(:,:,1);


%% Set initial snake
% in order to start with a circle around the center of the image we first
% need to find the center pixel (the center of the image)
center = floor((size(f)+1)/2);
% generate a bunch of theta values to be used for determining the
% initial x and y coordinates of the snake nodes
numNodes = 400;
theta = 0:(2*pi)/numNodes:(2*pi*(1-1/numNodes));

% set up the original nodes in a circle surrounding the brain
if 1
    % Start from scratch
    x0 = 160*cos(theta)+center(1);
    y0 = 120*sin(theta)+center(2);
else
    % Use previous snake (already in memory)
    x0 = x';
    y0 = y';
end

% Close the loop (add an extra point to x0 and y0)
plotX0 = x0;
plotX0(end+1) = x0(1);
plotY0 = y0;
plotY0(end+1) = y0(1);

%% display the initial image
figure(2);
imshow(f,[]);
hold on;
plot(plotY0,plotX0,'-y');


%% Run the Snakes method
% close in on the contour of the brain
[x, y] = Snake(f,x0,y0);


%% display the final result
plotX = x;
plotX(end+1) = x(1);
plotY = y;
plotY(end+1) = y(1);
imshow(f,[]);
hold on;
plot(plotY,plotX,'-y',plotY0,plotX0,'-r');

% the reverse order of printing is due to the fact that matlab and image
% coordinates are different

hold off;
