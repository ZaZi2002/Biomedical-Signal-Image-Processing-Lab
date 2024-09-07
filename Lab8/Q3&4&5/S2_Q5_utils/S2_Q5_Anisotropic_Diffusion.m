%% - Anisotropic Diffusion Demo

clear; close all; clc;

%% Set some parameters
addpath('./S2_Q5_utils')

method = 2; % 0 for blur, 1 for simple AD, 2 for sophisticated AD
K = 3;  % Try changing K (edge threshold): try 3, 5, 10

%% Read an image
f = imread('t1.jpg');
f = double( f(:,:,1) );
f = f + randn(size(f))*10; % Add noise
f_orig = f;

f_comp = f_orig; % used for display

%% PDE time-stepping parameters
delta_t = 0.1; % Time step
t_end = 30; % Try changing the termination time
lambda = 1; % diffusion scaling factor


counter = 0;
for t = 0:delta_t:t_end
    
    [dfdc dfdr] = gradient(f);
    
    gradmag_f2 = dfdr.^2 + dfdc.^2;
    
    div_f = (circshift(dfdr,[-1 0])-circshift(dfdr,[1 0]) + circshift(dfdc,[0 -1])-circshift(dfdc,[0 1])) / 2;
    
    % Try changing the edge-stopping function
    c = 1 ./ (1+gradmag_f2/(K/2)^2);
    %c = exp(-(gradmag_f2/K^2));
    if method==0
        c = ones(size(c))/5; % simple blurring (no spatial dependence)
    end
    
    % Right-hand side of PDE (for simple model)
    rhs = c.*div_f;
    
    % Include another term for the sophisticated version
    if method==2
        [dcdc dcdr] = gradient(c);
        rhs = rhs + dcdc.*dfdc + dcdr.*dfdr;
    end
    
    % Take a step in time
    f = f + lambda * delta_t * rhs;
    
    % Display every 10th iteration
    % Left half = original, right half = diffused
    if mod(counter,10)==0
        blah = round(size(f,2)/2);
        f_comp(:,blah:end) = f(:,blah:end);
        imshow(f_comp,[0 255]);
        title(['time = ' num2str(t) ' s']);
        drawnow;
        %pause;
    end
    counter = counter + 1;
    
end



