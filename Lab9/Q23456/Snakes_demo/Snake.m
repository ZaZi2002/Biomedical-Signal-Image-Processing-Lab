% Lab 4 Question 1 (a)
% Active contour (snake) function

function [x, y] = Snake(f,x0,y0)

    figure(2);

    switch 2  % choose 0, then 1, then 2
        case 0
            % Good initial convergence
            alpha0 = -10;
            beta0 = 0;
            omega = 1;
            gradblur = 5;
            delta_t = 1;
            timesteps = 150;
        case 1
            % continuation from above
            alpha0 = -1;
            beta0 = 0;
            omega = 3;
            gradblur = 2;
            delta_t = 0.05;
            timesteps = 100;
        case 2
            % unstable (from above)
            alpha0 = -100;
            beta0 = 0;
            omega = 6;
            gradblur = 1;
            delta_t = 0.01;
            timesteps = 100;
        case 3
            % collapse (elastic force too strong)
            alpha0 = -100;
            beta0 = 0;
            omega = 10;
            gradblur = 3;
            delta_t = 10;
            timesteps = 100;
        case 4
            % parameters for the active contours (snakes)
            alpha0 = -10; % try -10, -100, -200
            beta0 =  0; % try 0 and 10000
            omega = 1; % weight of Fext, try 1, 3
            gradblur = 5; % try 5, 1
            delta_t = 1; % time step, try 1, 10
            timesteps = 100;
    end


    % Compute average segment length
    h = norm([x0(1)-x0(2) y0(1)-y0(2)]);

    % now taking the gradient of the image (need it to calc the ext energy)
    [fx fy] = gradient(MyGaussianBlur(f,gradblur));

    % external energy would then be negative magnitude of the blurred image
    E_ext = sqrt(fx.^2 + fy.^2)*omega*delta_t;
    %E_ext = -(fx.^2 + fy.^2)*omega;

    % now getting the gradient of the external energy
    [dc dr] = gradient(E_ext);

    % now we will go ahead and deal with constructing the A matrix
    n=length(x0);
    alpha = alpha0 * ones(1,n)/h^2;
    beta = beta0 * ones(1,n)/h^4;

    % constructing the pentadiagonal banded matrix
    % first we need to calculate the coefficients
    a = -beta;
    b = -alpha + 4*beta;
    c = 2*alpha - 6*beta;
    d = -alpha + 4*beta;
    e = -beta;

    % now actually constructing the banded matrix
    % down the middle should be v_i coefficients
    A = diag(c);

    % next are v_{i-1} and v_{i+1} coefficients
    A = A + circshift( diag(b), [1 0] );
    A = A + circshift( diag(d), [-1 0] );
    %A = A + diag(b(1:n-1),-1) + diag(b(n), n-1);
    %A = A + diag(d(1:n-1),1) + diag(b(n), -n+1);

    % finally v_{i-2} and v_{i+2} coefficients
    A = A + circshift( diag(a), [2 0] );
    A = A + circshift( diag(e), [-2 0] );
    %A = A + diag(a(1:n-2),-2) + diag(a(n-1:n),n-2);
    %A = A + diag(e(1:n-2),2) + diag(e(n-1:n),-n+2);


    % Pre-compute the inverse of the linear operator
    % [1]
    gammaAinv = inv(diag(ones(1,n)) - delta_t*A);

    % now set the x and y to equal x0 and y0
    x = x0';
    y = y0';

    % iteratively approximate next x and y coordinates
    for j=1:timesteps
        
        % Interpolate the external force at node points
        % (can use interp2)
        % [2]
        %{
        for i=1:n
            xn(i,1) = MyInterp(dr,[x(i)  y(i)], 'linear');
            yn(i,1) = MyInterp(dc,[x(i)  y(i)], 'linear');
        end
        %}
        % Careful that you get the x and y in the right order.
        xn = interp2(dr, y, x);
        yn = interp2(dc, y, x);

        % Take a time-step
        x = gammaAinv*(x + xn);
        y = gammaAinv*(y + yn);

        % we want to interactively show the snake closing in on the brain
        % i decided to display the snake every few iterations
        if(mod(j,5)==0)
            %imshow(f,[]);
            imshow(E_ext, []);
            hold on;
            plotX = x;
            plotX(end+1) = x(1);
            plotY = y;
            plotY(end+1) = y(1);
            plot(plotY,plotX,'-y');
            drawnow;
            %pause(1);
        end
    end
