% Function g = Gaussian(sigma, dims)
%
%   Create a matrix containing a 2D or 3D Gaussian function,
%
%      (sqrt(2*pi)*sigma)^(-D) * exp( -x^2/2/sigma^2 )
%
%   where D = length(dims).
%
%  Input:
%    sigma is the standard deviation of the Gaussian, in
%          units of pixels.
%    dims is a vector with the desired array size (2D or 3D)
%
%  Output:
%    g is an array of size dims with a Gaussian function
%      centred at pixel
%         ctr = ceil( (dims-1) / 2 ) + 1
%
function g = Gaussian(sigma, dims)

	rows = dims(1);
	cols = dims(2);
    slices = 1;
    D = 2;
    if length(dims)>2
        slices = dims(3);
        D = 3;
    end
    
	% locate centre pixel.
    % For 256x256, centre is at (129,129)
    % For 257x257, centre is still at (129,129)
	cr = ceil( (rows-1)/2 ) + 1;
	cc = ceil( (cols-1)/2 ) + 1;
    cs = ceil( (slices-1)/2) + 1;
    
	% Set the parameter in exponent 
    a = 1 / (2*sigma^2);
	g = zeros(rows,cols,slices);

    for s = 1:slices
        for c = 1:cols
            for r = 1:rows
                r_sh = r - cr;
                c_sh = c - cc;
                s_sh = s - cs;
                g(r,c,s) = exp( -a * (r_sh^2 + c_sh^2 + s_sh^2) );
            end
        end
    end
    
    g = g / (sqrt(2*pi)*sigma)^D;
    
