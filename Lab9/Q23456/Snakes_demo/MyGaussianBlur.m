% Function G = MyGaussianBlur(F, fwhm)
%
%  Blur a dataset of images using a Gaussian filter.
%
%  F holds a stack of images.
%  fwhm is the full-width at half max of the Gaussian kernel.
%
%%%% [5] marks total
function B = MyGaussianBlur(F, fwhm)

	dims = size(F);
    
    if length(dims)==2
        dims = [dims 1];
    end

	B = zeros(size(F));

    % [1] for calling the Gaussian function properly
	k = Gaussian(fwhm, size(F));
    K = fftn(ifftshift(k)); % [1] for FFT of Gaussian
    % Note that the peak of the Gaussian must be at pixel (1,1,1)
    
    % [2] for element-wise mult. of FFT of image (volume) by K
    temp = fftn( F );
    B = real(ifftn(temp.*K));
    % [1] for taking 'real' of result
    % NOTE: taking 'abs' of result is not the same
    %       if the input has negative values.
    %       Do not grant mark for use of 'abs'.

    
    
    
    