function RGB = HSI2RGB(image)
% spectral data visualization especially for MIID dataset
%% Get the color matching functions. 
% ref1: http://cvrl.ioo.ucl.ac.uk/cmfs.htm
% ref2: https://ww2.mathworks.cn/matlabcentral/fileexchange/7021-spectral-and-xyz-color-functions?focused=5172027&tab=function

% if withillum is true, image must be reflectance image, and RGB is the
% rendered images with standard illumination


	[row,col,band] = size(image);
	[lambda, xFcn, yFcn, zFcn] = colorMatchFcn('1964_full');

%% Process color matching functions.
    
    wavelength = [400:10:780];
    wavelength = wavelength(1:band);
    
	xFcn = interp1(lambda,xFcn,wavelength,'cubic')';
    yFcn = interp1(lambda,yFcn,wavelength,'cubic')';
	zFcn = interp1(lambda,zFcn,wavelength,'cubic')';  
    
    %xFcn = xFcn/sum(xFcn);
    %yFcn = yFcn/sum(yFcn);
    %zFcn = zFcn/sum(zFcn);
    
%% Convert the hyperspectral image to XYZ image
    withillum = false;
    A = reshape(image,row*col,band);
    if withillum
        [lambda_light, energy] = illuminant('d65');
        illumination = interp1(lambda_light,energy,wavelength,'cubic'); 
        A = A.*repmat(illumination,row*col,1);
    end

    
    XYZ = A*[xFcn,yFcn,zFcn];
    
	XYZ = max(XYZ,0);
    XYZ = XYZ/max(max(XYZ(:)),1);

    RGB = xyz2rgb(reshape(XYZ,row,col,3),'WhitePoint','d65');
   RGB = max(RGB,0);
   RGB = RGB/max(RGB(:));
    %figure;imshow(RGB);

end

