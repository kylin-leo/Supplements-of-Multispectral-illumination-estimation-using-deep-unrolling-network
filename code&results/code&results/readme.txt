1)This folder contains three multiple illumination scenes.

2)To estimate the illumination of scenex, please run "python illumspec_estimation.py -- imgname scenex", and then a file named "res_scenex.mat" containing the estimated illumination spectra will be saved.

3)To compare the ground truth and the estimated spectral result of the scene, please run showresults.m in MATLAB.


other files:

environment.yaml - the running environment 
model/net_params.pkl  -  the network model
utils/colorMatchFcn - color matching function
utils/HSI2RGB - render spectral images to RGB images