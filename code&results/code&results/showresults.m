close;close;
addpath('utils');

load('scene1.mat');
load('res_scene1.mat');

gtspec = squeeze(gt_light);
predictedspec = squeeze(predict_light);

predictedspec = sum(gtspec.*predictedspec,1)./sum(predictedspec.*predictedspec,1).*predictedspec;


figure;
imshow(HSI2RGB(permute(squeeze(image),[3 2 1])));
hold on;
[posx,posy] = meshgrid([40:80:200,40:80:200]);
plot(posx,posy,'r+','linewidth',3);
title('Multispectral image');

figure;
set(gcf,'position',[0 0 900 600]);
for ii = 1:3
    for jj = 1:3
        subplot(3,3,jj+(ii-1)*3);
        plot(400:10:700,squeeze(predictedspec(:,-40+ii*80,-40+ii*80)),'r-','LineWidth',1.5);
        hold on; plot(400:10:700, squeeze(gtspec(:,-40+ii*80,-40+ii*80)),'k--','LineWidth',2);
        ylim([0 1]);
        xtick= 400:50:700;
        set(gca,'xtick',xtick);
        legend({'Predicted','GT'},'Location','northwest','FontSize',6)
        grid on;
    end
end
suptitle('Estimated illumination spectra at nine different spatial locations')

