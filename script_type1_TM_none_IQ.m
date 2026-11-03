%--------------------------------------------------------------------------
%  PICMUS in-vitro IQ dataset: "type1" phantom, dichotomous scheme, 75 PWs
%  (online at https://www.creatis.insa-lyon.fr/EvaluationPlatform/picmus/)
%  B-mode image reconstruction using Temme-Mueller migration
%  Created by D.N.Rakhmatov (daler@ece.uvic.ca), 2023/08/15
%--------------------------------------------------------------------------

clear all;
close all;
clc;


%-- setup PICMUS path to IQ/scan data
picmus_path = '.\PICMUS2';
addpath(genpath(picmus_path));


%-- generate dataset/scan/image file paths
dataset_path = [picmus_path,'\data\dataset_iq_in_vitro_type1_transmission_2_nbPW_75.hdf5'];
scan_path = [picmus_path,'\data\scanning_region_picmus.hdf5'];

%-- load scan and dataset
scan = linear_scan();
scan.read_file(scan_path);
dataset = us_dataset();
dataset.read_file(dataset_path);


PWfreq = 5.208e6;	%-- PICMUS: Verasonics Vantage-256, L11-4v probe
PWcycles = 2.5;     %-- PICMUS: number of cycles (transmit pulse)
PWbw = 0.67; 

fs = dataset.sampling_frequency;


%% Select IQ data frames for image reconstruction 

%PWsel = [38];                           %-- single PW (zero angle)
PWsel = (1:75);                         %-- all 75 PWs
%PWsel = round(linspace(1,75,15));       %-- 15 PWs

IQdata = dataset.data(:,:,PWsel);
[Nt,Nx,Nf] = size(IQdata);
Nz = Nt;


%% Configure reconstruction settings

theta = dataset.angles(PWsel);          %-- selected angles (rad)
c = dataset.c0;                         %-- sound speed (m/s)
v = c/2;                                %-- ERM velocity (m/s)

%-- determine original x-axis, t-axis, and z-axis
x_axis = dataset.probe_geometry(:,1);
t_axis = (0:size(dataset.data,1)-1).'/dataset.sampling_frequency + dataset.initial_time;
z_axis = v*t_axis;

dx = mean(diff(x_axis));        %-- original lateral spacing (m)
dt = mean(diff(t_axis));        %-- original time step (s)
dz = mean(diff(z_axis));        %-- original depth spacing (m)

probeSpan = x_axis(end) - x_axis(1);


%-- override default scan (other scan parameters updated automatically)

dxp = 1e-4;                     %-- lateral pixel size [m]
dzp = 1e-4;                     %-- axial pixel size [m]
zMin = 0;                       %-- min. depth [m]
zMax = 55*1e-3;                 %-- max. depth [m]

scan.x_axis = ((x_axis(1)-dxp):dxp:(x_axis(end)+dxp)).';
scan.z_axis = single(zMin:dzp:zMax).';

Nxp = numel(scan.x_axis);
Nzp = numel(scan.z_axis);


[Z,X] = ndgrid(z_axis,x_axis);
[iZ,iX] = ndgrid(scan.z_axis,scan.x_axis);


%-- configure 2D FFT grid: NxFFT and NtFFT must be EVEN
NtFFT = pow2(nextpow2(Nt)+1);
NxFFT = pow2(nextpow2(Nx)+1);

Kx0 = (-NxFFT/2:NxFFT/2-1)/dx/NxFFT;	%-- spatial frequencies
F0 = (-NtFFT/2:NtFFT/2-1)/dt/NtFFT;        	%-- temporal frequencies

[Kx,F] = meshgrid(Kx0,F0);              %-- Cartesian spectral grid
F = F + dataset.modulation_frequency;                        % physical Hz
Kz = F/v;                               %-- Kz corresponds to z = t*c/2


%-- configure image filter
lambda = c/PWfreq;
imFiltSize = double([max(1,round(lambda/dz)) max(1,round(lambda/dx))]);
% NOTE: dz/dx = axial/lateral step size during migration


%% Perform Temme-Mueller migration

D_Kz_Kx_comp = zeros(NtFFT,NxFFT);

%-- start timer here
tic

for k = 1:Nf

    sinTheta = sin(theta(k));
    cosTheta = cos(theta(k));

    D_t_x = IQdata(:,:,k);              %-- acquire raw IQ data

    % Perform TM migration

    denom = 2*(Kx*sinTheta + Kz*cosTheta);
    Knew = (Kx.^2 + Kz.^2)./denom;
    Knew(isnan(Knew)) = 0;
    Fnew = c*Knew;
    
    scaler = 2*c*abs(cosTheta*(Kx.^2 - Kz.^2) - 2*sinTheta*Kx.*Kz)./denom.^2;
    scaler(isnan(scaler)) = 0;
    for i = 1:NtFFT
        % negative Kx
        for j = 1:NxFFT/2        
            if Kz(i,j) <= -Kx(i,j)*cosTheta/(1-sinTheta)
                scaler(i,j) = 0;
            end
        end
        % positive Kx
        for j = NxFFT/2+1:NxFFT       
            if Kz(i,j) <= Kx(i,j)*cosTheta/(1+sinTheta)
                scaler(i,j) = 0;
            end
        end
    end
    
    
    %-- create F-x filter: PW angle compensation
    Filter_F_x = zeros(NtFFT,Nx);
    for j = 1:Nx
        tshift = x_axis(j)*sinTheta/c;
        Filter_F_x(:,j) = exp(1i*2*pi*tshift*F(:,1));
    end
    
    %-- create F-Kx filter: evanescent region rejection
    Filter_F_Kx = (F.^2 > (c*Kx - sinTheta*F).^2);
    
    %-- create Kz-Kx filter: magnitude scaling
    Filter_Kz_Kx = scaler;
    
    
    %-- start processing input data

    D_F_x = fftshift(fft(D_t_x,NtFFT,1),1);

    %-- apply F-x filter
    D_F_x = D_F_x.*Filter_F_x;

    D_F_Kx = fft(D_F_x,NxFFT,2);
    D_F_Kx = fftshift(D_F_Kx,2);        %-- shift zero-Kx to center

    %-- apply F-Kx filter
    D_F_Kx = D_F_Kx.*Filter_F_Kx;
    
    %-- map from F-Kx to Kz-Kx domain
    D_Kz_Kx = zeros(NtFFT,NxFFT);
    for j = 1:NxFFT
        D_Kz_Kx(:,j) = interp1(F(:,j),D_F_Kx(:,j),Fnew(:,j),'spline',0);
    end

    %-- apply Kz-Kx filter
    D_Kz_Kx = D_Kz_Kx.*Filter_Kz_Kx;


    %-- compound in Kz-Kx domain
    D_Kz_Kx_comp = D_Kz_Kx_comp + D_Kz_Kx;


    fprintf(' %d ', PWsel(k));
    
end


%-- transform compounded frame into z-x domain (original x-axis)
D_Kz_Kx = ifftshift(D_Kz_Kx_comp,2);

D_Kz_x = ifft(D_Kz_Kx,NxFFT,2);
D_Kz_x = D_Kz_x(:,1:Nx);

D_Kz_x = ifftshift(D_Kz_x, 1);             
IQdata_full = ifft(D_Kz_x, NtFFT, 1);       % complex baseband IQ

IQdataOut = IQdata_full(1:Nz, :);


%-- stop timer here
T_TM = toc;


%% Perform Gridded Interpolation 

%-- start timer here
tic

%-- apply gridded interpolant
gI = griddedInterpolant(Z,X,IQdataOut,'spline');
IQdata_gi = gI(iZ,iX);

%-- compute normalized envelope
env = abs(IQdata_gi);
env = env./max(env(:));

%-- stop timer here
T_gi = toc;


fprintf('\n\n');


%% Store image and evaluate quality metrics

imageTM = us_image('Temme-Mueller Migration');
imageTM.author = 'D.Rakhmatov <daler@ece.uvic.ca>';
imageTM.affiliation = 'University of Victoria, Canada';
imageTM.scan = scan;
imageTM.number_plane_waves = Nf;
imageTM.data = env;
imageTM.transmit_f_number = 0;
imageTM.receive_f_number = 0;
imageTM.transmit_apodization_window = 'none';
imageTM.receive_apodization_window = 'none';

%-- setup evaluation structure
info = tools.generate_data_info_structure('in_vitro_type1');
metrics = us_picmus_metrics();
metrics.image = imageTM;
metrics.scan = imageTM.scan;
metrics.set_data_information(info);
metrics.flagDisplay = 1;
metrics.evaluate();

%-- display and save evaluation scores
disp('--------------------------------')
disp('Speckle quality')
disp(metrics.scoreSpeckleQuality)
disp('--------------------------------')
disp('Geometric distortion')
disp(metrics.scoreGeometricalDistortion)
disp('--------------------------------')
disp('Intensity linearity')
disp(metrics.scoreLinearIntensity)
disp('--------------------------------')
disp('FWHM (1-axial, 2-lateral)')
disp(metrics.scoreFWHM)
disp('--------------------------------')
disp('Axial resolution')
disp(metrics.scoreResolutionAxial)
disp('--------------------------------')
disp('Lateral resolution')
disp(metrics.scoreResolutionLateral)
disp('--------------------------------')
disp('Contrast')
disp(metrics.scoreContrast)
disp('--------------------------------')
disp('gCNR')
disp(metrics.scoreGCNR)
disp('--------------------------------')
fprintf('\nDone.\n\n');
fprintf('TM time  = %f [sec]\n\n',T_TM);
fprintf('GI time = %f [sec]\n\n',T_gi);

logFILE = fopen(sprintf('metrics_type1_TM_none_PW%d.txt',Nf),'w');
fprintf(logFILE,'\nSelected IQ data frames:\n\n%s\n\n',num2str(PWsel));
fprintf(logFILE,'--------------------------------\n');
fprintf(logFILE,'Speckle quality\n%s',evalc('disp(metrics.scoreSpeckleQuality)'));
fprintf(logFILE,'--------------------------------\n');
fprintf(logFILE,'Geometric distortion\n%s',evalc('disp(metrics.scoreGeometricalDistortion)'));
fprintf(logFILE,'--------------------------------\n');
fprintf(logFILE,'Intensity linearity (pass/fail, error)\n%s',evalc('disp(metrics.scoreLinearIntensity)'));
fprintf(logFILE,'--------------------------------\n');
fprintf(logFILE,'FWHM (1-axial, 2-lateral)\n%s',evalc('disp(metrics.scoreFWHM)'));
fprintf(logFILE,'Mean FWHM (axial/lateral) = %.4f/%.4f\n',...
    mean(metrics.scoreFWHM(:,:,1)),mean(metrics.scoreFWHM(:,:,2)));
fprintf(logFILE,'--------------------------------\n');
fprintf(logFILE,'Axial resolution\n%s',evalc('disp(metrics.scoreResolutionAxial)'));
fprintf(logFILE,'--------------------------------\n');
fprintf(logFILE,'Lateral resolution\n%s',evalc('disp(metrics.scoreResolutionLateral)'));
fprintf(logFILE,'--------------------------------\n');
fprintf(logFILE,'Contrast\n%s',evalc('disp(metrics.scoreContrast)'));
fprintf(logFILE,'--------------------------------\n');

fprintf(logFILE,'\nTM time  = %f [sec]\n',T_TM);
fprintf(logFILE,'\nGI time = %f [sec]\n',T_gi);

fclose(logFILE);


%% Show reconstructed image

%-- compute log-compressed envelope (dB)
env = imageTM.data;
im = 20*log10(env./max(env(:)));

%-- display image (60-dB range)
vrange = [-60 0];
x_lim = [min(scan.x_axis) max(scan.x_axis)]*1e3;
z_lim = [min(scan.z_axis) max(scan.z_axis)]*1e3;
figure;
imagesc((scan.x_axis)*1e3,(scan.z_axis)*1e3,im);
shading flat; colormap gray;
set(gca,'FontName','Courier','FontSize',14,'FontWeight','demi');
axis([x_lim z_lim]); axis image; caxis(vrange); %colorbar;
xlabel('x (mm)','FontName','Courier','FontSize',16,'FontWeight','bold');
ylabel('z (mm)','FontName','Courier','FontSize',16,'FontWeight','bold');
set(gca,'YDir','reverse'); set(gca,'XTick',[-16 -8 0 8 16]);
title(sprintf('Type-1 TM, %d PW', Nf),'FontName','Courier','FontSize',24,'FontWeight','bold');
saveas(gcf,sprintf('type1_TM_none_PW_IQ%d',Nf),'epsc');
saveas(gcf,sprintf('type1_TM_none_PW_IQ%d.fig',Nf));


% %% Display Patch Images
% 
% x_ind = (133:333);      % between approximately -6 and +14 mm
% z_ind = (321:501);      % between approximately 32 and 50 mm
% 
% imagesc((scan.x_axis(x_ind))*1e3,(scan.z_axis(z_ind))*1e3,im(z_ind,x_ind));
% shading flat; colormap gray; caxis(vrange);
% set(gca,'YDir','reverse'); axis off;
% saveas(gcf,sprintf('type1_patch_TM_none_PW%d',Nf),'epsc');
% saveas(gcf,sprintf('type1_patch_TM_none_PW%d.fig',Nf));
