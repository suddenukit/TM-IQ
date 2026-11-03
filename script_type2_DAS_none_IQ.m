
%-- Implementation of conventional Delay-and-Sum (DAS) with apodization in reception
%-- The code works both for IQ and IQ format

%-- Authors: Alfonso Rodriguez-Molares (alfonso.r.molares@ntnu.no)
%--          Olivier Bernard (olivier.bernard@creatis.insa-lyon.fr)

%-- $Last Update: 2017/06/28 $

%-- Modified by D.N.Rakhmatov, 25 July 2023


clear all;
close all;
clc;


%-- setup PICMUS path to IQ/scan data
picmus_path = '.\PICMUS2';
addpath(genpath(picmus_path));

%-- generate dataset/scan/image file paths
dataset_path = [picmus_path,'\data\dataset_iq_in_vitro_type2_transmission_2_nbPW_75.hdf5'];
scan_path = [picmus_path,'\data\scanning_region_picmus.hdf5'];

%-- load scan and dataset
scan = linear_scan();
scan.read_file(scan_path);
dataset = us_dataset();
dataset.read_file(dataset_path);

%-- check if data is IQ
if(isempty(dataset.modulation_frequency)||dataset.modulation_frequency==0)
    dataset.data=reshape(hilbert(dataset.data(:,:)),size(dataset.data));
end


%% Define scan structure

%-- configure input settings

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


%% Perform DAS beamforming

%-- receive apodization:
%-- dynamically expanding receive aperture with selected apodization
winID = 'tukey25';
rx_f_number = 1.75;
rx_aperture = scan.z/rx_f_number;
rx_aperture_distance = abs(scan.x*ones(1,dataset.channels)-ones(scan.pixels,1)*dataset.probe_geometry(:,1).');
receive_apodization = tools.apodization(rx_aperture_distance,rx_aperture*ones(1,dataset.channels),winID);
receive_apodization = bsxfun(@rdivide,receive_apodization,sum(receive_apodization,2));

%-- angular apodization -> no apodization
angular_apodization = ones(scan.pixels,dataset.firings);
angular_apodization  = bsxfun(@rdivide,angular_apodization,sum(angular_apodization,2));


%-- select frames to beamform
PWsel = 38;                             %-- single PW (zero angle)
%PWsel = (1:75);                         %-- all 75 PWs
%PWsel = round(linspace(1,75,15));       %-- 15 PWs

Nf = length(PWsel);


%-- beamforming loop
beamformed_data = zeros(scan.pixels,1);
time_vector = dataset.initial_time+(0:(dataset.samples-1))/dataset.sampling_frequency;
w0 = 2*pi*dataset.modulation_frequency;

%-- start timer here
tic

for pw=PWsel

    %-- transmit delay
    transmit_delay = scan.z*cos(dataset.angles(pw))+scan.x*sin(dataset.angles(pw));
    for nrx=1:dataset.channels
        %-- receive delay
        receive_delay = sqrt((dataset.probe_geometry(nrx,1)-scan.x).^2+(dataset.probe_geometry(nrx,3)-scan.z).^2);
        %-- total delay
        delay = (transmit_delay+receive_delay)/dataset.c0;
        %-- phase shift
        phase_shift = exp(1i.*w0*(delay-2*scan.z/dataset.c0));
        %-- beamformed data
        beamformed_data = beamformed_data + ...
            phase_shift.*angular_apodization(:,pw).*receive_apodization(:,nrx)...
            .*interp1(time_vector,dataset.data(:,nrx,pw),delay,'spline',0);
    end

    fprintf(' %d ', pw);
    
end

%-- get rid of NaN (if any) and reshape
beamformed_data(isnan(beamformed_data)) = 0;
dasIQdata = reshape(beamformed_data,[numel(scan.z_axis) numel(scan.x_axis)]);


%-- stop timer here
T_das = toc;


%-- compute max-normalized envelope
envelope_dasIQdata = abs(dasIQdata);
envelope_dasIQdata = envelope_dasIQdata/max(envelope_dasIQdata(:));


fprintf('\n\n');


%% Store image and evaluate quality metrics

%-- declare an us_image object to store the beamformed data
imageDAS = us_image('DAS beamforming');
imageDAS.author = 'Alfonso Rodriguez-Molares <alfonso.r.molares@ntnu.no>';
imageDAS.affiliation = 'Norwegian University of Science and Technology (NTNU)';
imageDAS.algorithm = 'Delay-and-Sum (IQ version)';
imageDAS.scan = scan;
imageDAS.number_plane_waves = Nf;
imageDAS.data = envelope_dasIQdata;
imageDAS.transmit_f_number = 0;
imageDAS.receive_f_number = rx_f_number;
imageDAS.transmit_apodization_window = 'none';
imageDAS.receive_apodization_window = 'Tukey 25%';

%-- setup evaluation structure
info = tools.generate_data_info_structure('in_vitro_type2');
metrics = us_picmus_metrics();
metrics.image = imageDAS;
metrics.scan = imageDAS.scan;
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
disp('mean FWHM (1-axial, 2-lateral)')
disp(mean(metrics.scoreFWHM))
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
fprintf('DAS time = %f [sec]\n\n',T_das);

logFILE = fopen(sprintf('metrics_type2_DAS%s_none_PW%d.txt',winID,Nf),'w');
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

fprintf(logFILE,'\nDAS time  = %f [sec]\n',T_das);

fclose(logFILE);


%% Show reconstructed image

%-- compute log-compressed envelope (dB)
env = imageDAS.data;
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
title(sprintf('Type-2 DAS, %d PW', Nf),'FontName','Courier','FontSize',24,'FontWeight','bold');
%title(sprintf('%d-PW DAS',Nf),'FontName','Courier','FontSize',16,'FontWeight','bold');
saveas(gcf,sprintf('type2_DAS%s_none_PW_IQ%d',winID,Nf),'epsc');
saveas(gcf,sprintf('type2_DAS%s_none_PW_IQ%d.fig',winID,Nf));

