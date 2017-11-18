function processAudio(f)
%PROCESSAUDIO - read in sound sample, encode, plot RMS error
%
[s, Fs] = audioread(f, 'native');           % Get the song in native format
s = double(s);                              % convert to float
sLen = size(s, 1);
sout = zeros(size(s));
ehat = zeros(size(s));

close all;                                  % remove existing plots

nbits = 4;                                  % choose optimised values
for alpha = [0.96]
    for deltamin = [8]
        for deltamax = [1400]
            % print out all initial values
            stitle1=sprintf('nbits:%d, deltamin/max: %d %d, alpha:%0.4f - ',nbits,deltamin,deltamax,alpha); fprintf(stitle1);
            [sout, ehat, snr]=encode_adpcm(s, Fs, nbits,alpha,deltamin,deltamax);
            stitle2 = sprintf('Encoded snr (dB): %8.2f\n',snr); fprintf(stitle2);

            plotSignals(s, ehat, Fs, strcat(stitle1, stitle2));
        end
    end
end

fprintf('Press a key to play processed audio');
pause;
fprintf('\n');
player = audioplayer(int16(sout), Fs);
playblocking(player);

%{
N = 256;                                % split the s into chunks of N samples
fprintf('Frames: %f\n', sLen/N);
resolutions = 4:16;                     % compress into n bits
rmsError = zeros(size(resolutions, 2), size(s, 2)); % results measurement per channel

for j = 1:size(resolutions, 2)
    nbits = resolutions(j);
    fprintf('%2d bit res block-based DPCM, framelength %4d\t', nbits, N);
    i = 1;
    gmax = 0; gmin = 0;
    while i <= sLen
        frameEnd = min(i+N-1, sLen);
        frame = s(i:frameEnd, :);
        [g, scale, g1] = frameEncode(frame, nbits);

        decodedFrame = frameDecode(g, scale, g1, nbits);
        %plotLR(decodedFrame, 16);

        sout(i:frameEnd, :) = decodedFrame;

        % monitor the full-scale values we're getting out of the encoder
        gmax = max(gmax, max(g(:)));
        gmin = min(gmin, min(g(:)));
        i = i + N;
    end
    fprintf('Sample range: %5d .. %5d\n', gmax, gmin);

    % Calculate and plot error signal down in the small bits
    err = sout-s;
    %plotLR(err, 12); 
    rmsError(j,:) = sqrt(mean(double(err).^2));
    fprintf('RMS error [Left, Right] = %f %f\n', rmsError(j, 1), rmsError(j, 2));


end
figure;
plot(resolutions, mean(rmsError, 2));
title('Mean RMS error vs output bit depth for codec');
ylabel('RMS error (units: 16-bit FSD)');
xlabel('Output bit depth');

disp(rmsError);
%}

end

function plotSignals(s, ehat, Fs, myTitle)
% plot long-time average power density spectrum of signal and error signals
figure;
Nwin=512;
Nfft=1024;
[P2,F,R,T]=pd_spect(s,Fs,Nfft,Nwin);
plot(F,10*log10(P2),'r','LineWidth',2),grid on;
xlabel('Frequency'),ylabel('Log Magnitude');hold on;
title(myTitle);

[P2,F,R,T]=pd_spect(ehat,Fs,Nfft,Nwin);
plot(F,10*log10(P2),'b','LineWidth',2);
legend('input signal power spectrum', 'error signal power spectrum');
hold off;
end

function plotLR(d, nbits)
% Plot an audio sample as left/right channels, with Y scaling suitable for
% n bits of resolution
fsd = 2^(nbits-1);

ax1 = subplot(2,1,1);       % top subplot
plot(ax1, d(:,1), 'g')
axis([1 size(d, 1) -fsd fsd-1])
title(ax1,'Left channel')

ax2 = subplot(2,1,2);       % middle
plot(ax2, d(:,2), 'r')
axis([1 size(d, 1) -fsd fsd-1])
title(ax2,'Right channel')

drawnow;
end

%--------------------------------------------------------------------------
% Encoder and decoder functions
%--------------------------------------------------------------------------

function [xhats, ehat, snr]=encode_adpcm(xin,fs,nbits,alpha,deltamin,deltamax)
% ADPCM coder with adaptive step sizes based on Jayant algorithm
% Credits: https://uk.mathworks.com/matlabcentral/fileexchange/45310-adpcm

% Inputs:
%   xin: speech array
%   fs: sampling rate of speech
%   nbits: number of quantizer bits
%   alpha: 1-tap predictor value
%   deltamin: minimum value of delta
%   deltamax: maximum value of delta

% Outputs:
%   xhats: quantized signal
%   ehat: error signal
%   snr: signal-to-noise ratio for most recent adpcm run

% calculate step size multipliers based on value of nbits (negative values
% of nbits used for more aggressive attack for 4 and 5 bit quantizers)
if (nbits == 2)
    P=[0.8 1.6];
elseif (nbits == 3)
    P=[0.9 0.9 1.25 1.75];
elseif (nbits == 4)
    P=[.9 .9 .9 .9 1.2 1.6 2.0 2.4];
elseif (nbits == -4)
    nbits=4;
    P=[.9 .9 .9 .9 1.2 1.6 4.0 9.6];
elseif (nbits == 5)
    P=[0.9 0.9 0.9 0.9 0.95 0.95 0.95 0.95 1.2 1.5 1.8 2.1 2.4...
        2.7 3.0 3.3];
elseif (nbits == -5)
    nbits=5;
    P=[0.9 0.9 0.9 0.9 0.95 0.95 0.95 0.95 2.4 3.0 3.6 4.2 4.8...
        5.4 6.0 6.6];
else
    fprintf('nbits must be in range of 2-5: \n');
    pause
end

% determine starting and ending samples of speech file
ss=1; es=length(xin);
nsamp=es;

% calculate mean, xbar, and standard deviation, sigmax, of input speech
x=xin(ss:es)';
xbar=sum(x)/length(x);
sigmax=sqrt(sum(x.^2)/length(x)-xbar^2);

% loop initial conditions for adpcm coder
%   xhat is the quantized value of x; xhats is the saved values of x
%   xtilde is the predicted value of x; xtildes is the saved values of
%   xtilde
%   d is the difference signal; ds is the saved values of d
%   dhat is the quantized value of d; dhats is the saved values of dhat
%   c is the codeword; cs is the saved values of c
%   csign is the sign of the codeword; csigns is the saved values of csign
%   P is the array of codeword weights
xhats(1)=0;
xtildes=1;
ds=-1;
dhats=-1;
cs(1)=0;
csigns(1)=1;
pms=P(1);
deltao=deltamin;

% loop adpcm encoder over all samples 
fid=fopen('adpcm_encode.txt','wt');
fprintf(fid,'     n      x xtilde      d   dhat   xhat      c  delta \n');  %TODO
for n=2:length(x)
    xtilde=alpha*xhats(n-1);
    d=x(n)-xtilde;
    [dhat,csign,codeword,delta]=quant1(d,deltao,nbits,P,deltamin,deltamax);
    xhat=xtilde+dhat;
    xhats(n)=xhat;

    deltao=delta;
    cs(n)=codeword;
    csigns(n)=csign;

    if (n <= 10)
        fprintf(fid,'%6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f \n',...
            n,x(n),xtilde,d,dhat,xhat,codeword,delta);
    end
end
fclose(fid);

% error signal
ehat=xhats-x;
% determine error mean and sigma and print results
ebar=sum(x-xhats)/length(x);
sigmae=sqrt(sum((x-xhats).^2)/length(x)-ebar^2);
    
% determine snr for differential quantizer
snr=20*log10(sigmax/sigmae);    
end