function processAudio(f)
%PROCESSAUDIO - read in sound sample, encode, plot RMS error
%
[s, Fs] = audioread(f, 'native');       % Get the song in native format
sLen = size(s, 1);
sout = int16(zeros(size(s)));

N = 256;                                % split the s into chunks of N samples
fprintf('Frames: %f\n', sLen/N);
resolutions = 4:12;                     % compress into n bits
rmsError = zeros(size(resolutions, 2), size(s, 2)); % results measurement per channel

for j = 1:size(resolutions, 2)
    nbits = resolutions(j);
    fprintf('%2d bit res block-scaled PCM, framelength %4d\t', nbits, N);
    i = 1;
    gmax = 0; gmin = 0;
    while i <= sLen
        frameEnd = min(i+N-1, sLen);
        frame = s(i:frameEnd, :);
        [scale, g] = frameEncode(frame, nbits);

        decodedFrame = frameDecode(scale, g, nbits);
        %plotLR(decodedFrame, nbits);

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
%{
    fprintf('Press a key to play processed audio');
    pause;
    fprintf('\n');
    player = audioplayer(sout, Fs);
    playblocking(player);
%}
end

figure;
plot(resolutions, mean(rmsError, 2));
title('Mean RMS error vs output bit depth for codec');
ylabel('RMS error (units: 16-bit FSD)');
xlabel('Output bit depth');

disp(rmsError);
end

%--------------------------------------------------------------------------
% Encoder and decoder functions
%--------------------------------------------------------------------------
function [lrMax, g] = frameEncode(f, nbits)
% Encode a frame of audio with output resolution nbits
% returns the maximum scalar value in this frame

f = double(f);                      % promote to higher precision
fsd = 2^(nbits-1)-1;
frameMax = max(abs(f));             % get per-channel maximum amplitudes
lrMax = max(max(frameMax), fsd);
scale = fsd/lrMax; 
g = int16(f.*scale);                % now got reduced bit depth
end

function f = frameDecode(lrMax, g, nbits)
% Decode a frame of audio with
%  lrMax - target maximum scalar amplitude
%  encoded frame g
%  input resolution nbits
fsd = 2^(nbits-1);
scale = lrMax/fsd;
f = g .* scale;
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