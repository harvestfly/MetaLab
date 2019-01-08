from some_libs import *
#import librosa
#import librosa.display

# melspectram未见效，继续尝试
# 1 https://stackoverflow.com/questions/27403459/wavelet-transform-spectrogram-in-python
# 2 https://rdrr.io/cran/spectral/
#
def Sepectrogram(signal):
    if False:
        dt = 0.0005
        t = np.arange(0.0, 20.0, dt)
        s1 = np.sin(2 * np.pi * 1 * t)
        s2 = 2 * np.sin(2 * np.pi * 4 * t)
        # create a transient "chirp"
        mask = np.where(np.logical_and(t > 10, t < 12), 1.0, 0.0)
        # s2 = s2 * mask
        # add some noise into the mix
        nse = 0.01 * np.random.random(size=len(t))

        x = s1 + s2 + nse  # the signal
        NFFT, NOVER = 1024, 900
    else:
        dt = 0.001
        t = np.arange(len(signal))
        x = signal
        NFFT,NOVER = 128,96  # the length of the windowing segments
    Fs = int(1.0 / dt)  # the sampling frequency

    # Pxx is the segments x freqs array of instantaneous power, freqs is
    # the frequency vector, bins are the centers of the time bins in which
    # the power is computed, and im is the matplotlib.image.AxesImage
    # instance

    ax1 = plt.subplot(211)
    plt.title('NFFT={} Fs={}'.format(NFFT, Fs))
    plt.plot(t, x)
    plt.subplot(212)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=NOVER)
    # Pxx, freqs, bins, im = plt.specgram(x)

    plt.show()


def Sepectrogram_1(signal):
    # create a wave with 1Mhz and 0.5Mhz frequencies
    dt = 40e-9
    pi = np.pi;
    t = np.arange(0, 1000e-6, dt)
    fscale = t / max(t)
    y = np.cos(2 * pi * 1e6 * t * fscale) + (np.cos(2 * pi * 2e6 * t * fscale) * np.cos(2 * pi * 2e6 * t * fscale))
    y *= np.hanning(len(y))
    yy = np.concatenate((y, ([0] * 10 * len(y))))

    # FFT of this
    Fs = 1 / dt  # sampling rate, Fs = 500MHz = 1/2ns
    n = int(len(yy))  # length of the signal
    k = np.arange(n)
    k_half = range((int)(n / 2))
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[k_half]  # one side frequency range
    Y = np.fft(yy) / n  # fft computing and normalization
    Y = Y[k_half] / max(Y[k_half])

    # plotting the data
    plt.subplot(3, 1, 1)
    plt.plot(t * 1e3, y, 'r')
    plt.xlabel('Time (micro seconds)')
    plt.ylabel('Amplitude')
    plt.grid()

    # plotting the spectrum
    plt.subplot(3, 1, 2)
    plt.plot(frq[0:600], abs(Y[0:600]), 'k')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.grid()

    # plotting the specgram
    plt.subplot(3, 1, 3)
    Pxx, freqs, bins, im = plt.specgram(y, NFFT=512, Fs=Fs, noverlap=10)
    plt.show()


def make_melgram(mono_sig, sr, n_mels=128):   # @keunwoochoi upgraded form 96 to 128 mel bins in kapre
    if True:
        sr =256
        signal, sr = librosa.load('F:/baby cry detection/free_sound_org/cry/59578__morgantj__babycrying.mp3')
        x_tic=range(len(signal))
        #sin_x = [np.sin(xi/10000) for xi in x_tic]
        #signal = np.array( [a+a*a for a in sin_x] )
        x = librosa.feature.melspectrogram(signal,sr=sr, n_mels=n_mels)
        log_S = librosa.power_to_db(x, ref=np.max)
        melgram = librosa.amplitude_to_db(x)
        plt.figure(figsize=(16, 8))
        # Display the spectrogram on a mel scale    sample rate and hop length parameters are used to render the time axis
        plt.subplot(3, 1, 1)
        #librosa.display.specshow(x, sr=sr, x_axis='time', y_axis='mel')
        plt.plot(range(len(signal)),signal)
        plt.subplot(3, 1, 2)
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.subplot(3, 1, 3)
        librosa.display.specshow(melgram, sr=sr, x_axis='time', y_axis='mel')
        plt.title('sr={} n_mels={} signal={} X={} log_S={} melgram={}'.format(sr,n_mels,signal.shape,x.shape,log_S.shape,melgram.shape))
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()

    hop_length = 512
    nT =  (int)(len(mono_sig)*1.0/hop_length+1),
    melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(mono_sig,
        sr=sr, n_mels=n_mels))[np.newaxis,:,:,np.newaxis]     # last newaxis is b/c tensorflow wants 'channels_last' order
    assert melgram.shape[2] == nT and melgram.shape[1] == n_mels
    assert melgram.shape[3] == 1

    '''
    # librosa docs also include a perceptual CQT example:
    CQT = librosa.cqt(mono_sig, sr=sr, fmin=librosa.note_to_hz('A1'))
    freqs = librosa.cqt_frequencies(CQT.shape[0], fmin=librosa.note_to_hz('A1'))
    perceptual_CQT = librosa.perceptual_weighting(CQT**2, freqs, ref=np.max)
    melgram = perceptual_CQT[np.newaxis,np.newaxis,:,:]
    '''
    return melgram