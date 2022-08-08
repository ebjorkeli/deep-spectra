import numpy as np
import matplotlib.pyplot as plt

def fft_spectra(timespectra, nth=0, plot=True):
    # Number of samplepoints
    N = timespectra.shape[1]
    # sample spacing
    T = 1.0 / 1024.0 * 0.5
    y = timespectra[nth,:,0] + 1j * timespectra[nth,:,1]
    x = np.arange(N)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    axis = 0
    ss = [slice(None) for i in range(x.ndim)]
    ss[axis] = slice(0, 1)
    ss = tuple(ss)
    y[ss] *= 0.5

    yf = np.fft.fftshift(np.fft.fft(y, axis=axis, norm='ortho'), axes=axis)

    if plot:
        # Lag noe for ppm
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(x[0:], 2.0/N * (yf)[0:])
        plt.show()

    return yf

def ifft_spectra(freqspectra, nth=0):
    N = freqspectra.shape[1]
    x = np.arange(N)
    axis = 0
    yf = np.fft.ifft(np.fft.ifftshift(y, axis=axis, norm='ortho'), axes=axis)
    ss = [slice(None) for i in range(x.ndim)]
    ss[axis] = slice(0, 1)
    ss = tuple(ss)
    y[ss] *= 2
    return yf

def hz2ppm(cf, hz, shift=True, shift_amount=4.65):
    """Convert frequency scale to frequency scale with optional shift."""
    if shift:
        return 1e6 * hz / cf + shift_amount
    else:
        return 1e6 * hz / cf