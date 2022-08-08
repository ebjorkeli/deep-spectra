import numpy as np

def phase(freq_domain, index=550, phi_0=0, phi_1=True):
    new_freq = []
    for i, freq in enumerate(freq_domain):
        re, im = freq[..., 0], freq[..., 1]
        phi = phi_0
        if phi_1:
            d = 2
            regly  = np.sum(re[index-d:index+d])
            imgly  = np.sum(im[index-d:index+d])
            phi_1  = np.arctan(imgly/regly)
            phi += phi_1
        phased = re*np.cos(phi) + im*np.sin(phi)
        if np.any(np.isnan(phased)):
            new_freq.append(re)
        else:
            new_freq.append(phased)
    return np.asarray(new_freq)