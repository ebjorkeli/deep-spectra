
def nifti(data):
    n_frames = data.shape[0] * data.shape[-1]

    header = {
        'type_of_sig': hdr[0],
        'number_of_points': int(data.shape[1]),
        'sampling_interval': hdr[2],
        'begin_time ': hdr[3],
        'zero_order_phs': hdr[4],
        'transmitter_frequency': hdr[5],
        'magnetic_field': hdr[6],
        'type_of_nucleus': 'H1',
        'reference_frequency_hz': 42.577, # or water?
        'reference_frequency_ppm': hdr[9],
        'fid_or_echo': 'fid',
        'apodizing': hdr[11],
        'num_zeros_view': hdr[12],
    }

    data = data[...,0] + 1j * data[...,1] # re + im

    data = data.reshape((n_frames, header['number_of_points'])).T
    data = data.squeeze()
    data = data.conj()

    #newshape = (1, 1, 1) + data.shape
    #data = data.reshape(newshape)
    # data shape (1, 1, 1, t, 1, N)  (t, channels, samples)
    return data, header, file_str