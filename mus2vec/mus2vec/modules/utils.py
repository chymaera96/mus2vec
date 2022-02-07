import librosa
import numpy as np
import scipy

def compute_cqt_sync(y,sr):
    cqt_sync = []

    if len(y)/sr > 600:
        return cqt_sync
    
    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=40, n_bins=72, hop_length=256))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
    beats = librosa.util.fix_frames(beats,x_max=cqt.shape[1])
    
    for ix,beat in enumerate(beats[:-1]):
        cqt_beat = cqt.T[beats[ix]:beats[ix+1]+1]
    #     print(beats[ix],beats[ix+1]+1)
        segment = []
        for jx in np.arange(0,len(cqt_beat),len(cqt_beat)/63):
            jx = np.round(jx,4)
            fract = jx - int(jx)
            if int(jx) + 1 < len(cqt_beat):
                seg = (1-fract)*cqt_beat[int(jx)] + fract*cqt_beat[int(jx)+1]
            elif jx == len(cqt_beat):
                print(ix,jx)
                continue
            else:
    #             print(ix,jx,len(cqt_beat))
                seg = (1-fract)*cqt_beat[int(jx)] + fract*cqt_beat[-1]
            segment.append(seg)
        segment.append(cqt_beat[-1])
        cqt_sync.append(np.array(segment).T)

    return cqt_sync


def compute_cqt_segments(audio, beats, sr, n_cqt_per_beat = 64, beat_sync=True):
    segments=[]
    prev_frame=0
    for idx,frame in enumerate(beats):
        seg = audio[prev_frame:frame]
        segments.append(seg)
    
    cqt_segments=[]
    for seg in segments:
        # hop_length = int(seg.size/n_cqt_per_beat)
        # This variable hop_length will cause problems because it is not a multiple of 128
        hop_length = 128
        C = librosa.power_to_db(np.abs(librosa.cqt(y=audio, sr=sr,hop_length=hop_length, fmin=40)))
        cqt_segments.append(C)

    return np.array(cqt_segments)

def magnitude(X):
    """Magnitude of a complex matrix."""
    r = np.real(X)
    i = np.imag(X)
    return np.sqrt(r * r + i * i);
    
def compute_ffmc2d(X):
    """Computes the 2D-Fourier Magnitude Coefficients."""
    # 2d-fft
    fft2 = scipy.fftpack.fft2(X)

    # Magnitude
    fft2m = magnitude(fft2)

    # FFTshift and flatten
    fftshift = scipy.fftpack.fftshift(fft2m).flatten()

    #cmap = plt.cm.get_cmap('hot')
    #plt.imshow(np.log1p(scipy.fftpack.fftshift(fft2m)).T, interpolation="nearest",
    #    aspect="auto", cmap=cmap)
    #plt.show()

    # Take out redundant components
    return fftshift[:fftshift.shape[0] // 2 + 1]

def distance(X1, X2):
    """Computes the Euclidean distance between two vectors."""
    return np.linalg.norm(X1-X2)