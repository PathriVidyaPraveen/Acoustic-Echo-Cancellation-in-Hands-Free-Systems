import  numpy as np
import soundfile as sf
import scipy.signal as signal


def pre_emphasis(x,alpha=0.97):
    y = np.zeros_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


def generate_test_signals(duration_sec,fs,rir_length_sec=0.1):
    # generate synthetic audio signlas for testing

    np.random.seed(42)
    total_samples = int(duration_sec*fs)
    speech, fs_speech = sf.read("simulation/speech_farend.wav")
    assert fs_speech == fs, "Speech file must be 16 kHz"
    if speech.ndim > 1:
        speech = speech[:,0]

    if len(speech) < total_samples:
        reps = int(np.ceil(total_samples/len(speech)))
        speech = np.tile(speech, reps)


    far_end = speech[:total_samples].astype(np.float32)
    far_end /= np.max(np.abs(far_end) + 1e-10)

    rir_samples=int(rir_length_sec*fs)
    t = np.linspace(0,rir_length_sec,rir_samples)
    envelope= np.exp(-30*t)
    rir = envelope * np.random.normal(0,1,rir_samples)
    rir = rir/np.linalg.norm(rir)

    echo_full = np.convolve(far_end,rir,mode='full')
    echo_component = echo_full[:total_samples].astype(np.float32)
    clean_near_end = np.zeros(total_samples,dtype=np.float32)
    dt_start = int(0.4*total_samples)
    dt_end = int(0.6*total_samples)
    dt_len = dt_end-dt_start
    #clean_near_end[dt_start:dt_end]= np.random.normal(0,0.3,dt_len)
    background_noise = np.random.normal(0,0.001,total_samples).astype(np.float32)
    mic_signal = echo_component + clean_near_end + background_noise

    return far_end,mic_signal,clean_near_end,echo_component

