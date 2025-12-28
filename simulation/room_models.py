import numpy as np

def generate_exponential_rir(length,decay_rate=5.0):
    # generates synthetic room impulse response using exponentially decaying white noise

    t = np.linspace(0,1,length)
    envelope = np.exp(-decay_rate * t)
    noise = np.random.normal(0,1,length)
    rir = envelope*noise
    epsilon = 1e-10
    norm_factor = np.linalg.norm(rir) + epsilon
    rir = rir/norm_factor

    return rir.astype(np.float32)