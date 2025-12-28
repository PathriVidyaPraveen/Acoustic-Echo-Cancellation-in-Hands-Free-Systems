import numpy as np

def residual_nlp(E_f,Y_hat_f,min_gain):

    #apply classical residual echo suppression (non linear processor)in frequency doamin

    magnitude_E = np.abs(E_f)
    magnitude_Y = np.abs(Y_hat_f)

    aggressiveness = 1.5
    est_residual = magnitude_Y * aggressiveness

    epsilon = 1e-10
    gain = (magnitude_E - est_residual)/(magnitude_E + epsilon)

    gain = np.clip(gain,min_gain,1.0)
    E_enhanced = E_f * gain
    return E_enhanced


