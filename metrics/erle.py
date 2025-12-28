import numpy as np

def compute_erle(echo_signal,error_signal):
    # compute echo return loss enhancement

    epsilon = 1e-10
    power_echo = np.mean(echo_signal**2)
    power_error = np.mean(error_signal**2)

    erle_db = 10*np.log10((power_echo+epsilon)/(power_error+epsilon))
    return erle_db
