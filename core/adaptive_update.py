import numpy as np

def pbfda_nlms_update(W,X_history,E,step_size,epsilon,*,adapt):

    #does NLMS update for PBFDAF

    if not adapt:
        return W
    
    # W = W*(1.0- step_size*leakage)
    
    input_power = np.sum(np.abs(X_history)**2 , axis=0)
    normalization = step_size/(input_power+epsilon)
    for p in range(W.shape[0]):
        W[p] += normalization*np.conj(X_history[p]) * E
        W[p] *= (1.0 - 1e-4)

    W_norm = np.linalg.norm(W)
    W_MAX = 50.0
    if W_norm > W_MAX:
        W *= (W_MAX / W_norm)

    return W

