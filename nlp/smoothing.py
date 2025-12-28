import numpy as np

class GainSmoother:
    #implement tempioral exponential smoothing for frequency domain gains

    def __init__(self,num_bins,alpha=0.6):
        self.num_bins = num_bins
        self.alpha = alpha
        self.prev_gain = np.ones(num_bins,dtype=np.float32)

    def smooth(self,current_gain):
        # apply exponential moving average to input gain vector

        if current_gain.shape[0] != self.num_bins:
            raise ValueError(f"Input gain size {current_gain.shape[0]} does not match initialized size {self.num_bins}")
        
        smoothed_gain = (self.alpha * self.prev_gain) + (1-self.alpha)*current_gain

        self.prev_gain = smoothed_gain
        return smoothed_gain
    
    