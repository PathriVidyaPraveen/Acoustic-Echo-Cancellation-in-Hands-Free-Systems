import numpy as np

class PBFDAF:
    def __init__(self,num_partitions,fft_size):
        self.num_partitions = num_partitions
        self.num_bins = fft_size//2 + 1
        self.W = np.zeros((num_partitions,self.num_bins),dtype=np.complex64)
        self.X_history = np.zeros((num_partitions,self.num_bins),dtype=np.complex64)

    def update_input_history(self,X_f):

        #updates internal frequency-domain delay line with new input block
        if X_f.shape[0] != self.num_bins:
            raise ValueError(f"Input spectrum size {X_f.shape[0]} does not match expected {self.num_bins}")
        
        self.X_history = np.roll(self.X_history,1,axis=0)
        self.X_history[0] = X_f

    def estimate_echo(self):

        # computes echo estimate in frequency domain with partitioned convolution
        Y_f = np.sum(self.W * self.X_history,axis=0)
        return Y_f
    
    

