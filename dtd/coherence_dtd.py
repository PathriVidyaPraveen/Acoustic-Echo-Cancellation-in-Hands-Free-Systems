import numpy as np

class CoherenceDTD:
    
    # coherence based double talk detector 
    def __init__(self,threshold,smoothing=0.9,epsilon=1e-10):
        self.threshold = threshold
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.Pxx = None
        self.Pdd = None
        self.Pxd = None

    def detect(self, X_f, D_f):
        # detects double talk based on magnitude squred coherence
        if self.Pxx is None:
            self.Pxx = np.zeros_like(X_f,dtype=np.float32)
            self.Pdd = np.zeros_like(D_f,dtype=np.float32)
            self.Pxd =np.zeros_like(X_f, dtype=np.complex64)

        abs_X2 = np.abs(X_f)**2
        abs_D2 = np.abs(D_f)**2
        cross_XD = X_f * np.conj(D_f)
        self.Pxx= self.smoothing*self.Pxx + (1-self.smoothing)*abs_X2
        self.Pdd = self.smoothing*self.Pdd + (1-self.smoothing)*abs_D2
        self.Pxd = self.smoothing*self.Pxd + (1-self.smoothing)*cross_XD
        coherence_numerator = np.abs(self.Pxd)**2
        coherence_denominator = self.Pxx *self.Pdd + self.epsilon
        coherence = coherence_numerator/coherence_denominator
        avg_coherence = np.mean(coherence)

        is_double_talk = False
        if avg_coherence < self.threshold:
            is_double_talk=True

        return is_double_talk
