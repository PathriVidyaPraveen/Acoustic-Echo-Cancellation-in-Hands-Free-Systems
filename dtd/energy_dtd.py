class EnergyDTD:
    # simple energy based double talk detector

    def __init__(self,threshold=1.0, epsilon=1e-10):
        self.threshold = threshold
        self.epsilon=epsilon

    def detect(self,x_energy,e_energy):
        #detect double talk based on energy ratio

        if x_energy < self.epsilon:
            return True
        
        is_double_talk = False
        if e_energy > self.threshold*x_energy:
            is_double_talk=True

        return is_double_talk
