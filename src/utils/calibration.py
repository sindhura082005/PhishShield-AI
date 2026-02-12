import numpy as np

class TemperatureScaler:
    def __init__(self, temp=1.0):
        self.temp = temp
    def set_temp(self, t): self.temp = float(t)
    def calibrate(self, logits, labels):
        # placeholder
        pass
    def transform(self, probs):
        # apply temperature to logits equivalent by inverse sigmoid
        logits = np.log(probs / (1 - probs))
        scaled = 1/(1+np.exp(-logits / self.temp))
        return scaled
