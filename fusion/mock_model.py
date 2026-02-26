import random

class MockModel:

    def __init__(self, name):
        self.name = name

    def decision_function(self, X):
        # simulate anomaly score between -1 and 1
        return [random.uniform(-1, 1)]
