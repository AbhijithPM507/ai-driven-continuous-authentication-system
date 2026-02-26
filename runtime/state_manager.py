# runtime/state_manager.py

class StateManager:

    def __init__(self, threshold):
        self.threshold = threshold
        self.state = "AUTHORIZED"

    def update(self, score):

        if score > self.threshold:
            self.state = "INTRUDER"
        else:
            self.state = "AUTHORIZED"

        return self.state
