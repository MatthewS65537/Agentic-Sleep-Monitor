import random

class ApneaModel():
    def __init__(self):
        super(ApneaModel, self).__init__()
        # Define stuff here

    def predict(inputs=None):
        # Bla bla bla
        if inputs == None: #Demo Mode
            verdicts = ["Apnea Detected", "Snoring Detected", "Normal Breathing", "No Breathing"]
            rand_index = random.randint(0, 3)
            return verdicts[rand_index]