import random

class MovementModel():
    def __init__(self):
        super(MovementModel, self).__init__()
        # Define stuff here

    def predict(inputs=None):
        # Bla bla bla
        if inputs == None: #Demo Mode
            verdicts = ["High Movement", "Medium Movement", "No Movement"]
            rand_index = random.randint(0, 2)
            return verdicts[rand_index]