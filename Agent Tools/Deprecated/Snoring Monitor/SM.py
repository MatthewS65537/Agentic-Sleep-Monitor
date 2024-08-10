import random
from AudioClassificationModel import *
import torch
from tqdm import tqdm

class SnoringModel():
    def __init__(self, checkpoint, device=None):
        super(SnoringModel, self).__init__()
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        if not device==None:
            self.device = device
        self.model = AudioClassificationTransformer(num_classes=2)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(checkpoint))

        self.demo_data = torch.load("./data/mfcc_tensors.pt")
        self.demo_labels = torch.load("./data/label_tensors.pt")

        print(f"[INFO] Loaded Snoring Detector on {self.device}.")

    # def predict(self, inputs=None):
    def predict(self, rand_index = None):
        # Bla bla bla
        # if inputs == None: #Demo Mode
        #     verdicts = ["Apnea Detected", "Snoring Detected", "Normal Breathing", "No Breathing"]
        #     rand_index = random.randint(0, 3)
        #     return verdicts[rand_index]
        if rand_index == None:
            rand_index = random.randint(0, 1000)
        data = self.demo_data[rand_index]
        data = torch.reshape(data, (1, 14, 2000))
        data = data.to(self.device)
        label = self.demo_labels[rand_index]
        label = label.to(self.device)
        logits = self.model(data)
        predicted_result = torch.argmax(logits)

        verdicts = {
            0 : "No Snoring Detected",
            1 : "Snoring Detected",
        }
        print(f"Predicted: {verdicts[predicted_result.item()]}, Actual: {verdicts[label.item()]}")

        return f"{verdicts[predicted_result.item()]}"
    
if __name__ == "__main__":
    model = SnoringModel("../../model_checkpoints/AudioTransformer.pt", "cpu")
    
    for i in tqdm(range(1000)):
        print(model.predict(rand_index = i))