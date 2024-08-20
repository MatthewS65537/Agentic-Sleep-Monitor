import os
import sys

sys.path.append("./Audio Monitor")

import argparse

import torch
import numpy as np
import platform
from AudioClassificationTransformer import *

if "macOS" in platform.platform() and "arm64" in platform.platform():
    import coremltools as ct

def get_inference_platform():
    platform_str = platform.platform()
    if "Windows" in platform_str or "Linux" in platform_str:
        return "Lightning"
    elif "macOS" in platform_str:
        if "arm64" in platform_str:
            # Apple Silicon Mac, has NPU
            return "ANE"
        else:
            return "Lightning"
    else:
        return "Lightning"
    
class UniversalInferenceEngine():
    def __init__(self, platform, model_name, checkpoint_dir = "../model_checkpoints", model_obj = None):
        self.platform = platform
        path_no_ext = os.path.join(checkpoint_dir, model_name)
        if self.platform == "Lightning":
            # Pytorch Lightning Inference
            self.model = model_obj.load_from_checkpoint(path_no_ext + ".ckpt")
            self.model.eval()
        elif self.platform == "ANE":
            # CoreML Inference
            self.model = ct.models.MLModel(path_no_ext + ".mlpackage")
        else:
            if self.platform in ["cpu", "cuda", "mps"]:
                # Pytorch Inference
                self.model = model_obj()
                self.model.load_state_dict(torch.load(path_no_ext + ".pt"))
                self.model.to(self.platform, dtype=torch.float32)
                self.model.eval()
            else:
                assert(False)
    
    def predict(self, data):
        # data -> numpy array of shape [1, 32, 937]
        if self.platform == "Lightning":
            data = torch.tensor(data).to(dtype=torch.float32)
            logits = self.model(data)
            label_idx = torch.argmax(logits)
        elif self.platform == "ANE":
            logits = self.model.predict({'x_1' : data})
            label_idx = np.argmax(logits)
        else:
            if self.platform in ["cpu", "cuda", "mps"]: 
                data = torch.tensor(data)
                logits = self.model(data.to(self.platform, dtype=torch.float32))
                label_idx = torch.argmax(logits)
            else:
                assert(False)
        return label_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Classification Model")
    parser.add_argument("--device", type=str, default=None, help="Specify the device to use (e.g., 'cpu', 'cuda', 'mps')")
    args = parser.parse_args()
    
    inf_platform = args.device
    if inf_platform == None:
        inf_platform = get_inference_platform()
    
    print(f"[INFO] Selected Inference Platform \"{inf_platform}\"")

    AudioModel = UniversalInferenceEngine(
        inf_platform,
        model_name = "AudioClassificationTransformer",
        model_obj = AudioTransformer if inf_platform == "Lightning" else AudioClassificationTransformer
    )
    randarray = np.random.randn(1, 32, 937)
    print(AudioModel.predict(randarray))
