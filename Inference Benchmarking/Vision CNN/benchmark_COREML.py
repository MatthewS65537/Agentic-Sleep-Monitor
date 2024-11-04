import coremltools as ct
import numpy as np
from tqdm import tqdm
import time

st = time.time()

model = ct.models.MLModel('VisionCNN.mlpackage')

en = time.time()

print(f"[INFO] Loaded Model {en - st : .2f} seconds")

dataset = []
print("Generating Dataset")
for _ in tqdm(range(10000)):
	dataset.append(np.random.randn(1, 1, 112, 112))

# print("READY")
# time.sleep(10)
# print("STARTING")

print("Starting Inference")
for data in tqdm(dataset):
	predictions = model.predict({'x_1' : data})