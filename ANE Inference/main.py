import coremltools as ct
import numpy as np
from tqdm import tqdm
import time

st = time.time()

model = ct.models.MLModel('AudioClassificationTransformer.mlpackage')

en = time.time()

print(f"[INFO] Loaded Model {en - st : .2f} seconds")

dataset = []
for _ in range(1000):
	dataset.append(np.random.randn(1, 32, 937))

for data in tqdm(dataset):
	predictions = model.predict({'x_1' : data})