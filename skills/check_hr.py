import sys

sys.path.append("../Agent Tools/Heart Rate Monitor")

from HRM import HRM_Model
from demo_monitor import *

def check_HRM(models):
    hr = demo_heart_rate_monitor()
    model = models["HRM"]
    model.set_data(50, 2)
    verdict = model.agent_check_anomaly(hr)
    return verdict

if __name__ == "__main__":
    print(check_HRM())