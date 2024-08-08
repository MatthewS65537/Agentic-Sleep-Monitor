import sys

sys.path.append("../Agent Tools/Snoring Monitor")

from SM import SnoringModel

def check_SM(models):
    model = models["SM"]
    return model.predict()

if __name__ == "__main__":
    print(check_SM())