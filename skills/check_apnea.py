import sys

sys.path.append("../Agent Tools/Apnea Monitor")

from AM import ApneaModel

def check_AM():
    model = ApneaModel()
    return ApneaModel.predict()

if __name__ == "__main__":
    print(check_AM())