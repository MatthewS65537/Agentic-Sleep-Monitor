import sys

sys.path.append("../Agent Tools/Movement Monitor")

from MM import MovementModel

def check_MM():
    model = MovementModel()
    return MovementModel.predict()

if __name__ == "__main__":
    print(check_MM())