import sys

from check_apnea import *
from check_hr import *
from check_movement import *

def anomaly_is_detected(hr, apnea, movement):
    return True

def check_vitals():
    while True:
        hr = check_HRM()
        apnea = check_AM()
        movement = check_MM()
        if anomaly_is_detected(hr, apnea, movement):
            break
    return f"#VITALS:\n##HEART RATE\n{hr}\n##APNEA\n{apnea}\n##MOVEMENT\n{movement}"