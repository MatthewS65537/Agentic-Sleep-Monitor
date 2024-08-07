import sys

from check_apnea import *
from check_hr import *
from check_movement import *

def check_vitals():
    hr = check_HRM()
    apnea = check_AM()
    movement = check_MM()
    return f"#VITALS:\n##HEART RATE\n{hr}\n##APNEA\n{apnea}\n##MOVEMENT\n{movement}"