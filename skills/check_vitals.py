from skills.check_snoring import *
from check_hr import *
from check_movement import *

models = {
    "SM" : SnoringModel("./model_checkpoints/AudioTransformer.pt", "cpu"),
    "HRM" : HRM_Model(),
    "MM" : MovementModel(),
}

def anomaly_is_detected(hr, apnea, movement):
    return True

def check_vitals():
    global models
    while True:
        hr = check_HRM(models)
        snoring = check_SM(models)
        movement = check_MM(models)
        # TODO: # log_values(save_to="logs.json")
        # time.sleep(60)
        # if anomaly_is_detected(hr, snoring, movement):
        #     break
        break
    return f"#VITALS:\n##HEART RATE\n{hr}\n##SNORING\n{snoring}\n##MOVEMENT\n{movement}"