import numpy as np
from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import Action
from datetime import datetime
from data.simglucose_offline_gen import ManualMealScenario, get_magni_reward

def test_single_eval():
    p_name = 'adolescent#001'
    meal = 50
    current_time = datetime(2024, 1, 1, 8, 0, 0)
    bolus = 5
    seed = 42

    patient = T1DPatient.withName(p_name)
    sensor = CGMSensor.withName('Dexcom', seed=seed)
    pump = InsulinPump.withName('Insulet')
    scenario = ManualMealScenario(current_time, meal)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    
    print("Resetting env...")
    env.reset()
    
    # Use the default patient state from reset for testing
    state_y = getattr(env.patient, '_state', getattr(env.patient, '_y', None))
    if state_y is None: state_y = env.patient._odesolver._y
    
    print(f"Initial state_y shape: {state_y.shape}")
    
    env.time = current_time
    env.scenario.start_time = current_time
    
    bg_window = []
    current_act = Action(insulin=bolus, meal=meal)
    print("Starting simulation steps...")
    try:
        for i in range(180):
            s, _, _, _ = env.step(current_act)
            bg_window.append(s.CGM)
            current_act = Action(insulin=0, meal=0)
            if i % 30 == 0:
                print(f"Step {i}, CGM: {s.CGM}")
    except Exception as e:
        print(f"Error during steps: {e}")
        return

    reward = get_magni_reward(bg_window)
    print(f"Final reward: {reward}")
    print(f"BG Window length: {len(bg_window)}")

if __name__ == "__main__":
    test_single_eval()
