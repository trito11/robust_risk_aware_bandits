import numpy as np
import pandas as pd
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from datetime import datetime
import os
from tqdm import tqdm

def get_magni_reward(bg_history):
    if not bg_history: return -500.0
    bg_history = np.array(bg_history)
    bg_history[bg_history < 1] = 1 
    risk = 1.509 * (np.power(np.log(bg_history), 1.084) - 5.381)
    risk_score = 10 * np.power(risk, 2)
    return -np.mean(risk_score)

def create_env_at_state(p_name, seed, current_time):
    patient = T1DPatient.withName(p_name)
    sensor = CGMSensor.withName('Dexcom', seed=seed)
    pump = InsulinPump.withName('Insulet')
    scenario = RandomScenario(start_time=current_time, seed=seed)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    env.reset()
    env.time = current_time
    return env

def collect_simglucose_data(n_train=8000, n_test=2000, save_path='data/simglucose_offline.npz'):
    patient_names = ['child#001', 'child#002', 'adolescent#001', 'adult#001']
    samples_per_patient = (n_train + n_test) // len(patient_names)
    test_per_patient = n_test // len(patient_names)

    train_contexts, train_actions, train_rewards = [], [], []
    test_contexts, test_mean_matrix = [], []

    # Resume logic: Load existing data if available
    if os.path.exists(save_path):
        print(f"Resuming from existing data: {save_path}")
        d = np.load(save_path)
        train_contexts = list(d['train_contexts'])
        train_actions = list(d['train_actions'])
        train_rewards = list(d['train_rewards'])
        test_contexts = list(d['test_contexts'])
        test_mean_matrix = list(d['test_mean'])
        print(f"Resumed {len(train_contexts)} train and {len(test_contexts)} test samples.")

    total_existing = len(train_contexts) + len(test_contexts)
    if total_existing >= (n_train + n_test):
        print("Dataset already complete.")
        return

    print(f"Targeting {n_train} train and {n_test} test samples across 4 patients.")

    for p_idx, p_name in enumerate(patient_names):
        # Calculate how many samples we already have for THIS patient
        current_p_train = len([c for c in train_contexts if c[2] == p_idx])
        current_p_test = len([c for c in test_contexts if c[2] == p_idx])
        
        if current_p_train + current_p_test >= samples_per_patient:
            print(f"Patient {p_name} already completed. Skipping.")
            continue

        print(f"\nProcessing: {p_name}")
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        scenario = RandomScenario(start_time=start_date, seed=p_idx)
        env = T1DSimEnv(T1DPatient.withName(p_name), CGMSensor.withName('Dexcom', seed=p_idx), 
                        InsulinPump.withName('Insulet'), scenario)
        controller = BBController()
        state, reward, done, info = env.reset()

        pbar = tqdm(total=samples_per_patient, initial=current_p_train + current_p_test)
        
        patient_count = current_p_train + current_p_test
        while patient_count < samples_per_patient:
            meal = env.scenario.get_action(env.time).meal
            if meal > 0:
                ctx = np.array([state.CGM, meal, float(p_idx), 0.5])
                is_test = (patient_count >= (samples_per_patient - test_per_patient))
                
                try:
                    if not is_test:
                        # Collect Train Data
                        ctrl_action = controller.policy(state, reward, done, **info)
                        bolus = ctrl_action.bolus
                        if np.random.rand() < 0.3: bolus += np.random.uniform(-2, 2)
                        bolus = int(np.round(max(0, min(10, bolus))))
                        
                        temp_env = create_env_at_state(p_name, p_idx, env.time)
                        bg_window = []
                        act = ctrl_action._replace(bolus=bolus)
                        for _ in range(180): # 3 hours
                            s, _, _, _ = temp_env.step(act)
                            bg_window.append(s.CGM)
                            act = ctrl_action._replace(bolus=0)
                        
                        train_contexts.append(ctx)
                        train_actions.append(bolus)
                        train_rewards.append(get_magni_reward(bg_window))
                    else:
                        # Collect Test Data (Oracle)
                        action_rewards = []
                        for a in range(11):
                            temp_env = create_env_at_state(p_name, p_idx, env.time)
                            ctrl_action = controller.policy(state, reward, done, **info)
                            act = ctrl_action._replace(bolus=a)
                            bg_window = []
                            for _ in range(180):
                                s, _, _, _ = temp_env.step(act)
                                bg_window.append(s.CGM)
                                act = ctrl_action._replace(bolus=0)
                            action_rewards.append(get_magni_reward(bg_window))
                        
                        test_contexts.append(ctx)
                        test_mean_matrix.append(action_rewards)
                    
                    patient_count += 1
                    pbar.update(1)
                    
                    # Save checkpoint every 50 samples to prevent data loss
                    if patient_count % 50 == 0:
                        np.savez(save_path, 
                                 train_contexts=np.array(train_contexts), 
                                 train_actions=np.array(train_actions), 
                                 train_rewards=np.array(train_rewards), 
                                 test_contexts=np.array(test_contexts), 
                                 test_mean=np.array(test_mean_matrix))
                                 
                except Exception:
                    pass # Skip ODE errors

            state, reward, done, info = env.step(controller.policy(state, reward, done, **info)._replace(bolus=0))
            if done: env.reset()
        pbar.close()

    # Final Save
    np.savez(save_path, 
             train_contexts=np.array(train_contexts), 
             train_actions=np.array(train_actions), 
             train_rewards=np.array(train_rewards), 
             test_contexts=np.array(test_contexts), 
             test_mean=np.array(test_mean_matrix))
    print(f"\nSuccess! Total samples: {len(train_contexts) + len(test_contexts)}")

if __name__ == "__main__":
    collect_simglucose_data()
