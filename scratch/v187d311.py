import numpy as np
import pandas as pd
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from datetime import datetime, timedelta
import os
import copy
from tqdm import tqdm

def get_magni_reward(bg_history):
    """
    Calculate Magni Risk Score. Higher is safer.
    Magni function: Risk = 1.509 * (log(BG)^1.084 - 5.381)
    We take the negative risk as reward.
    """
    bg_history = np.array(bg_history)
    bg_history[bg_history < 1] = 1 # Avoid log(0)
    risk = 1.509 * (np.power(np.log(bg_history), 1.084) - 5.381)
    risk_score = 10 * np.power(risk, 2)
    return -np.mean(risk_score) # Average risk over the 4-hour window

def collect_simglucose_data(n_train=8000, n_test=2000, save_path='data/simglucose_offline.npz'):
    """
    Advanced Medical Bandit Generator following paper setup.
    """
    patient_names = ['child#001', 'child#002', 'adolescent#001', 'adult#001']
    n_samples = n_train + n_test
    samples_per_patient = n_samples // len(patient_names)
    
    train_contexts, train_actions, train_rewards = [], [], []
    test_contexts, test_mean_matrix = [], []

    print(f"Upgrading Simglucose to Proper Medical Bandit...")
    print(f"- Patients: {patient_names}")
    print(f"- Window: 4 hours (240 mins) post-bolus")
    print(f"- Actions: 11 levels (0-10 units)")

    for p_idx, p_name in enumerate(patient_names):
        print(f"\nProcessing clinical data for: {p_name}")
        
        # We use a long simulation to find enough meal events
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        scenario = RandomScenario(start_time=start_time, seed=p_idx)
        patient = T1DPatient.withName(p_name)
        sensor = CGMSensor.withName('Dexcom', seed=p_idx)
        pump = InsulinPump.withName('Insulet')
        env = T1DSimEnv(patient, sensor, pump, scenario)
        controller = BBController()

        state, reward, done, info = env.reset()
        
        count = 0
        pbar = tqdm(total=samples_per_patient, desc=f"Patient {p_idx}")
        
        while count < samples_per_patient:
            # Advance environment until a meal occurs
            meal = env.scenario.get_action(env.time).meal
            if meal > 0:
                # 1. Capture Context
                bg = state.CGM
                ctx = np.array([bg, meal, float(p_idx), 0.5])
                
                # Identify if this sample belongs to Train or Test
                is_test = (count >= (samples_per_patient - (n_test // 4)))
                
                if not is_test:
                    # TRAINING MODE: Simulate only the behavior action
                    ctrl_action = controller.policy(state, reward, done, **info)
                    bolus = ctrl_action.bolus
                    if np.random.rand() < 0.3: # Add sub-optimality
                         bolus += np.random.uniform(-2, 2)
                    bolus = int(np.round(max(0, min(10, bolus))))
                    
                    # Simulation: Run for 240 minutes
                    bg_window = []
                    curr_env = copy.deepcopy(env) # Snapshot
                    act = ctrl_action._replace(bolus=bolus)
                    
                    for _ in range(240): # 4 hours
                        s, r, d, i = curr_env.step(act)
                        bg_window.append(s.CGM)
                        act = ctrl_action._replace(bolus=0) # Only first step has bolus
                    
                    train_contexts.append(ctx)
                    train_actions.append(bolus)
                    train_rewards.append(get_magni_reward(bg_window))
                else:
                    # TEST MODE: Oracle Evaluation (Simulate ALL 11 actions)
                    action_rewards = []
                    for a in range(11):
                        curr_env = copy.deepcopy(env)
                        ctrl_action = controller.policy(state, reward, done, **info)
                        act = ctrl_action._replace(bolus=a)
                        
                        bg_window = []
                        for _ in range(240):
                            s, r, d, i = curr_env.step(act)
                            bg_window.append(s.CGM)
                            act = ctrl_action._replace(bolus=0)
                        action_rewards.append(get_magni_reward(bg_window))
                    
                    test_contexts.append(ctx)
                    test_mean_matrix.append(action_rewards)

                count += 1
                pbar.update(1)
            
            # Normal env step to next minute
            state, reward, done, info = env.step(controller.policy(state, reward, done, **info)._replace(bolus=0))
            if done: env.reset()
        pbar.close()

    # Convert to arrays
    train_contexts = np.array(train_contexts)
    train_actions = np.array(train_actions)
    train_rewards = np.array(train_rewards)
    
    test_contexts = np.array(test_contexts)
    test_mean_matrix = np.array(test_mean_matrix)

    # Calculate biological clean rewards for training (without sensor failure noise)
    rewards_clean = train_rewards.copy()

    # Add Heavy-tailed nose (Outliers) to 5% of training samples
    outlier_idx = np.random.choice(len(train_rewards), int(len(train_rewards) * 0.05), replace=False)
    train_rewards[outlier_idx] += np.random.choice([-500, 500], size=len(outlier_idx))

    os.makedirs('data', exist_ok=True)
    np.savez(save_path, 
             train_contexts=train_contexts, 
             train_actions=train_actions, 
             train_rewards=train_rewards,
             test_contexts=test_contexts,
             test_mean=test_mean_matrix,
             rewards_clean=rewards_clean)
    
    print(f"\nUpgrade complete! Data saved to {save_path}")
    print(f"- Train size: {len(train_contexts)}")
    print(f"- Test size: {len(test_contexts)} (with full 11-action oracle matrix)")

if __name__ == "__main__":
    collect_simglucose_data()
