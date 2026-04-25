import numpy as np
import pandas as pd
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from datetime import datetime, timedelta
import os
from tqdm import tqdm

def collect_simglucose_data(n_samples=10000, save_path='data/simglucose_offline.npz'):
    """
    Use Simglucose to generate Offline Bandit data.
    Each sample is a time-step.
    """
    print("Initializing Simglucose environment...")
    from simglucose.actuator.pump import InsulinPump
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.simulation.scenario_gen import RandomScenario
    
    # Select virtual patient 'child#001'
    patient = T1DPatient.withName('child#001')
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    
    # Create random scenario for 10 days to get enough samples
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    scenario = RandomScenario(start_time=start_time, seed=1)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    
    # Behavior Policy: Use existing Basal-Bolus controller with noise
    controller = BBController()
    
    contexts = []
    actions = []
    rewards = []
    
    print(f"Collecting {n_samples} samples from simulator...")
    state, reward, done, info = env.reset()
    
    for i in tqdm(range(n_samples)):
        # 1. Context: Current BG, Expected Carbs, etc.
        bg = state.CGM
        # Get meal info from scenario
        meal = env.scenario.get_action(env.time).meal
        
        # Context vector (3 dimensions)
        ctx = np.array([bg, meal, 0.5]) 
        
        # 2. Action from Behavior Controller + Noise
        ctrl_action = controller.policy(state, reward, done, **info)
        bolus = ctrl_action.bolus
        
        # Add 30% noise to create sub-optimal offline data
        if np.random.rand() < 0.3:
            bolus += np.random.uniform(-1, 1)
        bolus = int(np.round(max(0, min(10, bolus)))) # Discretize to [0, 1, ..., 10]
        
        # 3. Step and get Reward
        action_to_env = ctrl_action._replace(bolus=bolus)
        state, reward, done, info = env.step(action_to_env)
        
        contexts.append(ctx)
        actions.append(bolus)
        # Simglucose reward is higher for lower risk
        rewards.append(reward) 
        
        if done:
            state, reward, done, info = env.reset()

    contexts = np.array(contexts)
    actions = np.array(actions)
    rewards = np.array(rewards)

    # Add Heavy-tailed noise (Outliers) to simulate sensor failure
    outlier_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    rewards[outlier_idx] += np.random.choice([-500, 500], size=len(outlier_idx))

    os.makedirs('data', exist_ok=True)
    np.savez(save_path, contexts=contexts, actions=actions, rewards=rewards)
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    collect_simglucose_data()
