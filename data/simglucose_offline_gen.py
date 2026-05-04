import numpy as np
import pandas as pd
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import Action
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

def sync_env_state(src_env, dst_env):
    """Synchronize physiological state and sensor state between environments."""
    dst_env.env.patient._state = src_env.env.patient._state.copy()
    dst_env.env.sensor.last_state = src_env.env.sensor.last_state
    dst_env.env.time = src_env.env.time
    # Reset internal scenario time to match master
    dst_env.env.scenario.start_time = src_env.env.time

class ManualMealScenario:
    """A scenario that only triggers a single meal at start_time."""
    def __init__(self, start_time, meal_size):
        self.start_time = start_time
        self.meal_size = meal_size
    def get_action(self, t):
        if t == self.start_time:
            return Action(meal=self.meal_size)
        return Action(meal=0)

def create_eval_env(p_name, meal_size, current_time, seed):
    patient = T1DPatient.withName(p_name)
    sensor = CGMSensor.withName('Dexcom', seed=seed)
    pump = InsulinPump.withName('Insulet')
    # Force the environment to have the exact meal we are covering
    scenario = ManualMealScenario(current_time, meal_size)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    env.reset()
    return env

def collect_simglucose_data(n_train=8000, n_test=2000, save_path='data/simglucose_offline.npz', alpha=0.05):
    patient_names = ['child#001', 'child#002', 'adolescent#001', 'adult#001']
    samples_per_patient = (n_train + n_test) // len(patient_names)
    test_per_patient = n_test // len(patient_names)
    n_oracle_trials = 5 # Number of noise realizations to estimate True CVaR for Oracle

    train_contexts, train_actions, train_rewards = [], [], []
    test_contexts, test_mean_matrix, test_cvar_matrix = [], [], []

    if not os.path.exists('data'):
        os.makedirs('data')

    for p_idx, p_name in enumerate(patient_names):
        print(f"\nProcessing: {p_name}")
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        # Master environment uses a standard scenario
        scenario = RandomScenario(start_time=start_date, seed=p_idx)
        master_env = T1DSimEnv(T1DPatient.withName(p_name), CGMSensor.withName('Dexcom', seed=p_idx), 
                        InsulinPump.withName('Insulet'), scenario)
        controller = BBController()
        state, reward, done, info = master_env.reset()

        pbar = tqdm(total=samples_per_patient)
        patient_count = 0
        
        while patient_count < samples_per_patient:
            meal = master_env.scenario.get_action(master_env.time).meal
            if meal > 0:
                ctx = np.array([state.CGM, meal, float(p_idx), 0.5])
                is_test = (patient_count >= (samples_per_patient - test_per_patient))
                
                try:
                    if not is_test:
                        # --- Collect Train Data ---
                        ctrl_action = controller.policy(state, reward, done, **info)
                        bolus = ctrl_action.bolus
                        if np.random.rand() < 0.3: bolus += np.random.uniform(-2, 2)
                        bolus = int(np.round(max(0, min(10, bolus))))
                        
                        eval_env = create_eval_env(p_name, meal, master_env.time, p_idx + 100)
                        sync_env_state(master_env, eval_env)
                        
                        bg_window = []
                        act = ctrl_action._replace(bolus=bolus)
                        for _ in range(180): # 3 hours
                            s, _, _, _ = eval_env.step(act)
                            bg_window.append(s.CGM)
                            act = ctrl_action._replace(bolus=0)
                        
                        train_contexts.append(ctx)
                        train_actions.append(bolus)
                        train_rewards.append(get_magni_reward(bg_window))
                    else:
                        # --- Collect Test Data (Oracle with True CVaR) ---
                        action_means, action_cvars = [], []
                        for a in range(11):
                            trial_rewards = []
                            for trial in range(n_oracle_trials):
                                # Use different seeds for noise realizations
                                eval_env = create_eval_env(p_name, meal, master_env.time, p_idx + trial * 13)
                                sync_env_state(master_env, eval_env)
                                
                                ctrl_action = controller.policy(state, reward, done, **info)
                                act = ctrl_action._replace(bolus=a)
                                bg_window = []
                                for _ in range(180):
                                    s, _, _, _ = eval_env.step(act)
                                    bg_window.append(s.CGM)
                                    act = ctrl_action._replace(bolus=0)
                                trial_rewards.append(get_magni_reward(bg_window))
                            
                            action_means.append(np.mean(trial_rewards))
                            sorted_rew = np.sort(trial_rewards)
                            # True CVaR at alpha level
                            action_cvars.append(np.mean(sorted_rew[:max(1, int(alpha * n_oracle_trials))]))
                        
                        test_contexts.append(ctx)
                        test_mean_matrix.append(action_means)
                        test_cvar_matrix.append(action_cvars)
                    
                    patient_count += 1
                    pbar.update(1)
                except Exception:
                    pass # Skip ODE errors

            # IMPORTANT: Advance master loop with a reasonable policy to visit healthy states
            # Use BBController's bolus to avoid hyperglycemia explosion
            master_action = controller.policy(state, reward, done, **info)
            state, reward, done, info = master_env.step(master_action)
            if done: 
                state, reward, done, info = master_env.reset()
        pbar.close()

    # Final Save with full statistics
    np.savez(save_path, 
             train_contexts=np.array(train_contexts), 
             train_actions=np.array(train_actions), 
             train_rewards=np.array(train_rewards), 
             test_contexts=np.array(test_contexts), 
             test_mean=np.array(test_mean_matrix),
             test_cvar=np.array(test_cvar_matrix))
    print(f"\nSuccess! Total samples: {len(train_contexts) + len(test_contexts)}")

if __name__ == "__main__":
    collect_simglucose_data()
