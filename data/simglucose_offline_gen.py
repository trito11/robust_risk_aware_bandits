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
import multiprocessing as mp
import time
from collections import namedtuple
import sys

# Define a generic Action that works for SimGlucose environment step
EnvAction = namedtuple('EnvAction', ['basal', 'bolus'])
# Define the Action used by Scenario (only meal)
from simglucose.simulation.scenario import Action as ScenarioAction

def get_magni_reward(bg_history):
    if not bg_history: return -500.0
    bg_history = np.array(bg_history)
    bg_history[bg_history < 1] = 1 
    risk = 1.509 * (np.power(np.log(bg_history), 1.084) - 5.381)
    risk_score = 10 * np.power(risk, 2)
    return -np.mean(risk_score)

def sync_env_state(src_env, dst_env):
    """Synchronize physiological state and sensor state between environments."""
    # Sync patient state
    state_synced = False
    for attr in ['_state', '_y', '_y0', 'y0', 'state']:
        if hasattr(src_env.patient, attr):
            try:
                val = getattr(src_env.patient, attr)
                if isinstance(val, np.ndarray):
                    setattr(dst_env.patient, attr, val.copy())
                    state_synced = True
                    break
            except (AttributeError, Exception):
                continue
    
    if not state_synced:
        try:
            dst_env.patient._odesolver._y = src_env.patient._odesolver._y.copy()
            dst_env.patient._odesolver.t = src_env.patient._odesolver.t
            state_synced = True
        except Exception:
            pass

    # Sync environment time
    for attr in ['_time', 'env_time', 'time']:
        if hasattr(src_env, attr):
            try:
                setattr(dst_env, attr, getattr(src_env, attr))
                break
            except (AttributeError, Exception):
                continue

    # Sync sensor state
    if hasattr(src_env.sensor, 'last_state'):
        dst_env.sensor.last_state = src_env.sensor.last_state
    
    current_time = getattr(dst_env, 'time', getattr(dst_env, '_time', None))
    if current_time is not None:
        dst_env.scenario.start_time = current_time

class ManualMealScenario:
    """A scenario that triggers a single meal on the first step."""
    def __init__(self, start_time, meal_size):
        self.start_time = start_time
        self.meal_size = meal_size
        self.meal_sent = False
    def get_action(self, t):
        # Deliver meal on the very first request
        if not self.meal_sent:
            self.meal_sent = True
            return ScenarioAction(meal=self.meal_size)
        return ScenarioAction(meal=0)
    def reset(self):
        self.meal_sent = False

def single_sim_eval(p_name, meal, current_time, state_y, sensor_last_state, bolus, seed):
    """Helper to run a single simulation trace and return reward."""
    patient = T1DPatient.withName(p_name)
    sensor = CGMSensor.withName('Dexcom', seed=seed)
    pump = InsulinPump.withName('Insulet')
    # Scenario should NOT provide the meal if we pass it manually in env.step at the first step
    # Or better: let scenario handle the meal and we only pass insulin.
    scenario = ManualMealScenario(current_time, meal)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    env.reset()
    
    # Sync time and state carefully
    
    
    # Inject state via protected attributes only if they are not read-only
    for attr in ['_state', '_y', 'y']:
        if hasattr(env.patient, attr):
            try:
                setattr(env.patient, attr, state_y.copy())
            except: pass
        
    if hasattr(env.patient, '_odesolver'):
        try:
            # Re-initialize solver with the correct state and relative time 0
            env.patient._odesolver.set_initial_value(state_y.copy(), 0)
        except:
            try:
                env.patient._odesolver._y = state_y.copy()
                env.patient._odesolver.t = 0
            except: pass
            
    if sensor_last_state is not None:
        env.sensor.last_state = sensor_last_state
    
    # Force patient BG state into CGM sensor to avoid delay-induced crashes at start
    if hasattr(env.sensor, 'last_state'):
        # SimGlucose CGM sensor stores the last BG value to simulate delay
        # state_y[0] is usually the blood glucose (G) in Magni model
        env.sensor.last_state = state_y[0]
    
    bg_window = []
    # Use positional arguments for EnvAction (basal, bolus)
    # The meal is handled by ManualMealScenario.get_action(current_time)
    current_act = EnvAction(0, bolus) 
    
    for i in range(180):
        try:
            s, _, _, _ = env.step(current_act)
            if s is None or not hasattr(s, 'CGM') or np.isnan(s.CGM):
                break
            bg_window.append(s.CGM)
        except Exception as e:
            # print(f"Simulation step error: {e}")
            break
        current_act = EnvAction(0, 0)
    
    return get_magni_reward(bg_window)

def log_error(msg):
    with open("simglucose_error.log", "a") as f:
        f.write(f"{datetime.now()} - {msg}\n")

def eval_sim_worker(p_name, meal, current_time, state_y, sensor_last_state, bolus_list, seed_base, queue):
    """Worker function to run one or more simulations."""
    try:
        results = []
        for i, bolus in enumerate(bolus_list):
            try:
                res = single_sim_eval(p_name, meal, current_time, state_y, sensor_last_state, bolus, seed_base + i)
                results.append(res)
            except Exception as e:
                import traceback
                err_msg = f"Error in single_sim_eval for {p_name} bolus {bolus}: {e}\n{traceback.format_exc()}"
                log_error(err_msg)
                results.append(-500.0)
        queue.put(("SUCCESS", results))
    except Exception as e:
        import traceback
        err_msg = f"Fatal error in eval_sim_worker for {p_name}: {e}\n{traceback.format_exc()}"
        log_error(err_msg)
        queue.put(("ERROR", str(e)))

def collect_simglucose_data(n_train=2000, n_test=200, n_oracle_trials=10, save_path='data/simglucose_offline.npz', alpha=0.05):
    patient_names = ['child#001', 'child#002', 'adolescent#001', 'adult#001']
    samples_per_patient = (n_train + n_test) // len(patient_names)
    test_per_patient = n_test // len(patient_names)
    
    if not os.path.exists('data'):
        os.makedirs('data')

    train_contexts, train_actions, train_rewards = [], [], []
    test_contexts, test_mean_matrix, test_cvar_matrix = [], [], []

    # Resume from existing file if available
    if os.path.exists(save_path):
        try:
            with np.load(save_path, allow_pickle=True) as data:
                train_contexts = list(data['train_contexts'])
                train_actions = list(data['train_actions'])
                train_rewards = list(data['train_rewards'])
                test_contexts = list(data['test_contexts'])
                test_mean_matrix = list(data['test_mean'])
                test_cvar_matrix = list(data['test_cvar'])
            print(f" [RESUME] Loaded {len(train_contexts) + len(test_contexts)} existing samples from {save_path}")
        except Exception as e:
            print(f" [WARNING] Could not resume from {save_path}: {e}")

    for p_idx, p_name in enumerate(patient_names):
        print(f"\nProcessing: {p_name}")
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        scenario = RandomScenario(start_time=start_date, seed=p_idx)
        master_env = T1DSimEnv(T1DPatient.withName(p_name), CGMSensor.withName('Dexcom', seed=p_idx), 
                        InsulinPump.withName('Insulet'), scenario)
        controller = BBController()
        state, reward, done, info = master_env.reset()

        # Count existing samples for this patient to resume progress
        existing_train = sum(1 for c in train_contexts if c[2] == float(p_idx))
        existing_test = sum(1 for c in test_contexts if c[2] == float(p_idx))
        patient_count = existing_train + existing_test
        
        pbar = tqdm(total=samples_per_patient, initial=patient_count)
        
        while patient_count < samples_per_patient:
            meal = master_env.scenario.get_action(master_env.time).meal
            if meal > 0:
                ctx = np.array([state.CGM, meal, float(p_idx), 0.5])
                is_test = (patient_count >= (samples_per_patient - test_per_patient))
                
                # Snapshot state
                state_y = getattr(master_env.patient, '_state', getattr(master_env.patient, '_y', None))
                if state_y is None: state_y = master_env.patient._odesolver._y
                sensor_state = getattr(master_env.sensor, 'last_state', None)

                try:
                    if not is_test:
                        if state.CGM < 70:
                            patient_count += 1
                            pbar.update(1)
                            continue

                        tqdm.write(f" [{p_name}] Found meal: {meal}g at {master_env.time}. Collecting train sample...")
                        ctrl_action = controller.policy(state, reward, done, **info)
                        bolus = ctrl_action.bolus
                        max_bolus = 3 if 'child' in p_name else 10
                        if np.random.rand() < 0.3: 
                            noise = np.random.uniform(-1, 1) if 'child' in p_name else np.random.uniform(-2, 2)
                            bolus += noise
                        bolus = int(np.round(max(0, min(max_bolus, bolus))))
                        
                        queue = mp.Queue()
                        p = mp.Process(target=eval_sim_worker, args=(p_name, meal, master_env.time, state_y, sensor_state, [bolus], p_idx + 100 + patient_count, queue))
                        p.start()
                        p.join(timeout=40)
                        
                        if p.is_alive():
                            p.terminate()
                            p.join()
                        else:
                            res = queue.get() if not queue.empty() else ("TIMEOUT", None)
                            if res[0] == "SUCCESS":
                                train_contexts.append(ctx)
                                train_actions.append(bolus)
                                train_rewards.append(res[1][0])
                                patient_count += 1
                                pbar.update(1)
                        queue.close()
                        queue.join_thread()
                    else:
                        if state.CGM < 70:
                            pass 
                        else:
                            tqdm.write(f" [{p_name}] Found test meal: {meal}g at {master_env.time}. Running batched oracle...")
                            # BATCHED ORACLE EVALUATION: 11 actions * n_trials in ONE process
                            all_boluses = []
                            for a in range(11):
                                actual_a = a if 'child' not in p_name else min(a, 3)
                                all_boluses.extend([actual_a] * n_oracle_trials)
                            
                            queue = mp.Queue()
                            p = mp.Process(target=eval_sim_worker, args=(p_name, meal, master_env.time, state_y, sensor_state, all_boluses, p_idx + 500 + patient_count, queue))
                            p.start()
                            p.join(timeout=600) # Increased to 10 mins for 110 simulations
                            
                            if p.is_alive():
                                p.terminate()
                                p.join()
                                tqdm.write(f" [TIMEOUT] Oracle evaluation timed out for {p_name} at {master_env.time}")
                            else:
                                res = queue.get() if not queue.empty() else ("TIMEOUT", None)
                                if res[0] == "SUCCESS":
                                    all_rewards = res[1]
                                    action_means, action_cvars = [], []
                                    for a_idx in range(11):
                                        trial_rewards = all_rewards[a_idx * n_oracle_trials : (a_idx + 1) * n_oracle_trials]
                                        action_means.append(np.mean(trial_rewards))
                                        sorted_rew = np.sort(trial_rewards)
                                        action_cvars.append(np.mean(sorted_rew[:max(1, int(alpha * n_oracle_trials))]))
                                    
                                    test_contexts.append(ctx)
                                    test_mean_matrix.append(action_means)
                                    test_cvar_matrix.append(action_cvars)
                                    patient_count += 1
                                    pbar.update(1)
                                else:
                                    tqdm.write(f" [ERROR] Oracle evaluation failed for {p_name}: {res[1]}")
                            queue.close()
                            queue.join_thread()
                except Exception as e:
                    tqdm.write(f" [WARNING] Simulation error for {p_name} at {master_env.time}: {e}")

                # Checkpointing every 100 samples total
                total_samples = len(train_contexts) + len(test_contexts)
                if total_samples > 0 and total_samples % 100 == 0:
                    tqdm.write(f" [CHECKPOINT] Saving {total_samples} samples to {save_path}...")
                    np.savez(save_path, 
                             train_contexts=np.array(train_contexts), 
                             train_actions=np.array(train_actions), 
                             train_rewards=np.array(train_rewards), 
                             test_contexts=np.array(test_contexts), 
                             test_mean=np.array(test_mean_matrix),
                             test_cvar=np.array(test_cvar_matrix))

            try:
                master_action = controller.policy(state, reward, done, **info)
                state, reward, done, info = master_env.step(master_action)
                
                # Update description instead of printing with \r
                if master_env.time.minute == 0:
                    pbar.set_description(f"Processing {p_name} | Day: {master_env.time.date()}")

                if done: 
                    state, reward, done, info = master_env.reset()
            except Exception:
                state, reward, done, info = master_env.reset()
        pbar.close()

    np.savez(save_path, 
             train_contexts=np.array(train_contexts), 
             train_actions=np.array(train_actions), 
             train_rewards=np.array(train_rewards), 
             test_contexts=np.array(test_contexts), 
             test_mean=np.array(test_mean_matrix),
             test_cvar=np.array(test_cvar_matrix))
    print(f"\nSuccess! Total samples: {len(train_contexts) + len(test_contexts)}")
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    # Increased sample size for a more robust dataset
    collect_simglucose_data(n_train=2000, n_test=400, n_oracle_trials=20)

