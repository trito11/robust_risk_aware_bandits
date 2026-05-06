import numpy as np

def report_data(path):
    d = np.load(path)
    ctx_tr = d['train_contexts']
    ctx_te = d['test_contexts']
    act_tr = d['train_actions']
    patients = {0: 'child#001', 2: 'adolescent#001'}
    
    print(f"--- DATASET: {path} ---")
    print("\n[TRAIN SET]")
    for p in [0, 2]:
        mask = (ctx_tr[:, 2] == p)
        count = np.sum(mask)
        u, c = np.unique(act_tr[mask], return_counts=True)
        print(f" {patients[p]}: {count} samples")
        print(f"   Distribution: {dict(zip(u, c))}")
        
    print("\n[TEST SET]")
    for p in [0, 2]:
        mask = (ctx_te[:, 2] == p)
        count = np.sum(mask)
        print(f" {patients[p]}: {count} samples")

if __name__ == "__main__":
    report_data("data/simglucose_offline.npz")
