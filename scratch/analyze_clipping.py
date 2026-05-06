import numpy as np

def analyze_rewards(path):
    print(f"Analyzing rewards in {path} to determine optimal TAU_N...")
    data = np.load(path)
    rewards = data['train_rewards']
    
    # Absolute values for clipping analysis
    abs_rewards = np.abs(rewards)
    
    print("\n[Reward Statistics]")
    print(f" Count: {len(rewards)}")
    print(f" Mean: {np.mean(rewards):.2f}")
    print(f" Std:  {np.std(rewards):.2f}")
    print(f" Min:  {np.min(rewards):.2f}")
    print(f" Max:  {np.max(rewards):.2f}")
    
    print("\n[Percentiles (Absolute Values)]")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        print(f" {p}th percentile: {np.percentile(abs_rewards, p):.2f}")
    
    print("\n[Clipping Impact Analysis]")
    thresholds = [50, 100, 150, 200, 300, 500]
    for t in thresholds:
        clipped_count = np.sum(abs_rewards > t)
        percent = (clipped_count / len(rewards)) * 100
        print(f" Threshold {t}: Clips {clipped_count} samples ({percent:.2f}%)")
        
    print("\n[Recommendation]")
    p95 = np.percentile(abs_rewards, 95)
    p99 = np.percentile(abs_rewards, 99)
    print(f" To preserve 95% of data, use TAU_N >= {p95:.2f}")
    print(f" To preserve 99% of data, use TAU_N >= {p99:.2f}")

if __name__ == "__main__":
    analyze_rewards("data/simglucose_offline.npz")
