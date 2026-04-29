import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_robust_results():
    # Cấu hình thư mục kết quả
    data_dir = "results/robust_syn_d=20_a=30_pi=eps-greedy0.1_std=0.1"
    
    if not os.path.exists(data_dir):
        print(f"Không tìm thấy thư mục: {data_dir}")
        return

    n_list = [1000, 2000, 3000, 5000, 10000, 15000, 20000]
    
    summary = {
        'n': [],
        'regret_mean': [], 'regret_std': [],
        'gt_mean_mean': [], 'gt_mean_std': [],
        'gt_var_mean': [], 'gt_var_std': [],
        'gt_cvar_mean': [], 'gt_cvar_std': [],
        'oracle_mean_vals': [],
        'oracle_var_vals': [],
        'oracle_cvar_vals': []
    }

    has_extra_stats = False

    n_regret = []
    n_extra = []

    for n in n_list:
        # Tìm file khớp với n và kiến trúc 32-32
        pattern = os.path.join(data_dir, f"*_n={n}_layers=32-32.npz")
        files = glob.glob(pattern)
        if not files:
            continue
            
        d = np.load(files[0])
        
        # Regret luôn có trong mọi file
        r_data = d['regrets'] 
        summary['regret_mean'].append(np.mean(r_data.flatten()))
        summary['regret_std'].append(np.std(r_data.flatten()) / np.sqrt(len(r_data.flatten()))) 
        n_regret.append(n)
        
        # Kiểm tra xem có chứa các chỉ số mở rộng không
        if 'gt_means' in d:
            has_extra_stats = True
            summary['gt_mean_mean'].append(np.mean(d['gt_means']))
            summary['gt_mean_std'].append(np.std(d['gt_means']) / np.sqrt(len(d['gt_means'])))
            
            summary['gt_var_mean'].append(np.mean(d['gt_vars']))
            summary['gt_var_std'].append(np.std(d['gt_vars']) / np.sqrt(len(d['gt_vars'])))
            
            summary['gt_cvar_mean'].append(np.mean(d['gt_cvars']))
            summary['gt_cvar_std'].append(np.std(d['gt_cvars']) / np.sqrt(len(d['gt_cvars'])))
            
            summary['oracle_mean_vals'].append(np.mean(d['oracle_means']))
            summary['oracle_var_vals'].append(np.mean(d['oracle_vars']))
            summary['oracle_cvar_vals'].append(np.mean(d['oracle_cvars']))
            n_extra.append(n)

    # Chuyển đổi sang numpy array
    for k in summary:
        summary[k] = np.array(summary[k])
    
    n_regret = np.array(n_regret)
    n_extra = np.array(n_extra)

    # --- VẼ BIỂU ĐỒ ---
    if not has_extra_stats:
        # Chỉ vẽ Regret nếu không có dữ liệu khác
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.errorbar(n_regret, summary['regret_mean'], yerr=summary['regret_std'], fmt='-o', capsize=5, label='Agent Regret')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Regret vs Sample Size (Log-Log)')
        ax.set_xlabel('n')
        ax.set_ylabel('Regret')
        ax.grid(True, which="both", ls="-", alpha=0.5)
    else:
        # Vẽ đầy đủ 4 biểu đồ
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Regret Plot
        ax = axes[0, 0]
        ax.errorbar(n_regret, summary['regret_mean'], yerr=summary['regret_std'], fmt='-o', capsize=5, label='Agent Regret')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Regret vs Sample Size (Log-Log)')
        ax.set_xlabel('n')
        ax.grid(True, which="both", ls="-", alpha=0.5)

        # 2. Mean Plot
        ax = axes[0, 1]
        ax.plot(n_extra, summary['gt_mean_mean'], 's-', label='Agent Mean', color='green')
        ax.fill_between(n_extra, summary['gt_mean_mean'] - summary['gt_mean_std'], 
                        summary['gt_mean_mean'] + summary['gt_mean_std'], alpha=0.2, color='green')
        ax.plot(n_extra, summary['oracle_mean_vals'], '--', color='black', label='Oracle Mean')
        ax.set_title('Mean Reward')
        ax.set_xlabel('n')
        ax.legend()

        # 3. CVaR Plot
        ax = axes[1, 0]
        ax.plot(n_extra, summary['gt_cvar_mean'], 'd-', label='Agent CVaR', color='red')
        ax.fill_between(n_extra, summary['gt_cvar_mean'] - summary['gt_cvar_std'], 
                        summary['gt_cvar_mean'] + summary['gt_cvar_std'], alpha=0.2, color='red')
        ax.plot(n_extra, summary['oracle_cvar_vals'], '--', color='black', label='Oracle CVaR')
        ax.set_title('CVaR (Risk)')
        ax.set_xlabel('n')
        ax.legend()

        # 4. Variance Plot
        ax = axes[1, 1]
        ax.plot(n_extra, summary['gt_var_mean'], 'v-', label='Agent Var', color='purple')
        ax.fill_between(n_extra, summary['gt_var_mean'] - summary['gt_var_std'], 
                        summary['gt_var_mean'] + summary['gt_var_std'], alpha=0.2, color='purple')
        ax.plot(n_extra, summary['oracle_var_vals'], '--', color='black', label='Oracle Var')
        ax.set_title('Reward Variance')
        ax.set_xlabel('n')
        ax.legend()

    plt.suptitle(f"Robust Bandit Performance Analysis", fontsize=16)
    
    # Lưu ảnh thay vì show
    save_path = "robust_performance_plots.png"
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")

if __name__ == "__main__":
    plot_robust_results()
