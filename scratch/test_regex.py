import re
fname = "risk_lin_lcb_robust_syn_linear_d=20_a=30_pi=eps-greedy0.1_std=0.1_n=100.npz"
n = 100
regex = f"_n={n}(_|\.npz)"
print(f"Regex: {regex}")
print(f"Match: {bool(re.search(regex, fname))}")

regex2 = f"_n={n}(_|\\.npz)"
print(f"Regex2: {regex2}")
print(f"Match2: {bool(re.search(regex2, fname))}")
