import numpy as np

data = np.load('data/simglucose_offline.npz')
train_ctx = data['train_contexts']
test_ctx = data['test_contexts']

print("Train Contexts p_idx unique values:", np.unique(train_ctx[:, 2]))
print("Test Contexts p_idx unique values:", np.unique(test_ctx[:, 2]))

for i in np.unique(train_ctx[:, 2]):
    n_train = np.sum(train_ctx[:, 2] == i)
    n_test = np.sum(test_ctx[:, 2] == i)
    print(f"p_idx {i}: {n_train} train, {n_test} test")
