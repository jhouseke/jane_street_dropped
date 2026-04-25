import torch
import glob
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import json
import time

torch.set_grad_enabled(False)

# Load pieces
pieces = {}
for p in glob.glob('pieces/*.pth'):
    idx = int(p.split('_')[-1].split('.')[0])
    sd = torch.load(p, map_location='cpu', weights_only=True)
    pieces[idx] = sd

A = {}
B = {}
Last = {}
for idx, sd in pieces.items():
    w = sd['weight']
    if w.shape == (96, 48):
        A[idx] = sd
    elif w.shape == (48, 96):
        B[idx] = sd
    elif w.shape == (1, 48):
        Last[idx] = sd

assert len(A) == 48, f"Expected 48 A layers, got {len(A)}"
assert len(B) == 48, f"Expected 48 B layers, got {len(B)}"
assert len(Last) == 1, f"Expected 1 Last layer, got {len(Last)}"

last_idx = list(Last.keys())[0]
last_sd = Last[last_idx]
print(f"A layers: {len(A)}, B layers: {len(B)}, Last layer: {last_idx}")

# Load historical data
df = pd.read_csv('historical_data.csv')
X = torch.tensor(df.iloc[:, :-2].values, dtype=torch.float32)
y_true = torch.tensor(df['true'].values, dtype=torch.float32).unsqueeze(1)
print(f"Data shape: X={X.shape}, y={y_true.shape}")

BLOCK_FNS = ['res_relu', 'res_relu_post', 'res_tanh', 'res_none',
             'mlp_relu', 'mlp_tanh', 'mlp_none']


def _apply_block(x, a_sd, b_sd, block_fn):
    h = x @ a_sd['weight'].T + a_sd['bias']
    if block_fn == 'res_relu':
        h = torch.relu(h)
        out = h @ b_sd['weight'].T + b_sd['bias']
        return x + out
    if block_fn == 'res_relu_post':
        h = torch.relu(h)
        out = h @ b_sd['weight'].T + b_sd['bias']
        return torch.relu(x + out)
    if block_fn == 'res_tanh':
        h = torch.tanh(h)
        out = h @ b_sd['weight'].T + b_sd['bias']
        return x + out
    if block_fn == 'res_none':
        out = h @ b_sd['weight'].T + b_sd['bias']
        return x + out
    if block_fn == 'mlp_relu':
        h = torch.relu(h)
        return h @ b_sd['weight'].T + b_sd['bias']
    if block_fn == 'mlp_tanh':
        h = torch.tanh(h)
        return h @ b_sd['weight'].T + b_sd['bias']
    if block_fn == 'mlp_none':
        return h @ b_sd['weight'].T + b_sd['bias']
    raise ValueError(f"Unknown block_fn: {block_fn}")


def evaluate_sequence(a_indices, b_indices, block_fn='res_relu', start=0, x_in=None):
    """MSE of the model from block `start` onwards, given x_in = activation entering block start."""
    x = x_in if x_in is not None else X
    for k in range(start, len(a_indices)):
        x = _apply_block(x, A[a_indices[k]], B[b_indices[k]], block_fn)
    y_pred = x @ last_sd['weight'].T + last_sd['bias']
    return torch.mean((y_pred - y_true) ** 2).item()


def forward_with_cache(a_indices, b_indices, block_fn='res_relu',
                       start=0, prev_cache=None):
    """
    Returns (mse, cache) where cache[k] = activation entering block k (cache[n] = output).

    If start > 0, prev_cache must be supplied; cache[0..start] is reused from it.
    """
    if start == 0:
        cache = [X]
        x = X
    else:
        cache = list(prev_cache[:start + 1])
        x = cache[start]
    for k in range(start, len(a_indices)):
        x = _apply_block(x, A[a_indices[k]], B[b_indices[k]], block_fn)
        cache.append(x)
    y_pred = x @ last_sd['weight'].T + last_sd['bias']
    mse = torch.mean((y_pred - y_true) ** 2).item()
    return mse, cache


def build_permutation(a_indices, b_indices):
    """Build full permutation array of 97 indices from block A/B lists."""
    perm = []
    for a, b in zip(a_indices, b_indices):
        perm.append(a)
        perm.append(b)
    perm.append(last_idx)
    return perm


def between_block_score(b_idx, a_next_idx):
    w_b = B[b_idx]['weight']
    w_a = A[a_next_idx]['weight']
    prod = w_b @ w_a
    tr = torch.trace(prod).abs().item()
    frob = torch.norm(prod, 'fro').item()
    return tr / frob if frob > 0 else 0


def greedy_order(adj, n, start):
    order = [start]
    used = {start}
    current = start
    while len(order) < n:
        best_next = None
        best_next_score = -1
        for j in range(n):
            if j not in used and adj[current, j] > best_next_score:
                best_next_score = adj[current, j]
                best_next = j
        order.append(best_next)
        used.add(best_next)
        current = best_next
    return order


def hill_climb_swap(current_a, current_b, block_fn,
                    max_iterations=5000, log_prefix=""):
    """Pairwise-swap hill climbing with prefix caching. Stops on local minimum."""
    n = len(current_a)
    current_mse, cache = forward_with_cache(current_a, current_b, block_fn)
    if current_mse < 1e-12:
        return current_mse, current_a, current_b
    print(f"{log_prefix}swap start MSE={current_mse:.6f}", flush=True)

    iters = 0
    last_log = time.perf_counter()
    last_log_mse = current_mse
    while iters < max_iterations:
        iters += 1
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                new_a = current_a.copy()
                new_b = current_b.copy()
                new_a[i], new_a[j] = new_a[j], new_a[i]
                new_b[i], new_b[j] = new_b[j], new_b[i]
                mse = evaluate_sequence(new_a, new_b, block_fn,
                                        start=i, x_in=cache[i])
                if mse < current_mse:
                    current_mse = mse
                    current_a = new_a
                    current_b = new_b
                    _, cache = forward_with_cache(current_a, current_b, block_fn,
                                                  start=i, prev_cache=cache)
                    improved = True
                    if current_mse < 1e-12:
                        return current_mse, current_a, current_b
                    break
                # Heartbeat every ~10s of scanning
                now = time.perf_counter()
                if now - last_log > 10.0:
                    print(f"{log_prefix}swap iter={iters} scanning ({i},{j}) "
                          f"MSE={current_mse:.6f}", flush=True)
                    last_log = now
            if improved:
                break
        if improved:
            now = time.perf_counter()
            if now - last_log > 1.0 or current_mse < last_log_mse * 0.95:
                print(f"{log_prefix}swap iter={iters} MSE={current_mse:.6f}", flush=True)
                last_log = now
                last_log_mse = current_mse
        else:
            print(f"{log_prefix}swap converged at iter={iters} MSE={current_mse:.6f}", flush=True)
            break
    return current_mse, current_a, current_b


def hill_climb_b_repair(current_a, current_b, block_fn,
                        max_iterations=2000, log_prefix=""):
    """Swap B pieces between positions only (A is fixed) with prefix caching."""
    n = len(current_b)
    current_mse, cache = forward_with_cache(current_a, current_b, block_fn)
    if current_mse < 1e-12:
        return current_mse, current_b
    print(f"{log_prefix}b-repair start MSE={current_mse:.6f}", flush=True)

    iters = 0
    last_log = time.perf_counter()
    last_log_mse = current_mse
    while iters < max_iterations:
        iters += 1
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                new_b = current_b.copy()
                new_b[i], new_b[j] = new_b[j], new_b[i]
                mse = evaluate_sequence(current_a, new_b, block_fn,
                                        start=i, x_in=cache[i])
                if mse < current_mse:
                    current_mse = mse
                    current_b = new_b
                    _, cache = forward_with_cache(current_a, current_b, block_fn,
                                                  start=i, prev_cache=cache)
                    improved = True
                    if current_mse < 1e-12:
                        return current_mse, current_b
                    break
                now = time.perf_counter()
                if now - last_log > 10.0:
                    print(f"{log_prefix}b-repair iter={iters} scanning ({i},{j}) "
                          f"MSE={current_mse:.6f}", flush=True)
                    last_log = now
            if improved:
                break
        if improved:
            now = time.perf_counter()
            if now - last_log > 1.0 or current_mse < last_log_mse * 0.95:
                print(f"{log_prefix}b-repair iter={iters} MSE={current_mse:.6f}", flush=True)
                last_log = now
                last_log_mse = current_mse
        else:
            print(f"{log_prefix}b-repair converged at iter={iters} MSE={current_mse:.6f}", flush=True)
            break
    return current_mse, current_b


def run_block_function(block_fn, paired_a, paired_b, adj):
    """All per-function search work: 48 greedy starts + hill climb + B-repair."""
    n = len(paired_a)
    log_prefix = f"[{block_fn}] "
    t0 = time.perf_counter()

    # Greedy ordering from each possible start
    best_fn_mse = float('inf')
    best_fn_order = None
    for start in range(n):
        order = greedy_order(adj, n, start)
        a_ord = [paired_a[k] for k in order]
        b_ord = [paired_b[k] for k in order]
        mse = evaluate_sequence(a_ord, b_ord, block_fn=block_fn)
        if mse < best_fn_mse:
            best_fn_mse = mse
            best_fn_order = order
        if best_fn_mse < 1e-12:
            break

    print(f"{log_prefix}greedy best MSE: {best_fn_mse:.6f}", flush=True)

    a_ord = [paired_a[k] for k in best_fn_order]
    b_ord = [paired_b[k] for k in best_fn_order]

    if best_fn_mse < 1e-12:
        perm = build_permutation(a_ord, b_ord)
        return block_fn, best_fn_mse, perm, time.perf_counter() - t0

    # Pairwise-swap hill climbing
    swap_mse, a_ord, b_ord = hill_climb_swap(a_ord, b_ord, block_fn,
                                             log_prefix=log_prefix)
    print(f"{log_prefix}post-swap MSE: {swap_mse:.6f}", flush=True)
    if swap_mse < 1e-12:
        perm = build_permutation(a_ord, b_ord)
        return block_fn, swap_mse, perm, time.perf_counter() - t0

    # B-repair hill climbing
    repair_mse, b_ord = hill_climb_b_repair(a_ord, b_ord, block_fn,
                                            log_prefix=log_prefix)
    print(f"{log_prefix}post-repair MSE: {repair_mse:.6f}", flush=True)

    final_mse = min(swap_mse, repair_mse)
    perm = build_permutation(a_ord, b_ord)
    return block_fn, final_mse, perm, time.perf_counter() - t0


def main():
    # Pair A with B using Hungarian on within-block scores
    a_list = sorted(A.keys())
    b_list = sorted(B.keys())

    cost_matrix = np.zeros((48, 48))
    for i, a_idx in enumerate(a_list):
        for j, b_idx in enumerate(b_list):
            w_a = A[a_idx]['weight']
            w_b = B[b_idx]['weight']
            prod = w_b @ w_a
            tr = torch.trace(prod).abs().item()
            frob = torch.norm(prod, 'fro').item()
            score = tr / frob if frob > 0 else 0
            cost_matrix[i, j] = -score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    paired_a = [a_list[i] for i in row_ind]
    paired_b = [b_list[j] for j in col_ind]

    print("\nHungarian pairing MSE tests:")
    for fn in BLOCK_FNS:
        mse = evaluate_sequence(paired_a, paired_b, block_fn=fn)
        print(f"  {fn}: MSE={mse:.6f}")
        if mse < 1e-12:
            perm = build_permutation(paired_a, paired_b)
            print(f"  >>> FOUND EXACT SOLUTION with {fn}!")
            print(f"  >>> Permutation: {perm}")
            with open('solution.json', 'w') as f:
                json.dump(perm, f)
            with open('solution.txt', 'w') as f:
                f.write(','.join(map(str, perm)) + '\n')
            return

    # Between-block adjacency for paired blocks
    n = 48
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i, j] = between_block_score(paired_b[i], paired_a[j])

    # Run each block function sequentially (torch+OpenMP doesn't play well with fork
    # on this build; the prefix-caching speedup makes serial fast enough).
    print(f"\nRunning {len(BLOCK_FNS)} block functions sequentially...")
    best_overall_mse = float('inf')
    best_overall_perm = None
    best_overall_fn = None

    for fn in BLOCK_FNS:
        fn_name, mse, perm, dt = run_block_function(fn, paired_a, paired_b, adj)
        print(f"[{fn_name}] DONE in {dt:.1f}s, MSE={mse:.6f}")
        if mse < best_overall_mse:
            best_overall_mse = mse
            best_overall_perm = perm
            best_overall_fn = fn_name
        if mse < 1e-12:
            print(f">>> FOUND EXACT SOLUTION with {fn_name}!")
            print(f">>> Permutation: {perm}")
            with open('solution.json', 'w') as f:
                json.dump(perm, f)
            with open('solution.txt', 'w') as f:
                f.write(','.join(map(str, perm)) + '\n')
            return

    print(f"\n=== BEST OVERALL ===")
    print(f"Block function: {best_overall_fn}")
    print(f"MSE: {best_overall_mse:.6f}")
    print(f"Permutation: {best_overall_perm}")

    with open('solution.json', 'w') as f:
        json.dump(best_overall_perm, f)
    with open('solution.txt', 'w') as f:
        f.write(','.join(map(str, best_overall_perm)) + '\n')


if __name__ == '__main__':
    main()
