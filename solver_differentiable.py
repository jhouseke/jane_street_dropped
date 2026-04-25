import torch
import glob
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    with torch.no_grad():
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
    with torch.no_grad():
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


def sinkhorn(M, tau, n_iters=20):
    """Numerically stable Sinkhorn-Knopp algorithm."""
    M = M / tau
    M = M - M.max(dim=1, keepdim=True)[0]
    K = torch.exp(M)
    for _ in range(n_iters):
        K = K / (K.sum(dim=1, keepdim=True) + 1e-8)
        K = K / (K.sum(dim=0, keepdim=True) + 1e-8)
    return K


def soft_forward(x_in, paired_a, paired_b, P, block_fn='res_relu'):
    """Forward pass using soft permutation matrix P."""
    x = x_in
    n = len(paired_a)
    for j in range(n):
        out = torch.zeros_like(x)
        for i in range(n):
            block_out = _apply_block(x, A[paired_a[i]], B[paired_b[i]], block_fn)
            out = out + P[j, i] * block_out
        x = out
    y_pred = x @ last_sd['weight'].T + last_sd['bias']
    return y_pred


def train_soft_permutation(paired_a, paired_b, block_fn='res_relu',
                           n_epochs=200, lr=0.01, init_tau=1.0, min_tau=0.01,
                           batch_size=512):
    """Train soft permutation matrix using Sinkhorn relaxation."""
    n = len(paired_a)
    M = torch.randn(n, n, requires_grad=True)
    optimizer = torch.optim.Adam([M], lr=lr)

    best_mse = float('inf')
    best_M = None
    n_samples = X.shape[0]

    for epoch in range(n_epochs):
        tau = max(min_tau, init_tau * (0.99 ** epoch))

        P = sinkhorn(M, tau)

        # Sample a random mini-batch each epoch for speed
        if batch_size < n_samples:
            idx = torch.randperm(n_samples)[:batch_size]
            x_batch = X[idx]
            y_batch = y_true[idx]
        else:
            x_batch = X
            y_batch = y_true

        optimizer.zero_grad()
        y_pred = soft_forward(x_batch, paired_a, paired_b, P, block_fn)
        loss = torch.mean((y_pred - y_batch) ** 2)

        loss.backward()
        optimizer.step()

        mse = loss.item()
        if mse < best_mse:
            best_mse = mse
            best_M = M.detach().clone()

        if epoch % 50 == 0:
            print(f"[{block_fn}] Epoch {epoch}: MSE={mse:.6f}, tau={tau:.4f}")

        if mse < 1e-12:
            break

    # Collapse to hard permutation using Hungarian on best_M
    with torch.no_grad():
        P_final = sinkhorn(best_M, tau=0.05, n_iters=50).numpy()
    row_ind, col_ind = linear_sum_assignment(-P_final)

    a_ord = [paired_a[col_ind[j]] for j in range(n)]
    b_ord = [paired_b[col_ind[j]] for j in range(n)]

    hard_mse = evaluate_sequence(a_ord, b_ord, block_fn)
    print(f"[{block_fn}] Soft best MSE: {best_mse:.6f}, Hard MSE: {hard_mse:.6f}")

    return hard_mse, a_ord, b_ord


def hill_climb_swap(current_a, current_b, block_fn,
                    max_iterations=5000, stagnation_limit=200, log_prefix=""):
    """Pairwise-swap hill climbing with prefix caching and stagnation cutoff."""
    n = len(current_a)
    current_mse, cache = forward_with_cache(current_a, current_b, block_fn)
    if current_mse < 1e-12:
        return current_mse, current_a, current_b

    iters = 0
    stagnant_scans = 0
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
            if improved:
                break
        if not improved:
            stagnant_scans += 1
            if stagnant_scans >= stagnation_limit:
                break
        else:
            stagnant_scans = 0
        if iters % 100 == 0:
            print(f"{log_prefix}swap iter {iters}: MSE={current_mse:.6f}", flush=True)
    return current_mse, current_a, current_b


def hill_climb_b_repair(current_a, current_b, block_fn,
                        max_iterations=2000, stagnation_limit=100, log_prefix=""):
    """Swap B pieces between positions only (A is fixed) with prefix caching."""
    n = len(current_b)
    current_mse, cache = forward_with_cache(current_a, current_b, block_fn)
    if current_mse < 1e-12:
        return current_mse, current_b

    iters = 0
    stagnant_scans = 0
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
            if improved:
                break
        if not improved:
            stagnant_scans += 1
            if stagnant_scans >= stagnation_limit:
                break
        else:
            stagnant_scans = 0
        if iters % 100 == 0:
            print(f"{log_prefix}b-repair iter {iters}: MSE={current_mse:.6f}", flush=True)
    return current_mse, current_b


def run_block_function(block_fn, paired_a, paired_b):
    """Soft permutation search + hill climbing for a single block function."""
    torch.set_grad_enabled(True)
    torch.set_num_threads(2)
    log_prefix = f"[{block_fn}] "
    t0 = time.perf_counter()

    # Train soft permutation
    soft_mse, a_ord, b_ord = train_soft_permutation(
        paired_a, paired_b, block_fn=block_fn
    )

    print(f"{log_prefix}soft best MSE: {soft_mse:.6f}", flush=True)

    if soft_mse < 1e-12:
        perm = build_permutation(a_ord, b_ord)
        return block_fn, soft_mse, perm, time.perf_counter() - t0

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

    # Run all 7 block functions in parallel
    print(f"\nLaunching {len(BLOCK_FNS)} block-function workers in parallel...")
    best_overall_mse = float('inf')
    best_overall_perm = None
    best_overall_fn = None

    n_workers = min(len(BLOCK_FNS), max(1, os.cpu_count() // 2))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_block_function, fn, paired_a, paired_b): fn
                   for fn in BLOCK_FNS}
        for fut in as_completed(futures):
            fn, mse, perm, dt = fut.result()
            print(f"\n[{fn}] DONE in {dt:.1f}s, MSE={mse:.6f}")
            if mse < best_overall_mse:
                best_overall_mse = mse
                best_overall_perm = perm
                best_overall_fn = fn
            if mse < 1e-12:
                print(f">>> FOUND EXACT SOLUTION with {fn}!")
                print(f">>> Permutation: {perm}")
                with open('solution.json', 'w') as f:
                    json.dump(perm, f)
                with open('solution.txt', 'w') as f:
                    f.write(','.join(map(str, perm)) + '\n')
                # Cancel pending futures (running ones will continue but we have our answer)
                for other_fut, other_fn in futures.items():
                    if not other_fut.done():
                        other_fut.cancel()
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
