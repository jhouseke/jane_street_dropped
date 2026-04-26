import torch
import glob
import pandas as pd
from scipy.optimize import linear_sum_assignment
import json
import time

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required to run solver_differentiable.py")

DEVICE = torch.device('cuda')
torch.set_default_dtype(torch.float64)
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(DEVICE)})")

# Load pieces
pieces = {}
for p in glob.glob('pieces/*.pth'):
    idx = int(p.split('_')[-1].split('.')[0])
    sd = torch.load(p, map_location=DEVICE, weights_only=True)
    sd = {
        key: value.to(device=DEVICE, dtype=torch.float64)
        if torch.is_tensor(value) else value
        for key, value in sd.items()
    }
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
X = torch.tensor(df.iloc[:, :-2].values, dtype=torch.float64, device=DEVICE)
y_true = torch.tensor(df['true'].values, dtype=torch.float64,
                      device=DEVICE).unsqueeze(1)
print(f"Data shape: X={X.shape}, y={y_true.shape}")


def _apply_block(x, a_sd, b_sd):
    h = x @ a_sd['weight'].T + a_sd['bias']
    h = torch.relu(h)
    out = h @ b_sd['weight'].T + b_sd['bias']
    return x + out


def stack_layers(a_list, b_list):
    """Stack A and B tensors so soft training uses batched CUDA kernels."""
    return (
        torch.stack([A[idx]['weight'] for idx in a_list]),
        torch.stack([A[idx]['bias'] for idx in a_list]),
        torch.stack([B[idx]['weight'] for idx in b_list]),
        torch.stack([B[idx]['bias'] for idx in b_list]),
    )


def _apply_all_a_layers(x, a_weight, a_bias):
    return torch.einsum('bd,nhd->bnh', x, a_weight) + a_bias.unsqueeze(0)


def _apply_all_b_layers(h, b_weight, b_bias):
    return torch.einsum('bh,ndh->bnd', h, b_weight) + b_bias.unsqueeze(0)


def evaluate_sequence(a_indices, b_indices, start=0, x_in=None):
    """MSE of the model from block `start` onwards, given x_in = activation entering block start."""
    with torch.no_grad():
        x = x_in if x_in is not None else X
        for k in range(start, len(a_indices)):
            x = _apply_block(x, A[a_indices[k]], B[b_indices[k]])
        y_pred = x @ last_sd['weight'].T + last_sd['bias']
        return torch.mean((y_pred - y_true) ** 2).item()


def forward_with_cache(a_indices, b_indices, start=0, prev_cache=None):
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
            x = _apply_block(x, A[a_indices[k]], B[b_indices[k]])
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


def soft_forward(x_in, a_list, b_list, P_A, P_B, layer_tensors=None):
    """Forward pass with independent soft routing for A and B layers."""
    x = x_in
    n = len(a_list)
    if layer_tensors is None:
        layer_tensors = stack_layers(a_list, b_list)
    a_weight, a_bias, b_weight, b_bias = layer_tensors

    # Collapse mixture-of-layers into one effective linear layer per step.
    eff_a_w = torch.einsum('sp,phd->shd', P_A, a_weight)
    eff_a_b = torch.einsum('sp,ph->sh', P_A, a_bias)
    eff_b_w = torch.einsum('sp,pdh->sdh', P_B, b_weight)
    eff_b_b = torch.einsum('sp,pd->sd', P_B, b_bias)

    for j in range(n):
        h = x @ eff_a_w[j].T + eff_a_b[j]
        out = torch.relu(h) @ eff_b_w[j].T + eff_b_b[j]
        x = x + out
    y_pred = x @ last_sd['weight'].T + last_sd['bias']
    return y_pred


def train_soft_permutation(a_list, b_list,
                           n_epochs=200, lr=0.01, init_tau=1.0, min_tau=0.01,
                           tau_decay=0.95):
    """Train soft permutation matrix using Sinkhorn relaxation."""
    n = len(a_list)
    M_A = torch.randn(n, n, device=DEVICE, requires_grad=True)
    M_B = torch.randn(n, n, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([M_A, M_B], lr=lr)

    best_mse = float('inf')
    best_M_A = None
    best_M_B = None
    layer_tensors = stack_layers(a_list, b_list)
    n_samples = X.shape[0]
    use_mse_refine = False

    for epoch in range(n_epochs):
        tau = max(min_tau, init_tau * (tau_decay ** epoch))
        sinkhorn_iters = 5 if tau > 0.1 else 10

        optimizer.zero_grad()
        # Build Sinkhorn graph exactly once per epoch; M_A/M_B are independent of
        # data chunks, so this avoids repeated graph construction.
        P_A = sinkhorn(M_A, tau, n_iters=sinkhorn_iters)
        P_B = sinkhorn(M_B, tau, n_iters=sinkhorn_iters)

        # Deterministic full-batch update.
        y_pred = soft_forward(X, a_list, b_list, P_A, P_B, layer_tensors=layer_tensors)
        if use_mse_refine:
            total_loss = torch.mean((y_pred - y_true) ** 2)
        else:
            total_loss = torch.nn.functional.huber_loss(
                y_pred, y_true, delta=1.0, reduction='mean'
            )

        total_loss.backward()
        optimizer.step()

        mse = torch.mean((y_pred.detach() - y_true) ** 2).item()
        if (not use_mse_refine) and mse < 0.1:
            use_mse_refine = True
            print("[res_relu] Switching loss from Huber to MSE refinement.")
        if mse < best_mse:
            best_mse = mse
            best_M_A = M_A.detach().clone()
            best_M_B = M_B.detach().clone()

        if epoch % 50 == 0:
            print(f"[res_relu] Epoch {epoch}: MSE={mse:.6f}, tau={tau:.4f}")

        if mse < 1e-12:
            break

    # Collapse to hard permutation using Hungarian on best_M
    with torch.no_grad():
        P_A_final = sinkhorn(best_M_A, tau=0.05, n_iters=50).detach().cpu().numpy()
        P_B_final = sinkhorn(best_M_B, tau=0.05, n_iters=50).detach().cpu().numpy()
    row_ind_a, col_ind_a = linear_sum_assignment(-P_A_final)
    row_ind_b, col_ind_b = linear_sum_assignment(-P_B_final)

    a_ord = [a_list[col_ind_a[j]] for j in range(n)]
    b_ord = [b_list[col_ind_b[j]] for j in range(n)]

    hard_mse = evaluate_sequence(a_ord, b_ord)
    print(f"[res_relu] Soft best MSE: {best_mse:.6f}, Hard MSE: {hard_mse:.6f}")

    return hard_mse, a_ord, b_ord


def move_item(items, src, dst):
    moved = items.copy()
    item = moved.pop(src)
    moved.insert(dst, item)
    return moved


def hill_climb_insert(current_a, current_b,
                      max_iterations=5000, stagnation_limit=200,
                      window_size=6,
                      log_prefix="[res_relu] "):
    """Pop-and-insert hill climbing with prefix caching and stagnation cutoff."""
    n = len(current_a)
    current_mse, cache = forward_with_cache(current_a, current_b)
    if current_mse < 1e-12:
        return current_mse, current_a, current_b

    iters = 0
    stagnant_scans = 0
    while iters < max_iterations:
        iters += 1
        improved = False
        for i in range(n):
            start_j = max(0, i - window_size)
            end_j = min(n, i + window_size + 1)
            for j in range(start_j, end_j):
                if i == j:
                    continue
                new_a = move_item(current_a, i, j)
                new_b = move_item(current_b, i, j)
                changed_start = min(i, j)
                mse = evaluate_sequence(new_a, new_b, start=changed_start,
                                        x_in=cache[changed_start])
                if mse < current_mse:
                    current_mse = mse
                    current_a = new_a
                    current_b = new_b
                    _, cache = forward_with_cache(current_a, current_b,
                                                  start=changed_start,
                                                  prev_cache=cache)
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
            print(f"{log_prefix}insert iter {iters}: MSE={current_mse:.6f}", flush=True)
    return current_mse, current_a, current_b


def hill_climb_b_repair(current_a, current_b,
                        max_iterations=2000, stagnation_limit=100,
                        log_prefix="[res_relu] "):
    """Pop and insert B pieces between positions only (A is fixed)."""
    n = len(current_b)
    current_mse, cache = forward_with_cache(current_a, current_b)
    if current_mse < 1e-12:
        return current_mse, current_b

    iters = 0
    stagnant_scans = 0
    while iters < max_iterations:
        iters += 1
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                new_b = move_item(current_b, i, j)
                changed_start = min(i, j)
                mse = evaluate_sequence(current_a, new_b, start=changed_start,
                                        x_in=cache[changed_start])
                if mse < current_mse:
                    current_mse = mse
                    current_b = new_b
                    _, cache = forward_with_cache(current_a, current_b,
                                                  start=changed_start,
                                                  prev_cache=cache)
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


def run_solver(a_list, b_list):
    """Soft permutation search + hill climbing for the residual ReLU block."""
    torch.set_grad_enabled(True)
    torch.set_num_threads(2)
    log_prefix = "[res_relu] "
    t0 = time.perf_counter()

    # Train soft permutation
    soft_mse, a_ord, b_ord = train_soft_permutation(a_list, b_list)

    print(f"{log_prefix}soft best MSE: {soft_mse:.6f}", flush=True)

    if soft_mse < 1e-12:
        perm = build_permutation(a_ord, b_ord)
        return soft_mse, perm, time.perf_counter() - t0

    # Pop-and-insert hill climbing
    insert_mse, a_ord, b_ord = hill_climb_insert(a_ord, b_ord,
                                                 log_prefix=log_prefix)
    print(f"{log_prefix}post-insert MSE: {insert_mse:.6f}", flush=True)
    if insert_mse < 1e-12:
        perm = build_permutation(a_ord, b_ord)
        return insert_mse, perm, time.perf_counter() - t0

    # B-repair hill climbing
    repair_mse, b_ord = hill_climb_b_repair(a_ord, b_ord,
                                            log_prefix=log_prefix)
    print(f"{log_prefix}post-repair MSE: {repair_mse:.6f}", flush=True)

    final_mse = min(insert_mse, repair_mse)
    perm = build_permutation(a_ord, b_ord)
    return final_mse, perm, time.perf_counter() - t0


def main():
    a_list = sorted(A.keys())
    b_list = sorted(B.keys())

    print(f"\nRunning residual ReLU solver on {DEVICE}...")
    mse, perm, dt = run_solver(a_list, b_list)
    print(f"\n[res_relu] DONE in {dt:.1f}s, MSE={mse:.6f}")
    if mse < 1e-12:
        print(">>> FOUND EXACT SOLUTION with res_relu!")
        print(f">>> Permutation: {perm}")
        with open('solution.json', 'w') as f:
            json.dump(perm, f)
        with open('solution.txt', 'w') as f:
            f.write(','.join(map(str, perm)) + '\n')
        return

    print(f"\n=== BEST OVERALL ===")
    print("Block function: res_relu")
    print(f"MSE: {mse:.6f}")
    print(f"Permutation: {perm}")

    with open('solution.json', 'w') as f:
        json.dump(perm, f)
    with open('solution.txt', 'w') as f:
        f.write(','.join(map(str, perm)) + '\n')


if __name__ == '__main__':
    main()
