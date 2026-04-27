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


def gumbel_sinkhorn(M, tau, n_iters=20, noise_factor=0.0, use_ste=False):
    """Sinkhorn with optional Gumbel noise and Straight-Through Estimator."""
    if noise_factor > 0 and M.requires_grad:
        gumbel_noise = -torch.empty_like(M).exponential_().log()
        M_eff = M + gumbel_noise * noise_factor
    else:
        M_eff = M

    M_scaled = M_eff / tau
    M_scaled = M_scaled - M_scaled.max(dim=1, keepdim=True)[0]
    K = torch.exp(M_scaled)
    for _ in range(n_iters):
        K = K / (K.sum(dim=1, keepdim=True) + 1e-8)
        K = K / (K.sum(dim=0, keepdim=True) + 1e-8)

    if not use_ste:
        return K

    with torch.no_grad():
        row_ind, col_ind = linear_sum_assignment(-K.detach().cpu().numpy())
        P_hard = torch.zeros_like(K)
        P_hard[row_ind, col_ind] = 1.0

    # Forward uses hard assignment, backward follows soft Sinkhorn gradients.
    P_ste = (P_hard - K).detach() + K
    return P_ste


def soft_forward(x_in, paired_a, paired_b, P_A, P_B=None, layer_tensors=None):
    """Forward pass with decoupled soft routing for A and B."""
    x = x_in
    n = len(paired_a)
    if layer_tensors is None:
        layer_tensors = stack_layers(paired_a, paired_b)
    a_weight, a_bias, b_weight, b_bias = layer_tensors
    if P_B is None:
        P_B = P_A

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


def build_pairing_cost_matrix(a_list, b_list):
    a_weight = torch.stack([A[idx]['weight'] for idx in a_list])
    b_weight = torch.stack([B[idx]['weight'] for idx in b_list])
    prod = torch.einsum('joh,ihd->ijod', b_weight, a_weight)
    tr = torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1)
    frob = torch.sqrt(torch.sum(prod * prod, dim=(-2, -1)))
    score = torch.where(frob > 0, tr / frob, torch.zeros_like(frob))
    return score.detach().cpu().numpy()


def train_soft_permutation(paired_a, paired_b,
                           n_epochs=600, lr=0.01, init_tau=1.0, min_tau=0.01,
                           tau_decay=None):
    """Alternating soft stage: ordering (A) vs pairing (B) updates."""
    n = len(paired_a)
    # Shared identity-biased prior so the soft stage starts with the
    # heuristic A/B pairing respected (position j -> paired_a[j], paired_b[j]).
    identity_prior = torch.eye(n, device=DEVICE) * 5.0
    M_A = (identity_prior + 0.1 * torch.randn(n, n, device=DEVICE)).detach().clone().requires_grad_(True)
    M_B = (identity_prior + 0.1 * torch.randn(n, n, device=DEVICE)).detach().clone().requires_grad_(True)
    optimize_a = True
    optimizer = torch.optim.Adam([M_A], lr=lr)
    best_mse = float('inf')
    best_M_A = None
    best_M_B = None
    layer_tensors = stack_layers(paired_a, paired_b)
    use_mse_refine = False
    phase_steps = 500
    n_alternations = 8
    total_steps = phase_steps * n_alternations
    if n_epochs != total_steps:
        print(f"[res_relu] Overriding n_epochs={n_epochs} with scheduled steps={total_steps}.")
    if tau_decay is None:
        tau_decay = (min_tau / init_tau) ** (1.0 / max(1, total_steps))

    for step in range(total_steps):
        phase_idx = step // phase_steps
        desired_optimize_a = (phase_idx % 2 == 0)
        if desired_optimize_a != optimize_a:
            optimize_a = desired_optimize_a
            optimizer = torch.optim.Adam([M_A if optimize_a else M_B], lr=lr)

        tau = max(min_tau, init_tau * (tau_decay ** step))
        sinkhorn_iters = 10 if tau > 0.5 else (20 if tau > 0.1 else 40)
        use_ste = False
        noise = max(0.2, 1.0 * tau) if not use_ste else 0.0
        optimizer.zero_grad()
        P_A = gumbel_sinkhorn(M_A, tau, n_iters=sinkhorn_iters,
                              noise_factor=noise, use_ste=use_ste)
        P_B = gumbel_sinkhorn(M_B, tau, n_iters=sinkhorn_iters,
                              noise_factor=noise, use_ste=use_ste)
        y_pred = soft_forward(X, paired_a, paired_b, P_A, P_B, layer_tensors=layer_tensors)
        if use_mse_refine:
            total_loss = torch.mean((y_pred - y_true) ** 2)
        else:
            total_loss = torch.nn.functional.huber_loss(
                y_pred, y_true, delta=1.0, reduction='mean'
            )
        total_loss.backward()
        if optimize_a:
            M_B.grad = None
        else:
            M_A.grad = None
        optimizer.step()

        mse = torch.mean((y_pred.detach() - y_true) ** 2).item()
        if (not use_mse_refine) and mse < 0.1:
            use_mse_refine = True
            print("[res_relu] Switching loss from Huber to MSE refinement.")
        if mse < best_mse:
            best_mse = mse
            best_M_A = M_A.detach().clone()
            best_M_B = M_B.detach().clone()
        if step % 50 == 0:
            phase = "ordering" if optimize_a else "pairing"
            print(f"[res_relu] Step {step}: MSE={mse:.6f}, tau={tau:.4f}, phase={phase}")
        if mse < 1e-12:
            break

    with torch.no_grad():
        P_A_final = gumbel_sinkhorn(best_M_A, tau=0.01, n_iters=100,
                                    noise_factor=0.0, use_ste=False).detach().cpu().numpy()
        P_B_final = gumbel_sinkhorn(best_M_B, tau=0.01, n_iters=100,
                                    noise_factor=0.0, use_ste=False).detach().cpu().numpy()
    _, col_ind_a = linear_sum_assignment(-P_A_final)
    _, col_ind_b = linear_sum_assignment(-P_B_final)
    a_ord = [paired_a[col_ind_a[j]] for j in range(n)]
    b_ord = [paired_b[col_ind_b[j]] for j in range(n)]
    hard_mse = evaluate_sequence(a_ord, b_ord)
    print(f"[res_relu] Soft best MSE: {best_mse:.6f}, Hard MSE: {hard_mse:.6f}")
    return hard_mse, a_ord, b_ord


def move_item(items, src, dst):
    moved = items.copy()
    item = moved.pop(src)
    moved.insert(dst, item)
    return moved


def swap_item(items, i, j):
    swapped = items.copy()
    swapped[i], swapped[j] = swapped[j], swapped[i]
    return swapped


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
                changed_start = min(i, j)
                new_a_insert = move_item(current_a, i, j)
                new_b_insert = move_item(current_b, i, j)
                mse_insert = evaluate_sequence(new_a_insert, new_b_insert, start=changed_start,
                                               x_in=cache[changed_start])
                if mse_insert < current_mse:
                    current_mse = mse_insert
                    current_a = new_a_insert
                    current_b = new_b_insert
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


def hill_climb_swap_order(current_a, current_b,
                          max_iterations=2000, stagnation_limit=100,
                          log_prefix="[res_relu] "):
    """Swap paired block positions to refine global order."""
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
            for j in range(i + 1, n):
                new_a = swap_item(current_a, i, j)
                new_b = swap_item(current_b, i, j)
                changed_start = i
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
            print(f"{log_prefix}swap-order iter {iters}: MSE={current_mse:.6f}", flush=True)
    return current_mse, current_a, current_b


def hill_climb_b_repair(current_a, current_b,
                        max_iterations=2000, stagnation_limit=100,
                        log_prefix="[res_relu] "):
    """Fix wrong A/B pairings by swapping only B assignments."""
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
            for j in range(i + 1, n):
                new_b = swap_item(current_b, i, j)
                changed_start = i
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

    print(f"{log_prefix}Pairing blocks using trace heuristic...")
    cost_matrix = build_pairing_cost_matrix(a_list, b_list)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    paired_a = [a_list[i] for i in row_ind]
    paired_b = [b_list[j] for j in col_ind]

    print(f"{log_prefix}Running Gumbel-Sinkhorn warm start...")
    routed_mse, a_ord, b_ord = train_soft_permutation(paired_a, paired_b)
    print(f"{log_prefix}Post-routing hard MSE: {routed_mse:.6f}", flush=True)

    if routed_mse < 1e-12:
        perm = build_permutation(a_ord, b_ord)
        return routed_mse, perm, time.perf_counter() - t0

    best_mse = routed_mse
    while True:
        improved = False

        b_mse, b_ord = hill_climb_b_repair(a_ord, b_ord, log_prefix=log_prefix)
        if b_mse < best_mse:
            best_mse = b_mse
            improved = True
            print(f"{log_prefix}cascade improved via b-repair: MSE={best_mse:.6f}", flush=True)
            if best_mse < 1e-12:
                break

        swap_mse, a_ord, b_ord = hill_climb_swap_order(a_ord, b_ord, log_prefix=log_prefix)
        if swap_mse < best_mse:
            best_mse = swap_mse
            improved = True
            print(f"{log_prefix}cascade improved via swap-order: MSE={best_mse:.6f}", flush=True)
            if best_mse < 1e-12:
                break

        insert_mse, a_ord, b_ord = hill_climb_insert(a_ord, b_ord, log_prefix=log_prefix)
        if insert_mse < best_mse:
            best_mse = insert_mse
            improved = True
            print(f"{log_prefix}cascade improved via insert: MSE={best_mse:.6f}", flush=True)
            if best_mse < 1e-12:
                break

        if not improved:
            print(f"{log_prefix}cascade converged with no further improvements.", flush=True)
            break

    perm = build_permutation(a_ord, b_ord)
    return best_mse, perm, time.perf_counter() - t0


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
