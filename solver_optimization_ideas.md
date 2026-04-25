# Solver Optimization Ideas

## Why the Current Solver is Slow

The hill climbing approach re-evaluates the **full model** for every candidate swap:

- 48 blocks = ~1,100 possible swap pairs
- Each evaluation = 48 matrix multiplications + activations across the entire dataset
- Up to 50,000 iterations × 7 block functions = potentially millions of full forward passes

With a 5MB+ dataset on CPU, this is inherently expensive.

---

## Potential Speedups

### 1. Early Bailout / Reduce Iterations
- Lower `max_iterations` from 50,000 to something like 5,000–10,000
- Track the best MSE over time and abort if no improvement after N consecutive iterations
- Most gains happen early; later iterations have diminishing returns

### 2. Vectorized / Batched Evaluation
Instead of evaluating one permutation at a time, batch multiple permutations:
- Build a batch of candidate sequences
- Run them through the model in parallel using batched matrix ops
- PyTorch can evaluate dozens of permutations simultaneously on CPU, or hundreds on GPU

### 3. GPU Acceleration
- Move tensors to `cuda` if available
- A single forward pass is small, but millions of them add up — GPU parallelism helps massively with batched evaluation
- Even an old GPU could be 10–50× faster for this workload

### 4. Cache Intermediate Activations
- Swapping two adjacent blocks only affects outputs from that point onward
- Cache layer outputs and only recompute the tail of the network after a swap
- This turns O(48) work into O(1–2) work per evaluation
- More complex to implement but potentially 10–100× speedup

### 5. Smarter Search Algorithms
- **Simulated annealing**: escape local minima without exhaustive swap search
- **Genetic algorithms**: maintain a population of permutations, crossover + mutate
- **Beam search**: keep top-K candidates at each step instead of one greedy path
- These often find better solutions with fewer evaluations than naive hill climbing

### 6. Reduce Swap Search Space
- Instead of trying all ~1,100 swaps, only try swaps involving the most recently moved block
- Or use a neighborhood heuristic (e.g., only swap blocks with low between-block affinity)
- Dramatically cuts the number of evaluations per iteration

### 7. Parallel Hill Climbing
- The greedy initialization already tries 48 different starting points — these are embarrassingly parallel
- Run multiple hill climbing instances across CPU cores and take the best result
- Python `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`

### 8. Mixed-Precision / Compiled Evaluation
- Use `torch.compile()` (PyTorch 2.0+) to JIT-compile the evaluate function
- Use `float16` or `bfloat16` instead of `float32` if MSE tolerance allows

---

## Quick Wins (Minimal Code Change)

1. Add a progress print every 1,000 iterations so you know it hasn't hung
2. Track `best_mse` globally across all block functions and skip functions once a solution is found
3. Run the Hungarian pairing MSE test *before* the expensive hill climbing — if it's already exact, skip the rest
4. Time each block function and abort slow ones early if a good enough solution is already found

---

## If This Needs to Run Regularly

Consider rewriting the evaluation in a compiled language (Rust/C++) or using JAX/Numba for JIT compilation. The core operation is repeated small matrix multiplications — exactly where compiled/JIT code shines over Python loops.
