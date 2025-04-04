# Optimization Results Table
![image](https://github.com/user-attachments/assets/83649612-90b0-49a4-8c75-20cc0e100c5e)

Step 1 (Basic)
•	This is the unoptimized version.
•	Execution Time: 63.9325 ms
•	Memory Bandwidth: 0.000257 GB/s
•	No speedups here because it’s the baseline.

Step 2 (Tiled kernel)
•	A tiling optimization is applied (to make memory accesses more efficient).
•	But surprisingly, execution time increased to 188.1375 ms, which is worse.
•	Memory Bandwidth improved to 0.027046 GB/s, meaning it reads memory faster, but it might have introduced other overheads.
•	Step speedup = 0.3398 (worse than before).
•	Cumulative speedup = 0.3398 (still worse than the original).

Step 3 (Tiled + Thread Coarsening)
•	Added thread coarsening (each thread does more work).
•	Execution time improved to 135.5840 ms (better than step 2 but worse than step 1).
•	Memory Bandwidth dropped.
•	Step speedup = 1.3876 (faster than step 2).
•	Cumulative speedup = 0.47153 (still not as fast as basic).

Step 4 (Register tiling + Thread coarsening)
•	Added register tiling (uses faster register memory).
•	Execution time is now 91.1732 ms — best among optimized versions.
•	Memory Bandwidth improved to 0.000512 GB/s.
•	Step speedup = 1.48710 (better than step 3).
•	Cumulative speedup = 0.70122 (still slower than the basic version but improving).

Conclusion of observations:
•	Optimizations don’t always guarantee better performance.
•	While memory bandwidth increased with optimizations, execution time didn’t always benefit.
•	The original “basic” kernel actually performed fastest.
•	However, the final version (step 4) is starting to catch up in performance.

