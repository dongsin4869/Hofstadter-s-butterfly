from math import gcd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def get_chern_number(p, q, r):
    """
    Solves the Diophantine equation: r = s * p + t * q
    Returns the integer 's' (Chern number) which determines the gap's color.
    We look for the solution with the smallest magnitude |s|.
    """
    # Iterate possible values of s to find the one that satisfies the congruence
    # s * p = r (mod q)
    for s in range(-q, q + 1):
        if (s * p - r) % q == 0:
            # The physical Chern number is the minimal integer solution
            if abs(s) <= q / 2:
                return s
    return 0


# --- Parameters ---
# max_q determines resolution.
# 40 is fast (seconds). 80 is good for slides. 150+ for high-res print.
max_q = 150

fluxes = []
gap_min = []
gap_max = []
chern_numbers = []

print(f"Calculating Topological Phase Diagram (max_q={max_q})...")

# --- Main Calculation Loop ---
for q in range(1, max_q + 1):
    for p in range(1, q):
        if gcd(p, q) == 1:
            # 1. Construct Hamiltonian
            H = np.zeros((q, q))
            for n in range(q):
                H[n, n] = 2 * np.cos(2 * np.pi * (p / q) * n)
                if n < q - 1:
                    H[n, n + 1] = 1
                    H[n + 1, n] = 1
            H[0, q - 1] = 1
            H[q - 1, 0] = 1

            # 2. Get Eigenvalues
            evals = np.linalg.eigvalsh(H)

            # 3. Identify Gaps and Calculate Chern Numbers
            # If there are q bands, there are q-1 gaps between them.
            # Gap r is between band r-1 and band r.
            for r in range(1, q):
                e_bot = evals[r - 1]
                e_top = evals[r]
                gap_size = e_top - e_bot

                # Only color the gap if it is open (size > 0)
                if gap_size > 0.01:
                    s = get_chern_number(p, q, r)

                    fluxes.append(p / q)
                    gap_min.append(e_bot)
                    gap_max.append(e_top)
                    chern_numbers.append(s)

print("Calculation complete. Rendering image...")

# --- Visualization ---
plt.figure()

# We use LineCollection to plot thousands of vertical bars efficiently
lines = []
colors = []

# Prepare lines: each gap is a vertical line at x=p/q from y=min to y=max
for i in range(len(fluxes)):
    lines.append([(gap_min[i], fluxes[i]), (gap_max[i], fluxes[i])])
    colors.append(chern_numbers[i])

# Color Map Setup
import matplotlib.cm as cm
from matplotlib.colors import Normalize

cmap = cm.get_cmap("coolwarm")
# We normalize distinct integers to colors. Limit range to [-5, 5] for best contrast.
norm = Normalize(vmin=-5, vmax=5)

lc = LineCollection(lines, cmap=cmap, norm=norm, array=np.array(colors), linewidths=2)

plt.gca().add_collection(lc)

# Formatting
plt.ylabel("Magnetic Flux $(\\Phi/\\Phi_0)$")
plt.xlabel("Energy $(E/t)$")
plt.colorbar(lc, label="Chern Number ($s$)", extend="both")

plt.ylim(0, 1)
plt.xlim(-4, 4)

plt.savefig("topological_phase_diagram.png")
