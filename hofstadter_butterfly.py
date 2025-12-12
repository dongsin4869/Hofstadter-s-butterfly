from math import gcd

import matplotlib.pyplot as plt
import numpy as np


def get_hofstadter_eigenvalues(p, q):
    """
    Constructs the Harper-Hofstadter Hamiltonian matrix for a given flux p/q
    and returns its eigenvalues.
    """
    # 1. Initialize a q x q matrix with zeros
    # The matrix size is determined by the denominator q
    H = np.zeros((q, q))

    # 2. Fill the Hamiltonian matrix
    for n in range(q):
        # Diagonal elements: On-site energy term
        # E = 2 * cos(2 * pi * alpha * n)
        H[n, n] = 2 * np.cos(2 * np.pi * (p / q) * n)

        # Off-diagonal elements: Hopping terms (nearest neighbor interaction)
        # We set the hopping amplitude t = 1
        if n < q - 1:
            H[n, n + 1] = 1
            H[n + 1, n] = 1

    # 3. Apply Periodic Boundary Conditions (PBC)
    # This connects the last site back to the first site (wrapping around)
    H[0, q - 1] = 1
    H[q - 1, 0] = 1

    # 4. Calculate Eigenvalues
    # We use eigh because the matrix is Hermitian (symmetric)
    eigenvalues = np.linalg.eigvalsh(H)

    return eigenvalues


# --- Main Data Generation Loop ---

# Setup lists to hold our plotting data
x_data = []  # Will hold the Flux (p/q)
y_data = []  # Will hold the Energy Eigenvalues

# Resolution: higher max_q = more detailed fractal, but slower calculation.
max_q = 150

print(f"Calculating Hofstadter Butterfly for denominators up to q={max_q}...")

# Iterate through denominators (q)
for q in range(1, max_q + 1):
    # Iterate through numerators (p)
    for p in range(1, q + 1):
        # Optimization: Only calculate for irreducible fractions.
        # e.g., 2/4 is the same physical system as 1/2.
        if gcd(p, q) == 1:
            # Get the energies for this specific magnetic flux
            energies = get_hofstadter_eigenvalues(p, q)

            # Store data for plotting
            # We append the flux (p/q) for *every* energy eigenvalue found
            for energy in energies:
                x_data.append(p / q)
                y_data.append(energy)

print("Calculation complete. Generating plot...")

# --- Visualization ---

plt.figure()
plt.xlabel("Energy (E/t)")
plt.ylabel("Magnetic Flux ($\\Phi / \\Phi_0$)")
plt.xlim(-4, 4)
plt.ylim(0, 1)

# Use a scatter plot with small markers (s) to see the fractal structure
plt.scatter(y_data, x_data, s=0.5, c="black", alpha=0.6)
plt.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
plt.axhline(1 / 3, color="red", linestyle="--", linewidth=0.8)
plt.axhline(1 / 4, color="red", linestyle="--", linewidth=0.8)

# plt.grid(True, alpha=0.3)
plt.savefig("hofstadter_butterfly.png")
