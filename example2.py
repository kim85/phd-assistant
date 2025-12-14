import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Set backend to 'Agg' to prevent "FigureCanvasAgg" errors in headless environments
import matplotlib
matplotlib.use('Agg')

# --- Utility functions ---
def get_zero_matrix(rows, cols):
    return np.zeros((rows, cols))

def get_identity_matrix(dim):
    return np.eye(dim)

# --- Problem Setup ---
n = 2
m = 2
l = 2

# Given concrete matrices
A = np.array([[1, 3], [2, 4]])
b = np.array([[5], [6]])
F = np.array([[1, 0], [0, 1]])
D_zero = get_zero_matrix(l, m)
E_A = np.array([[0, 1], [0, 0]])
E_b = np.array([[0], [1]])

# --- CVXPY Variables ---
x = cp.Variable((n, 1))        
Lambda = cp.Variable(pos=True) # Scalar
nu = cp.Variable(pos=True)     # Scalar

# --- Objective Function ---
objective = cp.Minimize(nu)

# --- Data Storage ---
rho_values = np.arange(0, 2.1, 0.1)
gamma_results = []
x_trajectory = []

print("Solving SDP for different rho values...")

for rho in rho_values:
    # M depends on rho
    M = rho * get_identity_matrix(l)

    # Helper matrices
    I_n = get_identity_matrix(n)
    I_m = get_identity_matrix(m)
    I_l = get_identity_matrix(l)

    O_nl = get_zero_matrix(n, l)
    O_m1 = get_zero_matrix(m, 1)
    O_1m = get_zero_matrix(1, m)
    O_ln = get_zero_matrix(l, n)

    # --- Construct Blocks ---
    
    # Block (1,1)
    block11 = -I_n 

    # Block (1,2)
    block12 = F * Lambda 

    # Block (1,3)
    block13 = A @ x - b 
    
    # Block (1,4)
    block14 = O_nl 

    # Block (2,1)
    block21 = Lambda * F.T 

    # Block (2,2)
    block22 = -Lambda * I_m 

    # Block (2,3)
    block23 = O_m1 

    # Block (2,4)
    # Note: D_zero is (l, m), M is (l, l). D.T @ M.T is (m, l).
    block24 = Lambda * (D_zero.T @ M.T) 

    # Block (3,1)
    block31 = (A @ x - b).T 

    # Block (3,2)
    block32 = O_1m 

    # Block (3,3) - FIXED: Added order='C' to suppress FutureWarning
    block33 = -cp.reshape(nu, (1, 1), order='C')

    # Block (3,4)
    block34 = (E_A @ x - E_b).T @ M.T

    # Block (4,1)
    block41 = O_ln 

    # Block (4,2)
    block42 = M @ D_zero * Lambda 

    # Block (4,3)
    block43 = M @ (E_A @ x - E_b)

    # Block (4,4)
    block44 = -Lambda * I_l 

    # Assemble the full block matrix
    LMI_matrix = cp.bmat([
        [block11, block12, block13, block14],
        [block21, block22, block23, block24],
        [block31, block32, block33, block34],
        [block41, block42, block43, block44]
    ])

    # --- Constraints ---
    constraints = [
        LMI_matrix << 0 
    ]

    # --- Solve ---
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.SCS, verbose=False)

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            gamma_results.append(np.sqrt(nu.value))
            x_trajectory.append(x.value.flatten()) 
        else:
            print(f"rho={rho:.1f}: Status {problem.status}")
            gamma_results.append(np.nan)
            x_trajectory.append(np.array([np.nan, np.nan]))
    except Exception as e:
        print(f"rho={rho:.1f}: Error {e}")
        gamma_results.append(np.nan)
        x_trajectory.append(np.array([np.nan, np.nan]))

# --- Plotting ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(rho_values, gamma_results, marker='o', linestyle='-', color='blue')
plt.title(r'Minimal $\gamma = \sqrt{\nu}$ vs $\rho$')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\gamma$')
plt.grid(True)

plt.subplot(1, 2, 2)
x1_coords = [p[0] for p in x_trajectory]
x2_coords = [p[1] for p in x_trajectory]
plt.plot(x1_coords, x2_coords, marker='x', linestyle='--', color='red')
plt.title(r'Trajectory of $x=(x_1, x_2)$ vs $\rho$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.grid(True)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save the plot
filename = 'plot.png'
plt.savefig(filename)
print(f"\nProcessing complete. Plot saved to '{filename}'.")

# Removed plt.show() to avoid UserWarning in non-interactive environments