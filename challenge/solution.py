import stim
import numpy as np
from ldpc import bposd_decoder
import scipy.sparse as sp
from ldpc.mod2 import nullspace
from ldpc.mod2 import rank
import matplotlib.pyplot as plt

def ldpc_x_memory(H_matrix, logical_support_indices, p):
    num_checks, num_data = H_matrix.shape
    data_qubits = list(range(num_data))
    ancilla_qubits = list(range(num_data, num_data + num_checks))
    
    c = stim.Circuit()

    c.append("R", data_qubits + ancilla_qubits)

    c.append("X_ERROR", data_qubits, p)

    for check_idx in range(num_checks):
        ancilla = ancilla_qubits[check_idx]
        targets = np.where(H_matrix[check_idx] == 1)[0]
        
        c.append("H", ancilla)
        for q in targets:
            c.append("CZ", [ancilla, q])
        c.append("H", ancilla)
        c.append("M", ancilla)
        
        c.append("DETECTOR", [stim.target_rec(-1)], [check_idx])

    c.append("M", data_qubits)
    
    
    rec_targets = []
    for q_idx in logical_support_indices:

        relative_index = q_idx - num_data 
        rec_targets.append(stim.target_rec(relative_index))
        
    c.append("OBSERVABLE_INCLUDE", rec_targets, 0)
    
    return c

def shapes_to_parity_matrix(width, height, row_shapes):
    num_qubits = width * height
    stabilizer_rows = []

    # Iterate through every possible "root" position (r, c) on the lattice
    for r in range(height):
        current_shape = row_shapes[r % len(row_shapes)]
        
        # Find how "tall" the shape is.
        max_dr = max(dr for dr, dc in current_shape)
        
        # If the shape hangs off the bottom of the lattice, we don't place it.
        # (This creates the "logical degrees of freedom" at the boundaries)
        if r + max_dr >= height:
            continue
            
        for c in range(width):
            # Create a new row for the H-matrix (initially all zeros)
            h_row = np.zeros(num_qubits, dtype=int)
            
            # Apply the shape offsets to find involved qubits
            for dr, dc in current_shape:
                target_r = r + dr
                # Wrap column index (Periodic Boundary Condition)
                target_c = (c + dc) % width 
                
                # Map 2D (r, c) -> 1D index
                qubit_idx = target_r * width + target_c
                
                h_row[qubit_idx] = 1
            
            stabilizer_rows.append(h_row)

    return np.array(stabilizer_rows)

def decode_with_bposd(syndrome, H_matrix, error_rate=0.01, max_iter=30, osd_order=10):
    bpd = bposd_decoder(
        H_matrix,
        error_rate = float(error_rate),
        max_iter=max_iter,
        bp_method='ms',          # Min-Sum is commonly used for stability
        ms_scaling_factor=0.625, # Standard scaling factor for Min-Sum
        osd_method='osd_cs',     # OSD with Combination Sweep
        osd_order=osd_order      # Controls the search depth in OSD
    )
    

    predicted_error = bpd.decode(syndrome)
    
    return predicted_error


import numpy as np
import stim

def calculate_logical_error_rate(circuit, H_matrix, logical_support, num_shots=1000, p_error=0.01):
    sampler = circuit.compile_detector_sampler()
    syndromes_batch, actual_observables_batch = sampler.sample(
        shots=num_shots, 
        separate_observables=True
    )
    
    num_logical_errors = 0
    
    print(f"Running decoding for {num_shots} shots...")

    for i in range(num_shots):
        syndrome = syndromes_batch[i].astype(int)
        
        actual_flip = actual_observables_batch[i][0]
        
        correction = decode_with_bposd(syndrome, H_matrix, error_rate=p_error)
        

        predicted_flip = np.sum(correction[logical_support]) % 2
        
        if predicted_flip != actual_flip:
            num_logical_errors += 1

    ler = num_logical_errors / num_shots
    return ler


L = 50
H = 8   

shapes = [
    [ (0, 0),
    (2, -1),
    (2, 0),   
    (2, 1)], 
    [ (0, 0),
    (2, -1),
    (2, 0),   
    (2, 1)], 
    [ (0, 0),
    (2, -1),
    (1, 1),   
    (2, 1)],
    [ (0, 0),
    (2, -1),
    (1, 1),   
    (2, 1)], 
    [ (0, 0),
    (2, -1),
    (1, -1),   
    (1, 1)],
    [ (0, 0),
    (2, -1),
    (1, 1),   
    (2, 1)],
    ]

H_matrix = shapes_to_parity_matrix(width=L, height=H, row_shapes=shapes)

print(f"Lattice: {H}x{L} = {H*L} physical qubits")
print(f"Matrix Shape: {H_matrix.shape}")
print(f"Number of Stabilizers: {H_matrix.shape[0]}")
Hrank = rank(H_matrix)
k = 2*L
print(f"Logical Qubits (k): {k}")

physical_errors = np.logspace(-3.5, -0.8, 15).astype(float)

logical_errors = []
for p in physical_errors:
    logical_ops = [r * L for r in range(H)]

    circuit = ldpc_x_memory(H_matrix, logical_ops, p)
    #print(repr(circuit))
    sampler = circuit.compile_detector_sampler()
    syndrome_batch = sampler.sample(shots=1)
    single_syndrome = syndrome_batch[0].astype(int)

    correction = decode_with_bposd(single_syndrome, H_matrix, error_rate=p)

    #print(f"Syndrome: {single_syndrome}")
    #print(f"Predicted Correction: {correction}")

    logical_error_rate = calculate_logical_error_rate(
        circuit=circuit,
        H_matrix=H_matrix,
        logical_support=logical_ops,
        num_shots=100000,
        p_error=p
    )
    logical_errors.append(logical_error_rate)

print(logical_errors)
plt.figure(figsize=(8, 6))
plt.loglog(physical_errors, logical_errors, '-o', label=f'CA Code (H={H}, L={L})')
plt.loglog(physical_errors, physical_errors, '--', color='gray', label='Breakeven (y=x)')
plt.xlabel("Physical Error Rate (p)")
plt.ylabel("Logical Error Rate (LER)")
plt.title("Bit-Flip Error Correction Performance")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig("test.png")