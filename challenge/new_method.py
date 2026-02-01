import stim
import sinter
import numpy as np
import networkx as nx
from ldpc import bposd_decoder
import scipy.sparse as sp
from ldpc.mod2 import nullspace
from ldpc.mod2 import rank
import matplotlib.pyplot as plt

# d=4 cat-repetition code benchmark from section 5.1
D4_P = [0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.011,
        0.014, 0.018, 0.024, 0.031, 0.041, 0.053, 0.069, 0.090, 0.118, 0.153, 0.200]
D4_LER = [0.000010, 0.000000, 0.000000, 0.000040, 0.000030, 0.000020, 0.000060, 0.000060, 0.000190, 0.000300,
          0.000580, 0.000970, 0.001510, 0.002850, 0.004830, 0.007810, 0.013590, 0.023400, 0.038450, 0.064570, 0.105680]

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


class TimeMultiplexedGenerator:
    """
    Compiler that converts a raw H-matrix into a time-multiplexed
    stim circuit to minimize ancilla count.
    """

    def __init__(self, H_matrix, logical_support, p_error):
        self.H = H_matrix
        self.logical_support = logical_support
        self.p = p_error
        self.num_checks, self.num_data = H_matrix.shape

        # Architecture State
        self.schedule = None       # List of [check_indices] per time slice
        self.ancilla_map = {}      # check_idx -> physical_ancilla_id
        self.total_qubits = 0
        self.G = None              # Conflict graph

    def compile(self):
        """Full compilation pipeline."""
        self._generate_conflict_graph()
        self._optimize_schedule()
        self._map_physical_resources()
        return self._build_stim_circuit()

    def _generate_conflict_graph(self):
        """
        Build a networkx graph where nodes are check indices and edges
        connect checks that share data qubits (cannot be measured simultaneously
        on same ancilla).
        """
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_checks))

        # Two checks conflict if they share any data qubit
        for i in range(self.num_checks):
            for j in range(i + 1, self.num_checks):
                # Check if rows share any non-zero position
                if np.dot(self.H[i], self.H[j]) > 0:
                    self.G.add_edge(i, j)

    def _optimize_schedule(self):
        """
        Use graph coloring to assign checks to time slices.
        Checks with the same color can share ancillas.
        """
        # Use greedy coloring with largest_first strategy
        coloring = nx.coloring.greedy_color(self.G, strategy='largest_first')

        # Group checks by color (time slice)
        num_colors = max(coloring.values()) + 1 if coloring else 1
        self.schedule = [[] for _ in range(num_colors)]

        for check_idx, color in coloring.items():
            self.schedule[color].append(check_idx)

        # Sort each time slice for deterministic ordering
        for time_slice in self.schedule:
            time_slice.sort()

    def _map_physical_resources(self):
        """
        Assign physical ancilla IDs based on the coloring.
        Checks in different time slices can share the same physical ancilla.
        """
        # Find the maximum number of checks in any time slice
        max_checks_per_slice = max(len(ts) for ts in self.schedule) if self.schedule else 0

        # Assign physical ancilla IDs
        # Checks at the same position in different time slices share an ancilla
        for time_idx, time_slice in enumerate(self.schedule):
            for pos, check_idx in enumerate(time_slice):
                # Ancilla ID is based on position within time slice
                self.ancilla_map[check_idx] = pos

        # Total number of physical ancillas needed
        self.num_physical_ancillas = max_checks_per_slice
        self.total_qubits = self.num_data + self.num_physical_ancillas

    def _build_stim_circuit(self):
        """
        Generate the stim circuit with time-multiplexed ancilla usage.
        """
        c = stim.Circuit()

        data_qubits = list(range(self.num_data))
        ancilla_start = self.num_data
        ancilla_qubits = list(range(ancilla_start, ancilla_start + self.num_physical_ancillas))

        # Reset all qubits
        c.append("R", data_qubits + ancilla_qubits)

        # Apply X_ERROR on data qubits
        c.append("X_ERROR", data_qubits, self.p)

        # Track measurement index for DETECTOR references
        measurement_count = 0
        check_to_measurement = {}  # Maps check_idx to its measurement record index

        # Process each time slice
        for time_idx, time_slice in enumerate(self.schedule):
            if not time_slice:
                continue

            # Get the physical ancillas used in this time slice
            active_ancillas = set()
            for check_idx in time_slice:
                phys_ancilla = ancilla_start + self.ancilla_map[check_idx]
                active_ancillas.add(phys_ancilla)

            active_ancillas = sorted(active_ancillas)

            # Reset active ancillas for this time slice
            c.append("R", active_ancillas)

            # Apply H gates on ancillas
            c.append("H", active_ancillas)

            # Apply CZ gates for each check's data qubits
            for check_idx in time_slice:
                phys_ancilla = ancilla_start + self.ancilla_map[check_idx]
                targets = np.where(self.H[check_idx] == 1)[0].tolist()

                for data_q in targets:
                    c.append("CZ", [phys_ancilla, data_q])

            # Apply H gates on ancillas
            c.append("H", active_ancillas)

            # Measure ancillas
            c.append("M", active_ancillas)

            # Record measurement indices for each check
            # Measurements are recorded in order of active_ancillas
            ancilla_to_meas_offset = {a: i for i, a in enumerate(active_ancillas)}

            for check_idx in time_slice:
                phys_ancilla = ancilla_start + self.ancilla_map[check_idx]
                meas_offset = ancilla_to_meas_offset[phys_ancilla]
                # The measurement for this check is at position:
                # measurement_count + meas_offset (from the start)
                check_to_measurement[check_idx] = measurement_count + meas_offset

            measurement_count += len(active_ancillas)

        # Add DETECTOR for each check (in order of check index for consistency)
        for check_idx in range(self.num_checks):
            meas_idx = check_to_measurement[check_idx]
            # Calculate the relative index from the current position
            relative_idx = meas_idx - measurement_count
            c.append("DETECTOR", [stim.target_rec(relative_idx)], [check_idx])

        # Measure all data qubits
        c.append("M", data_qubits)

        # Add OBSERVABLE_INCLUDE for logical operator
        rec_targets = []
        for q_idx in self.logical_support:
            # The data qubit measurements start after all syndrome measurements
            # q_idx is the index within data_qubits (0 to num_data-1)
            # It's measured at position: measurement_count + q_idx
            # From the end, it's at: -(num_data - q_idx)
            relative_index = q_idx - self.num_data
            rec_targets.append(stim.target_rec(relative_index))

        c.append("OBSERVABLE_INCLUDE", rec_targets, 0)

        return c


def generate_optimized_task(H_matrix, logical_ops, p):
    """Factory function for creating optimized sinter.Task."""
    compiler = TimeMultiplexedGenerator(H_matrix, logical_ops, p)
    circuit = compiler.compile()
    return sinter.Task(
        circuit=circuit,
        decoder=f'bposd_p{p:.6f}',
        json_metadata={'p': float(p), 'impl': 'multiplexed'}
    )


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


class BPOSDCompiledDecoder(sinter.CompiledDecoder):
    """Compiled decoder for a specific H_matrix configuration."""

    def __init__(self, H_matrix, logical_support, error_rate, num_detectors, num_observables, max_iter=30, osd_order=10):
        self.H_matrix = H_matrix
        self.logical_support = logical_support
        self.error_rate = error_rate
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.max_iter = max_iter
        self.osd_order = osd_order

        # Pre-create the bposd_decoder instance for reuse
        self.bpd = bposd_decoder(
            H_matrix,
            error_rate=float(error_rate),
            max_iter=max_iter,
            bp_method='ms',
            ms_scaling_factor=0.625,
            osd_method='osd_cs',
            osd_order=osd_order
        )

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        num_shots = bit_packed_detection_event_data.shape[0]
        num_det_bytes = (self.num_detectors + 7) // 8

        # Output: one bit per observable, bit-packed
        num_obs_bytes = (self.num_observables + 7) // 8
        predictions = np.zeros((num_shots, num_obs_bytes), dtype=np.uint8)

        for shot_idx in range(num_shots):
            # Unpack the detection events for this shot
            packed_row = bit_packed_detection_event_data[shot_idx, :num_det_bytes]
            syndrome_bits = np.unpackbits(packed_row, bitorder='little')[:self.num_detectors]
            syndrome = syndrome_bits.astype(int)

            # Decode using BPOSD
            correction = self.bpd.decode(syndrome)

            # Compute logical flip
            predicted_flip = np.sum(correction[self.logical_support]) % 2

            # Pack the prediction (single observable)
            if predicted_flip:
                predictions[shot_idx, 0] = 1

        return predictions


class CustomBPOSDDecoder(sinter.Decoder):
    """Sinter decoder wrapper for ldpc bposd_decoder."""

    def __init__(self, H_matrix, logical_support, error_rate, max_iter=30, osd_order=10):
        self.H_matrix = H_matrix
        self.logical_support = logical_support
        self.error_rate = error_rate
        self.max_iter = max_iter
        self.osd_order = osd_order

    def compile_decoder_for_dem(self, *, dem):
        num_detectors = dem.num_detectors
        num_observables = dem.num_observables
        return BPOSDCompiledDecoder(
            H_matrix=self.H_matrix,
            logical_support=self.logical_support,
            error_rate=self.error_rate,
            num_detectors=num_detectors,
            num_observables=num_observables,
            max_iter=self.max_iter,
            osd_order=self.osd_order
        )


def generate_tasks(H_matrix, logical_ops, physical_errors):
    """Generate sinter.Task for each physical error rate."""
    for p in physical_errors:
        circuit = ldpc_x_memory(H_matrix, logical_ops, float(p))
        yield sinter.Task(
            circuit=circuit,
            decoder=f'bposd_p{p:.6f}',
            json_metadata={'p': float(p)},
        )


def create_custom_decoders(H_matrix, logical_support, physical_errors):
    """Create dict mapping decoder names to instances."""
    return {
        f'bposd_p{p:.6f}': CustomBPOSDDecoder(H_matrix, logical_support, float(p))
        for p in physical_errors
    }


def extract_results(results):
    """Extract (physical_errors, logical_errors) from sinter.TaskStats list."""
    sorted_results = sorted(results, key=lambda r: r.json_metadata['p'])
    physical_errors = [r.json_metadata['p'] for r in sorted_results]
    logical_errors = [r.errors / r.shots if r.shots > 0 else 0.0 for r in sorted_results]
    return physical_errors, logical_errors


def visualize_optimization_comparison(H_matrix, logical_ops, p=0.01):
    """
    Create a 3-panel visualization comparing standard vs time-multiplexed approaches.

    Panel 1: Stacked bar chart of qubit counts
    Panel 2: Reduction metrics bar chart
    Panel 3: Circuit structure diagram
    """
    from matplotlib.patches import FancyBboxPatch

    # Compile circuit to get actual stats
    compiler = TimeMultiplexedGenerator(H_matrix, logical_ops, p)
    compiler.compile()

    # Extract metrics
    num_data = compiler.num_data
    num_checks = compiler.num_checks
    optimized_ancillas = compiler.num_physical_ancillas
    num_time_slices = len(compiler.schedule)

    # Standard approach: one ancilla per check
    standard_ancillas = num_checks
    standard_total = num_data + standard_ancillas

    # Optimized approach
    optimized_total = compiler.total_qubits

    # Calculate reductions
    ancilla_reduction = 100 * (1 - optimized_ancillas / standard_ancillas)
    total_reduction = 100 * (1 - optimized_total / standard_total)

    # Print summary table
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Standard':<12} {'Optimized':<12}")
    print("-" * 60)
    print(f"{'Data qubits':<30} {num_data:<12} {num_data:<12}")
    print(f"{'Ancilla qubits':<30} {standard_ancillas:<12} {optimized_ancillas:<12}")
    print(f"{'Total qubits':<30} {standard_total:<12} {optimized_total:<12}")
    print(f"{'Time slices':<30} {'1':<12} {num_time_slices:<12}")
    print("-" * 60)
    print(f"{'Ancilla reduction':<30} {'':<12} {ancilla_reduction:.1f}%")
    print(f"{'Total qubit reduction':<30} {'':<12} {total_reduction:.1f}%")
    print("=" * 60 + "\n")

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Stacked Bar Chart of Qubit Counts
    ax1 = axes[0]
    labels = ['Standard', 'Time-Multiplexed']
    data_counts = [num_data, num_data]
    ancilla_counts = [standard_ancillas, optimized_ancillas]

    x = np.arange(len(labels))
    width = 0.5

    bars1 = ax1.bar(x, data_counts, width, label='Data Qubits', color='#2ecc71')
    bars2 = ax1.bar(x, ancilla_counts, width, bottom=data_counts, label='Ancilla Qubits', color='#e74c3c')

    ax1.set_ylabel('Number of Qubits')
    ax1.set_title('Total Qubit Count Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Add total labels on bars
    for i, (d, a) in enumerate(zip(data_counts, ancilla_counts)):
        ax1.annotate(f'{d + a}', xy=(i, d + a + 5), ha='center', fontweight='bold')

    # Panel 2: Reduction Metrics
    ax2 = axes[1]
    metrics = ['Ancilla\nReduction', 'Total Qubit\nReduction']
    values = [ancilla_reduction, total_reduction]
    colors = ['#3498db', '#9b59b6']

    bars = ax2.bar(metrics, values, color=colors, width=0.6)
    ax2.set_ylabel('Reduction (%)')
    ax2.set_title('Optimization Impact')
    ax2.set_ylim(0, 100)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val + 2),
                     ha='center', fontweight='bold')

    # Add time slice annotation
    ax2.annotate(f'Using {num_time_slices} time slices', xy=(0.5, 0.95),
                 xycoords='axes fraction', ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Circuit Structure Diagram
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Circuit Structure Comparison')

    # Standard approach - single large block
    standard_box = FancyBboxPatch((0.5, 5.5), 4, 3.5,
                                   boxstyle="round,pad=0.05,rounding_size=0.2",
                                   facecolor='#ffcccc', edgecolor='#e74c3c', linewidth=2)
    ax3.add_patch(standard_box)
    ax3.text(2.5, 7.25, 'Standard\nApproach', ha='center', va='center', fontsize=10, fontweight='bold')
    ax3.text(2.5, 6.2, f'{standard_ancillas} ancillas\n(parallel)', ha='center', va='center', fontsize=9)

    # Optimized approach - multiple time slices
    slice_height = 3.5 / min(num_time_slices, 5)  # Cap visual slices at 5
    display_slices = min(num_time_slices, 5)

    for i in range(display_slices):
        y_pos = 5.5 + i * slice_height
        slice_box = FancyBboxPatch((5.5, y_pos), 4, slice_height * 0.85,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor='#ccffcc', edgecolor='#2ecc71', linewidth=1.5)
        ax3.add_patch(slice_box)
        if i == display_slices // 2:
            ax3.text(7.5, y_pos + slice_height * 0.4, f't={i+1}', ha='center', va='center', fontsize=8)

    ax3.text(7.5, 4.8, 'Time-Multiplexed', ha='center', va='center', fontsize=10, fontweight='bold')
    ax3.text(7.5, 4.2, f'{optimized_ancillas} ancillas\n({num_time_slices} time slices)',
             ha='center', va='center', fontsize=9)

    # Add arrows showing time progression
    ax3.annotate('', xy=(9.8, 9), xytext=(9.8, 5.5),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax3.text(9.9, 7.25, 'time', ha='left', va='center', fontsize=8, color='gray', rotation=90)

    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to optimization_comparison.png")

    return fig


def calculate_logical_error_rate(circuit, H_matrix, logical_support, num_shots=1000, p_error=0.01):
    """Legacy function for single-threaded decoding."""
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


if __name__ == "__main__":
    import multiprocessing

    L = 50
    H = 8

    shapes = [
        [(0, 0),
         (2, -1),
         (2, 0),
         (2, 1)],
        [(0, 0),
         (2, -1),
         (2, 0),
         (2, 1)],
        [(0, 0),
         (2, -1),
         (1, 1),
         (2, 1)],
        [(0, 0),
         (2, -1),
         (1, 1),
         (2, 1)],
        [(0, 0),
         (2, -1),
         (1, -1),
         (1, 1)],
        [(0, 0),
         (2, -1),
         (1, 1),
         (2, 1)],
    ]

    H_matrix = shapes_to_parity_matrix(width=L, height=H, row_shapes=shapes)

    print(f"Lattice: {H}x{L} = {H*L} physical qubits")
    print(f"Matrix Shape: {H_matrix.shape}")
    print(f"Number of Stabilizers: {H_matrix.shape[0]}")
    Hrank = rank(H_matrix)
    k = 2 * L
    print(f"Logical Qubits (k): {k}")

    logical_ops = [r * L for r in range(H)]
    physical_errors = np.logspace(-3, np.log10(0.15), 21).astype(float)

    NUM_WORKERS = max(4, multiprocessing.cpu_count())
    NUM_SHOTS = 1_000_000

    print(f"Using {NUM_WORKERS} workers for {NUM_SHOTS} shots per error rate")

    # Use TimeMultiplexedGenerator for optimized ancilla usage
    print("\n--- Compiling Quantum Circuit ---")

    # Generate tasks using TimeMultiplexedGenerator
    tasks = []
    for i, p in enumerate(physical_errors):
        compiler = TimeMultiplexedGenerator(H_matrix, logical_ops, float(p))
        circuit = compiler.compile()
        tasks.append(
            sinter.Task(
                circuit=circuit,
                decoder=f'bposd_p{p:.6f}',
                json_metadata={'p': float(p)}
            )
        )

        # Print compilation stats for first error rate
        if i == 0:
            num_time_slices = len(compiler.schedule)
            original_ancillas = compiler.num_checks
            optimized_ancillas = compiler.num_physical_ancillas
            original_total = compiler.num_data + original_ancillas
            optimized_total = compiler.total_qubits

            print(f"Compressed {original_ancillas} checks into {optimized_ancillas} physical ancillas ({num_time_slices} time slices)")
            print(f"Lattice: {H}x{L} = {compiler.num_data} physical qubits")
            print(f"Total qubits: {optimized_total} (down from {original_total})")
            print(f"Ancilla reduction: {100 * (1 - optimized_ancillas / original_ancillas):.1f}%\n")

    # Generate comparison visualizations
    print("\n--- Generating Comparison Visualizations ---")
    visualize_optimization_comparison(H_matrix, logical_ops, p=0.01)

    custom_decoders = create_custom_decoders(H_matrix, logical_ops, physical_errors)

    results = sinter.collect(
        num_workers=NUM_WORKERS,
        tasks=tasks,
        custom_decoders=custom_decoders,
        max_shots=NUM_SHOTS,
        print_progress=True,
    )

    p_values, logical_errors = extract_results(results)

    print(logical_errors)
    plt.figure(figsize=(8, 6))
    plt.loglog(p_values, logical_errors, '-o', label=f'CA Code (H={H}, L={L})')
    plt.loglog(D4_P, D4_LER, '-s', label='Cat-Rep Code (d=4)')
    plt.loglog(p_values, p_values, '--', color='gray', label='Breakeven (y=x)')
    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Logical Error Rate (LER)")
    plt.title("Bit-Flip Error Correction Performance")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlim(1e-3, 1.5e-1)
    plt.savefig("new_test.png")
