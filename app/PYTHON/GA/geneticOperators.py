import json
import os
import shutil
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
from PYTHON.tools import helper
from PYTHON.tools import new2dAlterations as alterations
from PYTHON.tools.new2dAlterations import (
    analyze_path_collisions,
    is_layout_collision_free,
    is_layout_phase2_solution,
    is_layout_phase1_ready,
    compute_layout_path_length_excess,
    compute_layout_score,
)
from tqdm.auto import tqdm

# GA optimization phases:
#   Phase 1 — electrode-only greedy; fitness and transition on electrode violations only.
#   Phase 2 — ordered trace-by-trace resolution; harsh crossing penalty + trace separation.
# Phase 1 → 2: >= PHASE2_READY_THRESHOLD individuals with zero electrode violations.
OPTIMIZATION_PHASE = 1
PHASE2_READY_THRESHOLD = 3
PHASE2_CLEARANCE_FREE_THRESHOLD = PHASE2_READY_THRESHOLD
PHASE2_CLEARANCE_PARENT_POOL_GENERATIONS = 3
PHASE2_GENERATION_COUNTER = 0

# Optional 2D layout plot after each individual (save PNG; interactive show if enabled).
PLOT_EACH_INDIVIDUAL_2D = False
PLOT_EACH_INDIVIDUAL_SHOW = False
PLOT_EACH_INDIVIDUAL_DIR = "data/output/plots/individuals"

# Parallel offspring evaluation: 0/1 = sequential, None = auto (all logical CPUs), N > 1 = worker count.
GA_PARALLEL_WORKERS = 0
_PARALLEL_WORKER_CTX = {}
_PARALLEL_EXECUTOR = None
_PARALLEL_EXECUTOR_WORKERS = 0


def configure_parallel_breeding(workers=0):
    """
    Enable process-pool breeding across offspring in a generation.

    workers: 0 or 1 = sequential; None = all logical CPUs; int > 1 = fixed pool size.
    For high CPU utilization, set workers ~= POPULATION_SIZE and keep plotting off.
    """
    global GA_PARALLEL_WORKERS
    GA_PARALLEL_WORKERS = workers


def get_parallel_worker_limit():
    """Maximum worker processes for the persistent pool."""
    if GA_PARALLEL_WORKERS is None:
        return max(1, os.cpu_count() or 2)
    return max(1, int(GA_PARALLEL_WORKERS))


def get_parallel_workers(task_count=1):
    """Workers to use for a batch (capped by task count and configured limit)."""
    n_workers = get_parallel_worker_limit()
    if n_workers <= 1:
        return 1
    return max(1, min(n_workers, int(task_count)))


def _pin_worker_math_library_threads():
    """One compute thread per worker process avoids oversubscription across cores."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"


def _init_parallel_worker(ctx):
    """ProcessPool initializer: pin threads and warm layout caches once per worker."""
    global _PARALLEL_WORKER_CTX
    _pin_worker_math_library_threads()
    _PARALLEL_WORKER_CTX = ctx
    alterations.warm_subject_caches(
        ctx['subject_id'],
        ctx['electrodes'],
        ctx['fiducials'],
        ctx['original_paths'],
    )


def start_parallel_worker_pool(ctx):
    """Create a persistent process pool reused across generations."""
    global _PARALLEL_EXECUTOR, _PARALLEL_EXECUTOR_WORKERS
    n_workers = get_parallel_worker_limit()
    if n_workers <= 1:
        return False
    if _PARALLEL_EXECUTOR is not None:
        return True
    _PARALLEL_EXECUTOR_WORKERS = n_workers
    print(f"Starting persistent GA worker pool ({n_workers} processes)")
    _PARALLEL_EXECUTOR = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_parallel_worker,
        initargs=(ctx,),
    )
    return True


def shutdown_parallel_worker_pool():
    """Shut down the persistent pool at end of GA.run."""
    global _PARALLEL_EXECUTOR, _PARALLEL_EXECUTOR_WORKERS
    if _PARALLEL_EXECUTOR is None:
        return
    _PARALLEL_EXECUTOR.shutdown(wait=True)
    _PARALLEL_EXECUTOR = None
    _PARALLEL_EXECUTOR_WORKERS = 0


def _parallel_worker_context(subject_id, electrodes, fiducials, original_paths):
    return {
        'subject_id': subject_id,
        'electrodes': electrodes,
        'fiducials': fiducials,
        'original_paths': original_paths,
    }


def configure_individual_plotting(enabled=False, show=False, output_dir=None):
    """Enable/disable per-individual 2D layout plots during the GA run."""
    global PLOT_EACH_INDIVIDUAL_2D, PLOT_EACH_INDIVIDUAL_SHOW, PLOT_EACH_INDIVIDUAL_DIR
    PLOT_EACH_INDIVIDUAL_2D = enabled
    PLOT_EACH_INDIVIDUAL_SHOW = show
    if output_dir is not None:
        PLOT_EACH_INDIVIDUAL_DIR = output_dir


def plot_individual_2d_if_enabled(
    INDIVIDUAL_ID: str,
    SUBJECT_ID,
    electrodes,
    fiducials,
):
    if not PLOT_EACH_INDIVIDUAL_2D:
        return
    os.makedirs(PLOT_EACH_INDIVIDUAL_DIR, exist_ok=True)
    save_path = os.path.join(
        PLOT_EACH_INDIVIDUAL_DIR, f"GA_{SUBJECT_ID}_{INDIVIDUAL_ID}.png"
    )
    alterations.plot_individual_2d_layout(
        SUBJECT_ID=SUBJECT_ID,
        INDIVIDUAL_ID=INDIVIDUAL_ID,
        electrodes=electrodes,
        fiducials=fiducials,
        show_plot=PLOT_EACH_INDIVIDUAL_SHOW,
        save_path=save_path,
        ga_phase=OPTIMIZATION_PHASE,
    )


def set_ga_optimization_phase(phase: int):
    """Switch between clearance (1) and spacing-refinement (2) modes."""
    global OPTIMIZATION_PHASE, PHASE2_GENERATION_COUNTER
    OPTIMIZATION_PHASE = phase
    if phase == 2:
        PHASE2_GENERATION_COUNTER = 0
    print(f"GA optimization phase set to {phase}")


def get_ga_optimization_phase() -> int:
    return OPTIMIZATION_PHASE


def is_early_phase2() -> bool:
    """True during the first N phase-2 generations (clearance-only parent pool)."""
    return (
        OPTIMIZATION_PHASE == 2
        and PHASE2_GENERATION_COUNTER < PHASE2_CLEARANCE_PARENT_POOL_GENERATIONS
    )


def increment_phase2_generation_counter() -> None:
    global PHASE2_GENERATION_COUNTER
    if OPTIMIZATION_PHASE == 2:
        PHASE2_GENERATION_COUNTER += 1


def count_phase1_ready_individuals(individual_ids: list, SUBJECT_ID: str) -> tuple:
    """Return (count, ids) of layouts with no electrode violations."""
    ready_ids = []
    for individual_id in individual_ids:
        if _is_individual_phase1_ready(individual_id, SUBJECT_ID):
            ready_ids.append(individual_id)
    return len(ready_ids), ready_ids


def _metrics_mode_for_phase():
    return 'electrodes_only' if OPTIMIZATION_PHASE == 1 else 'full'


def _load_saved_collision_metrics(INDIVIDUAL_ID: str, SUBJECT_ID: str):
    log_path = f"data/output/logs/GA_{SUBJECT_ID}_{INDIVIDUAL_ID}_mod_connection_paths.json"
    with open(log_path, "r") as file:
        data = json.load(file)
    return data.get('collision_metrics')


def _is_layout_collision_free_from_metrics(metrics: dict) -> bool:
    if not metrics:
        return False
    if OPTIMIZATION_PHASE == 1:
        return int(metrics.get('electrode_violations', 1)) == 0
    return is_layout_phase2_solution(metrics)


def _is_individual_phase1_ready(individual_id: str, SUBJECT_ID: str) -> bool:
    metrics = _load_saved_collision_metrics(individual_id, SUBJECT_ID)
    if metrics is not None and 'electrode_violations' in metrics:
        return int(metrics['electrode_violations']) == 0
    analysis = _load_individual_metrics(individual_id, SUBJECT_ID)
    return is_layout_phase1_ready(analysis)


def maybe_transition_to_phase2_after_generation(SUBJECT_ID: str, individual_ids: list) -> None:
    """Switch to phase 2 after a full generation has >= N electrode-free layouts."""
    if OPTIMIZATION_PHASE != 1:
        return

    count, ready_ids = count_phase1_ready_individuals(individual_ids, SUBJECT_ID)
    print(
        f"Generation phase-1 check: {count}/{len(individual_ids)} electrode-free "
        f"(need {PHASE2_READY_THRESHOLD} to enter phase 2)"
    )
    if count >= PHASE2_READY_THRESHOLD:
        print(f"  Phase-1-ready individuals: {ready_ids}")
        set_ga_optimization_phase(2)


def build_clearance_free_parent_pool(parent_fitness_map: dict, SUBJECT_ID: str) -> dict:
    """Restrict early phase-2 parents to far-region-clear, electrode-free individuals."""
    cleared_pool = {}
    for individual_id in parent_fitness_map:
        if _is_individual_phase1_ready(individual_id, SUBJECT_ID):
            cleared_pool[individual_id] = parent_fitness_map[individual_id]

    if not cleared_pool:
        print(
            "⚠️ Early phase 2: no phase-1-ready parents available; "
            "falling back to full parent pool"
        )
        return parent_fitness_map

    print(
        f"Early phase 2 parent pool: {len(cleared_pool)}/{len(parent_fitness_map)} "
        f"phase-1-ready (phase-2 gen {PHASE2_GENERATION_COUNTER + 1}/"
        f"{PHASE2_CLEARANCE_PARENT_POOL_GENERATIONS})"
    )
    return cleared_pool

def saveFitnessTrackerToFile(data: dict, SUBJECT_ID: str = None):
    os.makedirs("data/output/logs", exist_ok=True)
    with open(f"data/output/logs/GA_{SUBJECT_ID}_fitness_tracker.json", "w") as file:
        json.dump(data, file, indent=4)

def calculate_2d_path_length(path_2d: list) -> float:
    """Calculate the total length of a 2D path. Returns a `float` representing the total length of ONE path only, not the sum of all paths."""
    if len(path_2d) < 2:
        return 0.0
    
    # Calculate differences between consecutive points
    diffs = np.diff(path_2d, axis=0)
    
    # Calculate Euclidean distances between points
    distances = np.linalg.norm(diffs, axis=1)
    
    # Sum all distances
    return float(np.sum(distances))



def compute_fitness_score(analysis: dict, path_length_excess: float = 0.0) -> float:
    """GA selection fitness (higher is better) = -layout_score."""
    if path_length_excess == 0.0 and 'path_length_excess' in analysis:
        path_length_excess = float(analysis['path_length_excess'])
    layout_score = compute_layout_score(
        analysis, OPTIMIZATION_PHASE, path_length_excess
    )
    return -layout_score


def _load_individual_metrics(INDIVIDUAL_ID: str, SUBJECT_ID: str) -> dict:
    """Recompute collision and terminal-entry spacing metrics from saved path geometry."""
    with open(f"data/output/logs/GA_{SUBJECT_ID}_{INDIVIDUAL_ID}_mod_connection_paths.json", "r") as file:
        mod_connection_paths = json.load(file)

    paths_2d = [np.array(path['modified_path_2d']) for path in mod_connection_paths['paths']]
    path_electrodes = [path['electrode'] for path in mod_connection_paths['paths']]
    path_terminals = [path['terminal'] for path in mod_connection_paths['paths']]
    _, slot_index, _ = alterations.slot_metadata_from_child_paths(mod_connection_paths['paths'])

    electrode_zones, terminal_zones = alterations.load_zones_for_subject(SUBJECT_ID)
    path_length_excess = 0.0
    if OPTIMIZATION_PHASE == 2:
        ctx = alterations.get_subject_layout(SUBJECT_ID)
        path_length_excess = compute_layout_path_length_excess(
            paths_2d,
            path_electrodes,
            path_terminals,
            ctx['electrodes_2d'],
            ctx['terminals_2d'],
        )
    return analyze_path_collisions(
        paths_2d,
        terminal_zones,
        electrode_zones=electrode_zones,
        path_electrodes=path_electrodes,
        path_terminals=path_terminals,
        metrics_mode=_metrics_mode_for_phase(),
        ga_phase=OPTIMIZATION_PHASE,
        path_length_excess=path_length_excess,
        slot_index_by_electrode=slot_index or None,
    )


def _load_individual_path_length_excess(INDIVIDUAL_ID: str, SUBJECT_ID: str) -> float:
    with open(f"data/output/logs/GA_{SUBJECT_ID}_{INDIVIDUAL_ID}_mod_connection_paths.json", "r") as file:
        mod_connection_paths = json.load(file)

    paths_2d = [np.array(path['modified_path_2d']) for path in mod_connection_paths['paths']]
    path_electrodes = [path['electrode'] for path in mod_connection_paths['paths']]
    path_terminals = [path['terminal'] for path in mod_connection_paths['paths']]

    ctx = alterations.get_subject_layout(SUBJECT_ID)
    return compute_layout_path_length_excess(
        paths_2d,
        path_electrodes,
        path_terminals,
        ctx['electrodes_2d'],
        ctx['terminals_2d'],
    )


def getIndividual2DFitnessScoreFromFileLogs(INDIVIDUAL_ID: str = None, SUBJECT_ID: str = None, verbose: bool = False) -> float:
    """
    GA fitness = -layout_score (higher is better).
    layout_score is the single penalty metric stored in collision_metrics.
    """
    if INDIVIDUAL_ID is None:
        raise ValueError("INDIVIDUAL_ID must be provided to load the correct file.")
    if SUBJECT_ID is None:
        raise ValueError("SUBJECT_ID must be provided to load the correct file.")

    metrics = _load_saved_collision_metrics(INDIVIDUAL_ID, SUBJECT_ID)
    if metrics is not None:
        path_length_excess = 0.0
        if OPTIMIZATION_PHASE == 2:
            path_length_excess = metrics.get('path_length_excess')
            if path_length_excess is None:
                path_length_excess = _load_individual_path_length_excess(
                    INDIVIDUAL_ID, SUBJECT_ID
                )
            else:
                path_length_excess = float(path_length_excess)
        layout_score = compute_layout_score(
            metrics, OPTIMIZATION_PHASE, path_length_excess
        )
        fitness_score = -layout_score
        if verbose:
            print(
                f"{INDIVIDUAL_ID}: layout_score={layout_score:.4f}, "
                f"fitness={fitness_score:.4f} (from saved metrics)"
            )
        return fitness_score

    analysis = _load_individual_metrics(INDIVIDUAL_ID, SUBJECT_ID)
    path_length_excess = 0.0
    if OPTIMIZATION_PHASE == 2:
        path_length_excess = _load_individual_path_length_excess(
            INDIVIDUAL_ID, SUBJECT_ID
        )
    layout_score = compute_layout_score(
        analysis, OPTIMIZATION_PHASE, path_length_excess
    )
    fitness_score = -layout_score

    if verbose:
        if OPTIMIZATION_PHASE == 1:
            print(
                f"{INDIVIDUAL_ID}: layout_score={layout_score:.4f}, "
                f"fitness={fitness_score:.4f} "
                f"(electrodes={analysis['electrode_violations']})"
            )
        else:
            min_dist = analysis.get(
                'min_closest_neighbor_entry_distance',
                analysis['min_terminal_entry_distance'],
            )
            min_dist_str = f"{min_dist:.2f}" if min_dist != float('inf') else "inf"
            print(
                f"{INDIVIDUAL_ID}: layout_score={layout_score:.2f}, "
                f"fitness={fitness_score:.4f}, "
                f"far_x={analysis.get('far_crossing_count', 0)}, "
                f"near_x={analysis.get('near_crossing_count', 0)}, "
                f"far_overlap={analysis.get('far_overlap_length', 0.0):.2f}, "
                f"near_overlap={analysis.get('near_overlap_length', 0.0):.2f}, "
                f"trace_sep_norm={analysis['trace_separation_deficit_normalized']:.2f}, "
                f"min_trace_sep={analysis.get('min_trace_separation', float('inf')):.2f}, "
                f"electrodes={analysis['electrode_violations']}, "
                f"length_excess={path_length_excess:.3f}, "
                f"min_closest_entry_neighbor={min_dist_str}"
            )

    return fitness_score


def getIndividualCollisionCountFromFileLogs(INDIVIDUAL_ID: str = None, SUBJECT_ID: str = None) -> int:
    if INDIVIDUAL_ID is None:
        raise ValueError("INDIVIDUAL_ID must be provided to load the correct file.")
    if SUBJECT_ID is None:
        raise ValueError("SUBJECT_ID must be provided to load the correct file.")
    analysis = _load_individual_metrics(INDIVIDUAL_ID, SUBJECT_ID)
    if OPTIMIZATION_PHASE == 2:
        return 0 if is_layout_phase2_solution(analysis) else 1
    return 0 if is_layout_collision_free(analysis) else 1


def get_N_elites_from_specific_gen(generation_counter_integer: int, n: int, SUBJECT_ID: int) -> dict:
    """Returns `n` elites from any specified geeration as a `dict`, with the elite individuals' IDs as keys and their fitness scores as values.
    - `generation_counter_integer` is the generation number from which to retrieve the elites. I.e., if `generation_counter_integer=0`, and `n=2`, it returns the 2 fittest individuals from the first generation.
    """
    with open(f"data/output/logs/GA_{SUBJECT_ID}_fitness_tracker.json", "r") as file:
        fitness_tracker = json.load(file)
    
    logs_for_specific_gen =  {key: value for key, value in fitness_tracker.items() if key.startswith(str(generation_counter_integer))}
    # Sort the individuals by their fitness scores in descending order and return the top `n` individuals
    n_elites = dict(sorted(logs_for_specific_gen.items(), key=lambda item: item[1], reverse=True)[:n])
    return n_elites

def rename_elite_dict(elite_dict: dict, generation_counter_integer: int) -> dict:
    """Renames the elite individuals' IDs to include the current generation number."""
    renamed_elites = {}
    for key, value in elite_dict.items():
        new_key = f"{generation_counter_integer}-{key.split('-')[1]}"  # Keep the individual ID part
        renamed_elites[new_key] = value
    return renamed_elites
    


def get_parents_mating_combinations_for_next_generation(GEN_IND: int=None, POPULATION_SIZE: int=None) -> dict:
    """Takes in the index `GEN_IND` and returns the combination of parental INDIVIDUAL_IDs for so the next generation can be created using Darwinistic operators.
    - `GEN_IND` is the index of the generation for which the parents are selected. **Example use:** If `GEN_IND=1`, it returns the parents for the `SECOND` generation (index `1`), which should be called when attempting to create the `SECOND` generation based on the `FIRST` generation (`gen 0`).
    - `POPULATION_SIZE` is the number of individuals that need to be created. This will determine the number of keys in the returned `dict`, one `key` for each new individual, with the value being a list of parental `INDIVIDUAL_IDs` (two each).
    """
    if GEN_IND is None or GEN_IND == 0:
        raise ValueError("GEN_IND must be provided and should not be 0. Use 1 or higher to get parents for the next generation.")
    if POPULATION_SIZE is None or POPULATION_SIZE <= 2:
        raise ValueError("POPULATION_SIZE must be provided and should be greater than 2.")

    PARENTS_GEN = GEN_IND - 1  # Eltern müssen immer aus der vorherigen Generation kommen
    
    # Synthetische liste von Eltern-IDs für die aktuelle Generation erstellen, mit der Annahme, dass alle Dateien diesbezüglich existieren
    potential_parent_ids = [f"{PARENTS_GEN}-{i}" for i in range(POPULATION_SIZE)]

    return potential_parent_ids


def single_point_crossover(parent1_paths: list, parent2_paths: list):
    """
    Nimmt direkt die 2D-Pfade der Elternteile und führt **Single Point Crossover** durch, um ein neues Individuum zu erzeugen.
    """
    
    anzahl_der_elektroden = len(parent1_paths['paths'])
    
    # Schnittstelle soll nicht nicht ganz am Anfang oder Ende liegen, sondern bei +- 1/3 der Anzahl der Elektroden
    minimum_wert, maximum_wert = int(anzahl_der_elektroden / 3), int(anzahl_der_elektroden * 2 / 3)
    single_crossover_point = random.randint(minimum_wert, maximum_wert)

    # Jetzt direkt die Pfade Tauschen für alle Elektroden, die nach dem Schnittpunkt liegen
    for i in range(single_crossover_point, anzahl_der_elektroden):
        # Tausche nur die modified_path_2d und n_collisions, aber NICHT terminal
        # Terminal assignments müssen konsistent bleiben basierend auf initial assignments
        parent1_paths['paths'][i]['modified_path_2d'] = parent2_paths['paths'][i]['modified_path_2d']
        parent1_paths['paths'][i]['n_collisions'] = parent2_paths['paths'][i]['n_collisions']
        # Terminal bleibt konsistent: parent1_paths['paths'][i]['terminal'] bleibt unverändert
    
    # noch nicht speichern - erst Returnen und dann mutieren, falls nötig, dann speichern...
    return parent1_paths

def uniform_crossover(parent1_paths: list, parent2_paths: list):
    """
    Nimmt direkt die 2D-Pfade der Elternteile und führt **Uniform Crossover** durch, um ein neues Individuum zu erzeugen.
    
    `RETURNS` The new path as 'updated' parent1_paths-style list with same keys, **untouched values EXCEPT** the `modified_path_2d` values, which are now a mix of both parents' paths.
    """

    if parent1_paths['paths'] == parent2_paths['paths']:
        print("The 2d paths are the same, no crossover needed.")
        pass
    else:
        pass

    if parent1_paths['paths'][5]['original_path_3d'] == parent2_paths['paths'][5]['original_path_3d']:
        pass
    else:
        print("🔴 WTF")
        raise ValueError("The 3D paths of the parents are not the same, which is unexpected!!! This legacy leftover should be same.")
    
    for i in range(len(parent1_paths['paths'])):
        # Randomly choose which parent's path to take for each electrode
        if random.random() < 0.5:
            # Take from parent1 - keep everything as is
            parent1_paths['paths'][i]['modified_path_2d'] = parent1_paths['paths'][i]['modified_path_2d']
        else:
            # Take from parent2 - copy path and collision count, but keep consistent terminal assignment
            parent1_paths['paths'][i]['modified_path_2d'] = parent2_paths['paths'][i]['modified_path_2d']
            parent1_paths['paths'][i]['n_collisions'] = parent2_paths['paths'][i]['n_collisions']
            # Terminal bleibt konsistent: parent1_paths['paths'][i]['terminal'] bleibt unverändert
    
    # noch nicht speichern - erst Returnen und dann mutieren, falls nötig, dann speichern...
    return parent1_paths # Alle Änderungen sind direkt in den Pfaden des ersten Elternteils gespeichert, weil faul oder so


def intercourse(SUBJECT_ID: int, parents: list, MUTATION_RATE: float, CROSSOVER_STRATEGY: str, original_paths, electrodes, fiducials) -> None:
    """
    Nimmt die 2D-Pfade der Elternteile, die aus dem `new_generation_map` geladen wurden, und führt **Crossover und Mutation** durch, um ein neues Individuum zu erzeugen.
    
    - Speichert das neue Individuum mit 2D-Pfaden direct ab
    - Die Elternteile werden aus den Pfaden geladen, die in `new_generation_map` gespeichert sind.
    - Mutation wird direkt angewendet, falls zutreffend (random), **bevor** das Kind gespeichert wird.
    """
    # PERCENTAGE OF ELECTRODES TO BE ALTERED RANDOMLY SHOULD A CHILD BE SELECTED FOR MUTATION LATER DURING CONCEPTION
    MUTATE_N_ELECTRODES_PERCENTAGE = 0.4
    
    # Elternteile laden (deren Genome und Koordinaten als 2D-Pfade)
    with open(f"data/output/logs/GA_{SUBJECT_ID}_{parents[0]}_mod_connection_paths.json", "r") as file1, \
        open(f"data/output/logs/GA_{SUBJECT_ID}_{parents[1]}_mod_connection_paths.json", "r") as file2:
        parent1_paths = json.load(file1)
        parent2_paths = json.load(file2)

    if CROSSOVER_STRATEGY == "SINGLE_POINT":
        child_paths = single_point_crossover(parent1_paths=parent1_paths, parent2_paths=parent2_paths)
        criterion = random.random()

        if criterion < MUTATION_RATE:
            child_paths = alterations.mutateRandomElectrodePathsForSelectedChild(child=child_paths, MUTATE_N_ELECTRODES_PERCENTAGE=MUTATE_N_ELECTRODES_PERCENTAGE, SUBJECT_ID=SUBJECT_ID, electrodes=electrodes, fiducials=fiducials, original_paths=original_paths, ga_phase=OPTIMIZATION_PHASE)

        child_paths = alterations.apply_smart_collision_resolution(
            child=child_paths,
            SUBJECT_ID=SUBJECT_ID,
            electrodes=electrodes,
            fiducials=fiducials,
            original_paths=original_paths,
            greedy_electrodes_only=(OPTIMIZATION_PHASE == 1),
        )

    elif CROSSOVER_STRATEGY == "UNIFORM":
        child_paths = uniform_crossover(parent1_paths=parent1_paths, parent2_paths=parent2_paths)

        criterion = random.random()
        if criterion < MUTATION_RATE:
            child_paths = alterations.mutateRandomElectrodePathsForSelectedChild(child=child_paths, MUTATE_N_ELECTRODES_PERCENTAGE=MUTATE_N_ELECTRODES_PERCENTAGE, SUBJECT_ID=SUBJECT_ID, electrodes=electrodes, fiducials=fiducials, original_paths=original_paths, ga_phase=OPTIMIZATION_PHASE)

        child_paths = alterations.apply_smart_collision_resolution(
            child=child_paths,
            SUBJECT_ID=SUBJECT_ID,
            electrodes=electrodes,
            fiducials=fiducials,
            original_paths=original_paths,
            greedy_electrodes_only=(OPTIMIZATION_PHASE == 1),
        )

    else:
        raise ValueError(f"Invalid CROSSOVER_STRATEGY: {CROSSOVER_STRATEGY}. Must be 'SINGLE_POINT' or 'UNIFORM'.")
    
    return child_paths
    
    
    
def getIndFitnessUsingFITNESSTRACKER(INDIVIDUAL_ID: str, SUBJECT_ID: str) -> float:
    """All it does is open the corresponding fitness tracker file and return the fitness score thats stored as value for the `INDIVIDUAL_ID` key."""

    with open(f"data/output/logs/GA_{SUBJECT_ID}_fitness_tracker.json", "r") as file:
        fitness_tracker = json.load(file)
        
    if INDIVIDUAL_ID not in fitness_tracker:
        raise ValueError(f"Individual ID {INDIVIDUAL_ID} not found in fitness tracker for subject {SUBJECT_ID}.")
    else:
        return fitness_tracker[INDIVIDUAL_ID]
    
def select_parents_by_tournament(current_generation_fitness, population_size, tournament_size=2, n_elites=None):
    """
    Select parents for the next generation using tournament selection with elitism support.
    Elite individuals will have None as parents (to be preserved unchanged).
    
    Args:
        current_generation_fitness (dict): Fitness scores of current generation {individual_id: fitness}
        population_size (int): Number of individuals in the next generation
        tournament_size (int): Number of participants in each tournament
        n_elites (int): Number of top individuals to preserve as elites
        
    Returns:
        dict: Mapping of new individual IDs to their parent IDs. 
              Elite individuals have None as parents.
              Format: {new_id: [parent1_id, parent2_id] or None}
    """
    if n_elites is None:
        raise ValueError("n_elites must be provided to determine how many elite individuals to preserve.")
    
    parent_selections = {}
    current_gen_num = int(list(current_generation_fitness.keys())[0].split('-')[0])
    next_gen_num = current_gen_num + 1
    
    # Get the elite individuals (top n_elites by fitness)
    elites = sorted(current_generation_fitness.items(), 
                   key=lambda x: x[1], 
                   reverse=True)[:n_elites]
    elite_ids = [id for id, _ in elites]
    
    for i in range(population_size):
        new_id = f"{next_gen_num}-{i}"
        
        # Preserve elites by setting parents to None
        if i < n_elites and i < len(elite_ids):
            # Mark as elite, no parents needed (only the ID of the actual elite)
            parent_selections[new_id] = ["ELITE", elite_ids[i]] 
        else:
            # Select first parent via tournament (non-elite)
            parent1 = tournament_selection(current_generation_fitness, tournament_size)

            if len(current_generation_fitness) == 1:
                parent2 = parent1
            else:
                parent2 = tournament_selection(current_generation_fitness, tournament_size)
                while parent2 == parent1:
                    parent2 = tournament_selection(current_generation_fitness, tournament_size)
            
            parent_selections[new_id] = [parent1, parent2]
    
    return parent_selections

def tournament_selection(population_fitness, tournament_size):
    """Helper function for tournament selection."""
    ids = list(population_fitness.keys())
    k = min(tournament_size, len(ids))
    participants = random.sample(ids, k)
    return max(participants, key=lambda x: population_fitness[x])


def _breed_single_individual(
    new_individual_id,
    parents,
    *,
    SUBJECT_ID,
    electrodes,
    fiducials,
    original_paths,
    MUTATION_RATE,
    CROSSOVER_STRATEGY,
    ga_phase,
):
    """Create one offspring (or copy an elite) and return fitness / optional solution flag."""
    global OPTIMIZATION_PHASE
    previous_phase = OPTIMIZATION_PHASE
    OPTIMIZATION_PHASE = ga_phase
    found_solution = None

    try:
        if "ELITE" in parents:
            elite_source_id = parents[1]
            src = f"data/output/logs/GA_{SUBJECT_ID}_{elite_source_id}_mod_connection_paths.json"
            dst = f"data/output/logs/GA_{SUBJECT_ID}_{new_individual_id}_mod_connection_paths.json"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

            if OPTIMIZATION_PHASE == 2:
                alterations.refresh_saved_individual_collision_metrics(
                    SUBJECT_ID,
                    new_individual_id,
                    ga_phase=2,
                )
        else:
            if parents is None or len(parents) != 2:
                raise ValueError(
                    f"Invalid parents for new individual {new_individual_id}: {parents}. "
                    "Expected exactly two parents."
                )

            new_offspring_genome = intercourse(
                SUBJECT_ID=SUBJECT_ID,
                parents=parents,
                MUTATION_RATE=MUTATION_RATE,
                CROSSOVER_STRATEGY=CROSSOVER_STRATEGY,
                original_paths=original_paths,
                electrodes=electrodes,
                fiducials=fiducials,
            )

            saved = alterations.only_save_new_2D_alteration(
                child=new_offspring_genome,
                SUBJECT_ID=SUBJECT_ID,
                electrodes=electrodes,
                fiducials=fiducials,
                INDIVIDUAL_ID=new_individual_id,
                metrics_mode=_metrics_mode_for_phase(),
                collision_metrics=new_offspring_genome.pop('collision_metrics', None),
                ga_phase=OPTIMIZATION_PHASE,
            )
            if saved != 200:
                raise ValueError(f"Failed to save new individual {new_individual_id}")

        fitness_score = round(
            getIndividual2DFitnessScoreFromFileLogs(
                INDIVIDUAL_ID=new_individual_id,
                SUBJECT_ID=SUBJECT_ID,
                verbose=False,
            ),
            4,
        )

        if OPTIMIZATION_PHASE == 2:
            offspring_metrics = _load_saved_collision_metrics(new_individual_id, SUBJECT_ID)
            if _is_layout_collision_free_from_metrics(offspring_metrics):
                found_solution = {
                    'ID': new_individual_id,
                    'fitness_score': fitness_score,
                }

        return {
            'individual_id': new_individual_id,
            'fitness_score': fitness_score,
            'found_solution': found_solution,
        }
    finally:
        OPTIMIZATION_PHASE = previous_phase


def _process_breeding_task(task):
    """ProcessPool entry point for one offspring."""
    ctx = _PARALLEL_WORKER_CTX
    return _breed_single_individual(
        task['new_individual_id'],
        task['parents'],
        SUBJECT_ID=ctx['subject_id'],
        electrodes=ctx['electrodes'],
        fiducials=ctx['fiducials'],
        original_paths=ctx['original_paths'],
        MUTATION_RATE=task['mutation_rate'],
        CROSSOVER_STRATEGY=task['crossover_strategy'],
        ga_phase=task['ga_phase'],
    )


def _process_init_individual_task(task):
    """ProcessPool entry point for generation-0 random initialization."""
    ctx = _PARALLEL_WORKER_CTX
    individual_id = task['individual_id']
    alterations.create_and_save_new_2D_alteration(
        SUBJECT_ID=ctx['subject_id'],
        original_paths=ctx['original_paths'],
        electrodes=ctx['electrodes'],
        fiducials=ctx['fiducials'],
        INDIVIDUAL_ID=individual_id,
    )
    fitness_score = round(
        getIndividual2DFitnessScoreFromFileLogs(
            INDIVIDUAL_ID=individual_id,
            SUBJECT_ID=ctx['subject_id'],
            verbose=False,
        ),
        4,
    )
    return {
        'individual_id': individual_id,
        'fitness_score': fitness_score,
    }


def _run_tasks_in_parallel(tasks, worker_fn, ctx, desc):
    """Run independent GA tasks in a process pool."""
    n_workers = get_parallel_workers(len(tasks))
    if n_workers <= 1:
        return [worker_fn(task) for task in tqdm(tasks, desc=desc, total=len(tasks))]

    global _PARALLEL_EXECUTOR
    if _PARALLEL_EXECUTOR is None:
        start_parallel_worker_pool(ctx)

    print(f"{desc}: {n_workers} workers, {len(tasks)} tasks")
    chunksize = max(1, len(tasks) // (n_workers * 2))
    return list(
        tqdm(
            _PARALLEL_EXECUTOR.map(worker_fn, tasks, chunksize=chunksize),
            desc=desc,
            total=len(tasks),
        )
    )


def initialize_generation_parallel(
    SUBJECT_ID,
    generation_id,
    population_size,
    electrodes,
    fiducials,
    original_paths,
):
    """Initialize generation 0 individuals, optionally in parallel."""
    ctx = _parallel_worker_context(SUBJECT_ID, electrodes, fiducials, original_paths)
    tasks = [
        {'individual_id': f"{generation_id}-{i}"}
        for i in range(population_size)
    ]
    results = _run_tasks_in_parallel(
        tasks,
        _process_init_individual_task,
        ctx,
        desc=f"Initializing generation {generation_id}",
    )
    return {item['individual_id']: item['fitness_score'] for item in results}


def lets_fucking_breed_whole_gen(SUBJECT_ID: str = None, potential_parents: list = None, tournament_size: int = None, N_ELITES: int = None, MUTATION_RATE: float = None, CROSSOVER_STRATEGY: str = None, original_paths=None, electrodes=None, fiducials=None) -> None:
    """Breed next generation; greedy aggressive resolution follows OPTIMIZATION_PHASE."""
    found_solution = False
    was_phase2_at_start = OPTIMIZATION_PHASE == 2
    early_parent_pool = is_early_phase2()
    print(
        f'GA phase {get_ga_optimization_phase()} '
        f"(electrode greedy={OPTIMIZATION_PHASE == 1}, "
        f"gentle={OPTIMIZATION_PHASE == 2}, "
        f"phase-1-ready parents={'on' if early_parent_pool else 'off'})"
    )
    if tournament_size is None or N_ELITES is None:
        raise ValueError("tournament_size and N_ELITES must be provided to perform breeding operations.")

    parent_fitness_map = {}

    for parent_name in potential_parents:
        parent_fitness_map[parent_name] = getIndFitnessUsingFITNESSTRACKER(INDIVIDUAL_ID=parent_name, SUBJECT_ID=SUBJECT_ID)

    selection_fitness_map = parent_fitness_map
    if is_early_phase2():
        selection_fitness_map = build_clearance_free_parent_pool(parent_fitness_map, SUBJECT_ID)

    # Select parents using tournament selection
    new_generation_map = select_parents_by_tournament(
        current_generation_fitness=selection_fitness_map,
        population_size=len(potential_parents),
        tournament_size=tournament_size,
        n_elites=N_ELITES,
    )

    # new_generation_map now looks something like {"1-0": ["0-1", "0-2"], "1-1": ["0-3", "0-4"]}

    # create a loop to load every new individual's parent's genomes from the map var and then let them fucki fucki
    with open(f"data/output/logs/GA_{SUBJECT_ID}_fitness_tracker.json", "r") as file:
        fitness_tracker = json.load(file)

    ctx = _parallel_worker_context(SUBJECT_ID, electrodes, fiducials, original_paths)
    breeding_tasks = [
        {
            'new_individual_id': new_individual_id,
            'parents': parents,
            'ga_phase': OPTIMIZATION_PHASE,
            'mutation_rate': MUTATION_RATE,
            'crossover_strategy': CROSSOVER_STRATEGY,
        }
        for new_individual_id, parents in new_generation_map.items()
    ]
    n_workers = get_parallel_workers(len(breeding_tasks))
    use_parallel = n_workers > 1

    if use_parallel and PLOT_EACH_INDIVIDUAL_2D:
        print("Parallel breeding: per-individual plots run in main process after workers finish")

    if use_parallel:
        breeding_results = _run_tasks_in_parallel(
            breeding_tasks,
            _process_breeding_task,
            ctx,
            desc=f"Breeding {len(breeding_tasks)} new inds",
        )
    else:
        breeding_results = []
        for task in tqdm(breeding_tasks, desc=f"Breeding {len(breeding_tasks)} new inds"):
            breeding_results.append(
                _breed_single_individual(
                    task['new_individual_id'],
                    task['parents'],
                    SUBJECT_ID=SUBJECT_ID,
                    electrodes=electrodes,
                    fiducials=fiducials,
                    original_paths=original_paths,
                    MUTATION_RATE=MUTATION_RATE,
                    CROSSOVER_STRATEGY=CROSSOVER_STRATEGY,
                    ga_phase=task['ga_phase'],
                )
            )

    for result in breeding_results:
        new_individual_id = result['individual_id']
        fitness_tracker[new_individual_id] = result['fitness_score']
        plot_individual_2d_if_enabled(
            new_individual_id, SUBJECT_ID, electrodes, fiducials
        )
        if result.get('found_solution'):
            found_solution = result['found_solution']

    saveFitnessTrackerToFile(data=fitness_tracker, SUBJECT_ID=SUBJECT_ID)

    maybe_transition_to_phase2_after_generation(
        SUBJECT_ID, list(new_generation_map.keys())
    )
    if was_phase2_at_start:
        increment_phase2_generation_counter()

    if found_solution != False:
        print(f"🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 \nFound solution in generation {new_individual_id.split('-')[0]}: {found_solution}\n🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉")
        return found_solution
    else:
        return None


if __name__ == "__main__":
    print("Ain't nobody got time for this 🎻. Usually this shouldn't be run.")