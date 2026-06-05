from tqdm.auto import tqdm
from PYTHON.tools.new2dAlterations import create_and_save_new_2D_alteration, warm_subject_caches
import json
import PYTHON.GA.geneticOperators as genetics

def run(
    electrodes,
    fiducials,
    SUBJECT_ID,
    *,
    start_generation=0,
    initial_fitness_tracker=None,
    n_generations=100,
    population_size=20,
):
    """
    Run the genetic algorithm.

    Parameters
    ----------
    start_generation : int
        First generation index to evolve (use 1 after warm-starting generation 0).
    initial_fitness_tracker : dict or None
        Pre-filled fitness map (e.g. from preset warm-start); merged before breeding.
    """
    genetics.set_ga_optimization_phase(1)

    # Optional: save a 2D PNG after every individual (set show=True to pop up each plot).
    PLOT_EACH_INDIVIDUAL_2D = False
    PLOT_EACH_INDIVIDUAL_SHOW = False
    genetics.configure_individual_plotting(
        enabled=PLOT_EACH_INDIVIDUAL_2D,
        show=PLOT_EACH_INDIVIDUAL_SHOW,
    )

    # Parallel offspring evaluation within each generation.
    # 0 = sequential (recommended on Windows with POPULATION_SIZE ~20).
    # None = all logical CPUs — each worker re-imports pyvista/shapely; high spawn cost.
    # N > 1 = fixed pool; match POPULATION_SIZE for best utilization.
    GA_PARALLEL_WORKERS = 0
    genetics.configure_parallel_breeding(workers=GA_PARALLEL_WORKERS)

    print('Starting two-phase GA (phase 1: electrode clearance, phase 2: trace resolution)')
    
    N_GENERATIONS = n_generations
    POPULATION_SIZE = population_size

    # GENETIC OPERATOR VARIABLES
    CROSSOVER_STRATEGY = "UNIFORM" # Can only be UNIFORM or SINGLE_POINT
    MUTATION_RATE = 0.5
    ELITISM_RATE = 0.05
    TOURNAMENT_SIZE = 2  # Number of parents to select for tournament selection
    
    N_ELITES = int(POPULATION_SIZE * ELITISM_RATE)  # Number of elites to carry over to the next generation
    
    print(N_ELITES, "elites will be carried over to each next generation")
    
    print(f'------ SETUP --------\nN_GENERATIONS={N_GENERATIONS}\nPOPULATION_SIZE={POPULATION_SIZE}\nCROSSOVER_STRATEGY={CROSSOVER_STRATEGY}\nELITISM_RATE={ELITISM_RATE}\nMUTATION_RATE={MUTATION_RATE}')
    print('----------------------')
    
    
    with open(f"data/json/init_connection_paths_{SUBJECT_ID}.json", "r") as file:
        OG_PATH_DATA = json.load(file)

    warm_subject_caches(SUBJECT_ID, electrodes, fiducials, OG_PATH_DATA)

    parallel_ctx = genetics._parallel_worker_context(
        SUBJECT_ID, electrodes, fiducials, OG_PATH_DATA
    )
    if genetics.get_parallel_worker_limit() > 1:
        genetics.start_parallel_worker_pool(parallel_ctx)
    
    all_gens_fitness_tracker = dict(initial_fitness_tracker or {})

    best_solution_opt = None
    try:
        gen_range = range(start_generation, N_GENERATIONS)
        for generationID in tqdm(
            gen_range,
            desc=f"\nEvolving gens {start_generation}–{N_GENERATIONS - 1}",
            total=len(gen_range),
        ):

            if generationID == 0:
                n_workers = genetics.get_parallel_workers(POPULATION_SIZE)
                if n_workers > 1:
                    gen0_fitness = genetics.initialize_generation_parallel(
                        SUBJECT_ID=SUBJECT_ID,
                        generation_id=generationID,
                        population_size=POPULATION_SIZE,
                        electrodes=electrodes,
                        fiducials=fiducials,
                        original_paths=OG_PATH_DATA,
                    )
                    all_gens_fitness_tracker.update(gen0_fitness)
                    for individual_id in gen0_fitness:
                        genetics.plot_individual_2d_if_enabled(
                            individual_id,
                            SUBJECT_ID,
                            electrodes,
                            fiducials,
                        )
                else:
                    for i in tqdm(range(POPULATION_SIZE), total=POPULATION_SIZE, desc="Initializing initial generation"):
                        CURRENT_INDIVIDUAL_ID = str(generationID) + "-" + str(i)
                        create_and_save_new_2D_alteration(SUBJECT_ID=SUBJECT_ID, original_paths=OG_PATH_DATA, electrodes=electrodes, fiducials=fiducials, INDIVIDUAL_ID=CURRENT_INDIVIDUAL_ID)
                        all_gens_fitness_tracker[CURRENT_INDIVIDUAL_ID] = round(genetics.getIndividual2DFitnessScoreFromFileLogs(INDIVIDUAL_ID=CURRENT_INDIVIDUAL_ID, SUBJECT_ID=SUBJECT_ID, verbose=False), 4)
                        genetics.plot_individual_2d_if_enabled(
                            CURRENT_INDIVIDUAL_ID,
                            SUBJECT_ID,
                            electrodes,
                            fiducials,
                        )

                genetics.saveFitnessTrackerToFile(data=all_gens_fitness_tracker, SUBJECT_ID=SUBJECT_ID)
                genetics.maybe_transition_to_phase2_after_generation(
                    SUBJECT_ID,
                    [f"{generationID}-{i}" for i in range(POPULATION_SIZE)],
                )

            else:
                selected_parents = genetics.get_parents_mating_combinations_for_next_generation(GEN_IND=generationID, POPULATION_SIZE=POPULATION_SIZE)

                best_solution_opt = genetics.lets_fucking_breed_whole_gen(
                    SUBJECT_ID=SUBJECT_ID,
                    potential_parents=selected_parents,
                    tournament_size=TOURNAMENT_SIZE,
                    N_ELITES=N_ELITES,
                    MUTATION_RATE=MUTATION_RATE,
                    CROSSOVER_STRATEGY=CROSSOVER_STRATEGY,
                    original_paths=OG_PATH_DATA,
                    electrodes=electrodes,
                    fiducials=fiducials,
                )

                if best_solution_opt is not None:
                    break
    finally:
        genetics.shutdown_parallel_worker_pool()

    return best_solution_opt

