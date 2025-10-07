####################################
###################### BindCraft Run
####################################
### Import dependencies
import sys
from functions import *

# Check if JAX-capable GPU is available, otherwise exit
check_jax_gpu()

######################################
### parse input paths
parser = argparse.ArgumentParser(description='Script to run BindCraft binder design.')

parser.add_argument('--settings', '-s', type=str, required=True,
                    help='Path to the basic settings.json file. Required.')
parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                    help='Path to the filters.json file used to filter design. If not provided, default will be used.')
parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json',
                    help='Path to the advanced.json file with additional design settings. If not provided, default will be used.')
parser.add_argument('--prefilters', '-p', type=str, default=None,
                    help='Path to the prefilters.json file used to pre-filter trajectories before MPNN redesign. If not provided, default will be used.')

args = parser.parse_args()

# perform checks of input setting files
settings_path, filters_path, advanced_path, prefilters_path = perform_input_check(args)

### load settings from JSON
target_settings, advanced_settings, filters, prefilters = load_json_settings(settings_path, filters_path, advanced_path, prefilters_path)

settings_file = os.path.basename(settings_path).split('.')[0]
filters_file = os.path.basename(filters_path).split('.')[0]
advanced_file = os.path.basename(advanced_path).split('.')[0]
if prefilters_path is not None:
    prefilters_file = os.path.basename(prefilters_path).split('.')[0]
else:
    prefilters_file = None

### load AF2 model settings
design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

### perform checks on advanced_settings
bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

### generate directories, design path names can be found within the function
design_paths = generate_directories(target_settings["design_path"])

### generate dataframes
trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
AF2_design_csv = os.path.join(target_settings["design_path"], 'AF2_design_stats.csv')
final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

create_dataframe(trajectory_csv, trajectory_labels)
create_dataframe(mpnn_csv, design_labels)
create_dataframe(AF2_design_csv, design_labels)
create_dataframe(final_csv, final_labels)
generate_filter_pass_csv(failure_csv, args.filters)

####################################
####################################
####################################
### initialise PyRosetta
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
print(f"Running binder design for target {settings_file}")
print(f"Design settings used: {advanced_file}")
print(f"Filtering designs based on {filters_file}")
if prefilters_file is not None and advanced_settings.get("pre_filter_trajectory", False):
    print(f"Pre-filtering trajectories based on {prefilters_file}")
elif prefilters_file is None and advanced_settings.get("pre_filter_trajectory", False):
    print("-------------------------------------------------------------------------------------")
    print("Warning: pre_filter_trajectory is enabled but no prefilters file provided, exiting...")
    print("-------------------------------------------------------------------------------------")
    sys.exit(1)

####################################
# initialise counters
script_start_time = time.time()
trajectory_n = 1
accepted_designs = 0

### start design loop
while True:
    ### check if we have the target number of binders
    if advanced_settings["enable_mpnn"]:
        final_designs_reached = check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels)
    else:
        final_designs_reached = check_accepted_designs(design_paths, AF2_design_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels)

    if final_designs_reached:
        # stop design loop execution
        break

    ### check if we reached maximum allowed trajectories
    max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings)

    if max_trajectories_reached:
        break

    ### Initialise design
    # measure time to generate design
    trajectory_start_time = time.time()

    # generate random seed to vary designs
    seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

    # sample binder design length randomly from defined distribution
    samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
    length = np.random.choice(samples)

    # load desired helicity value to sample different secondary structure contents
    helicity_value = load_helicity(advanced_settings)

    # generate design name and check if same trajectory was already run
    design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
    trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
    trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)

    if not trajectory_exists:
        print("Starting trajectory: "+design_name)

        ### Begin binder hallucination
        trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                            target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                            design_models, advanced_settings, design_paths, failure_csv)
        trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"]) # contains plddt, ptm, i_ptm, pae, i_pae
        trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")

        # round the metrics to two decimal places
        trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

        # time trajectory
        trajectory_time = time.time() - trajectory_start_time
        trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
        print("Starting trajectory took: "+trajectory_time_text)
        print("")

        # Proceed if there is no trajectory termination signal
        if trajectory.aux["log"]["terminate"] == "":
            # Relax binder to calculate statistics
            trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
            pr_relax(trajectory_pdb, trajectory_relaxed)

            # define binder chain, placeholder in case multi-chain parsing in ColabDesign gets changed
            binder_chain = "B"

            # Calculate clashes before and after relaxation
            num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
            num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

            # secondary structure content of starting trajectory binder and interface
            trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_i_plddt, trajectory_ss_plddt = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

            # analyze interface scores for relaxed af2 trajectory
            trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain)

            # starting binder sequence
            trajectory_sequence = trajectory.get_seq(get_best=True)[0]

            # analyze sequence
            traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings)

            # target structure RMSD compared to input PDB
            trajectory_target_rmsd = target_pdb_rmsd(trajectory_pdb, target_settings["starting_pdb"], target_settings["chains"])

            # save trajectory statistics into CSV
            trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], trajectory_sequence, trajectory_interface_residues, 
                                trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'], trajectory_metrics['pae'], trajectory_metrics['i_pae'],
                                trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed, trajectory_interface_scores['binder_score'],
                                trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'], trajectory_interface_scores['interface_packstat'],
                                trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'], trajectory_interface_scores['interface_dG_SASA_ratio'],
                                trajectory_interface_scores['interface_fraction'], trajectory_interface_scores['interface_hydrophobicity'], trajectory_interface_scores['interface_nres'], trajectory_interface_scores['interface_interface_hbonds'],
                                trajectory_interface_scores['interface_hbond_percentage'], trajectory_interface_scores['interface_delta_unsat_hbonds'], trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA, trajectory_target_rmsd, 
                                trajectory_time_text, traj_seq_notes, settings_file, filters_file, advanced_file]
            insert_data(trajectory_csv, trajectory_data)

            if not trajectory_interface_residues:
                print("No interface residues found for "+str(design_name)+", skipping MPNN optimization")
                continue

            if advanced_settings.get("direct_trajectory_filtering", False) or advanced_settings.get("pre_filter_trajectory", False) or advanced_settings["enable_mpnn"] == False:
                # Filter the original trajectory sequence directly without MPNN redesign
                if advanced_settings.get("pre_filter_trajectory", False):
                    if prefilters_file is not None:
                        print("Pre-filtering trajectory based on "+str(prefilters_file)+"...")
                        trajectory_filters = prefilters
                        trajectory_filters_file = prefilters_file
                    else:
                        print("No pre-filters file provided, skipping pre-filtering step...")
                        trajectory_filters = None
                        trajectory_filters_file = None
                elif advanced_settings.get("direct_trajectory_filtering", False) or advanced_settings["enable_mpnn"] == False:
                    trajectory_filters = filters
                    trajectory_filters_file = filters_file
                    print("Direct trajectory filtering enabled...")
                else:
                    trajectory_filters = None
                    trajectory_filters_file = None

                if trajectory_filters is not None:
                    complex_prediction_model, binder_prediction_model = init_prediction_models(
                        trajectory_pdb=trajectory_pdb,
                        length=length,
                        mk_afdesign_model=mk_afdesign_model,
                        multimer_validation=multimer_validation,
                        target_settings=target_settings,
                        advanced_settings=advanced_settings
                    )

                    filter_conditions, AF2_design_csv, failure_csv, final_csv = filter_design(
                        sequence=trajectory_sequence,
                        basis_design_name=design_name,
                        design_paths=design_paths,
                        trajectory_pdb=trajectory_pdb,
                        length=length,
                        helicity_value=helicity_value,
                        seed=seed,
                        prediction_models=prediction_models,
                        binder_chain=binder_chain,
                        filters=trajectory_filters,
                        design_labels=design_labels,
                        target_settings=target_settings,
                        advanced_settings=advanced_settings,
                        advanced_file=advanced_file,
                        settings_file=settings_file,
                        filters_file=trajectory_filters_file,
                        stats_csv=AF2_design_csv,
                        failure_csv=failure_csv,
                        final_csv=final_csv,
                        complex_prediction_model=complex_prediction_model,
                        binder_prediction_model=binder_prediction_model,
                        is_mpnn_model=False
                        )                   

            
            # === MPNN SEQUENCE OPTIMIZATION SECTION ===
            # If MPNN is enabled, use ProteinMPNN to generate alternative sequences for the designed backbone
            if advanced_settings["enable_mpnn"]:
                # Initialize counters for tracking MPNN designs
                mpnn_n = 1  # Counter for numbering each MPNN sequence (e.g., _mpnn1, _mpnn2, etc.)
                accepted_mpnn = 0  # Track how many MPNN designs pass all filters for this trajectory
                design_start_time = time.time()  # Start timer for entire MPNN optimization phase

                ### Generate MPNN redesigned sequences based on the trajectory backbone
                # mpnn_gen_sequence: Uses ProteinMPNN to sample new sequences for the fixed backbone
                #   - Fixes interface residues if mpnn_fix_interface=True (only redesigns non-interface)
                #   - Samples num_seqs sequences at specified temperature
                #   - Returns dict with 'seq', 'score', and 'seqid' (sequence identity to original)
                # ToDo: add disulfide cysteins to trajectory_interface_residues to keep them fixed
                mpnn_trajectories = mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings)
                
                # Load all previously accepted sequences from CSV to avoid duplicates across trajectories
                existing_mpnn_sequences = set(pd.read_csv(mpnn_csv, usecols=['Sequence'])['Sequence'].values)

                # Parse restricted amino acids if force_reject_AA is enabled
                # If force_reject_AA=True, sequences containing any omit_AAs will be rejected
                restricted_AAs = set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(',')) if advanced_settings["force_reject_AA"] else set()

                # Filter and deduplicate MPNN sequences, then sort by MPNN score (lower is better)
                # Dictionary comprehension ensures uniqueness by using sequence as key
                # Filters applied:
                #   1. Remove sequences containing restricted amino acids (if force_reject_AA=True)
                #   2. Remove duplicate sequences already in the CSV from previous trajectories
                #   3. Extract only the binder portion (last 'length' residues) from full complex sequence
                mpnn_sequences = sorted({
                    mpnn_trajectories['seq'][n][-length:]: {  # Use binder sequence as dict key for deduplication
                        'seq': mpnn_trajectories['seq'][n][-length:],  # Binder sequence only
                        'score': mpnn_trajectories['score'][n],  # MPNN log probability score
                        'seqid': mpnn_trajectories['seqid'][n]  # Sequence identity to trajectory sequence
                    } for n in range(advanced_settings["num_seqs"])
                    if (not restricted_AAs or not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
                    and mpnn_trajectories['seq'][n][-length:] not in existing_mpnn_sequences
                }.values(), key=lambda x: x['score'])  # Sort by MPNN score (most confident first)

                # Free memory from the large existing sequences set
                del existing_mpnn_sequences
  
                # === CHECK IF SEQUENCES SURVIVED FILTERING ===
                # check whether any sequences are left after amino acid rejection and duplication check, and if yes proceed with prediction
                if mpnn_sequences:
                    # Optimization: Increase AF2 recycles for beta-sheet-rich designs (they need more refinement)
                    # Beta-sheet structures are harder to predict accurately, so we give them more recycles
                    if advanced_settings["optimise_beta"] and float(trajectory_beta) > 15:
                        advanced_settings["num_recycles_validation"] = advanced_settings["optimise_beta_recycles_valid"]

                    ### === COMPILE ALPHAFOLD2 PREDICTION MODELS ===
                    # Pre-compile AF2 models once to avoid recompilation for each sequence (speeds up predictions)
                    clear_mem()  # Clear GPU memory before loading new models
                    
                    # Compile the binder-target complex prediction model
                    # This model predicts how the MPNN sequence folds when bound to the target
                    complex_prediction_model, binder_prediction_model = init_prediction_models(
                        trajectory_pdb=trajectory_pdb,
                        length=length,
                        mk_afdesign_model=mk_afdesign_model,
                        multimer_validation=multimer_validation,
                        target_settings=target_settings,
                        advanced_settings=advanced_settings
                    )

                    # === ITERATE OVER MPNN SEQUENCES FOR VALIDATION ===
                    # For each MPNN-designed sequence, predict its structure and calculate quality metrics
                    for mpnn_sequence in mpnn_sequences:
                       
                        filter_conditions, mpnn_csv, failure_csv, final_csv = filter_design(
                            sequence=mpnn_sequence,
                            basis_design_name=design_name,
                            design_paths=design_paths,
                            trajectory_pdb=trajectory_pdb,
                            length=length,
                            helicity_value=helicity_value,
                            seed=seed,
                            prediction_models=prediction_models,
                            binder_chain=binder_chain,
                            filters=filters,
                            design_labels=design_labels,
                            target_settings=target_settings,
                            advanced_settings=advanced_settings,
                            advanced_file=advanced_file,
                            settings_file=settings_file,
                            filters_file=filters_file,
                            stats_csv=mpnn_csv,
                            failure_csv=failure_csv,
                            final_csv=final_csv,
                            complex_prediction_model=complex_prediction_model,
                            binder_prediction_model=binder_prediction_model,
                            is_mpnn_model=True,
                            mpnn_n=mpnn_n
                            )   
                        
                        # === DESIGN PASSED FILTERS - ACCEPT IT ===
                        if filter_conditions == True:
                            accepted_mpnn += 1  # Increment counter for this trajectory
                            accepted_designs += 1  # Increment global counter
                        
                        # Increment MPNN design counter
                        mpnn_n += 1

                        # === CHECK IF ENOUGH DESIGNS ACCEPTED FROM THIS TRAJECTORY ===
                        # Stop processing more MPNN sequences if we've reached the limit for this trajectory
                        # This prevents wasting compute on a single very good trajectory
                        if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
                            break

                    # === TRAJECTORY MPNN OPTIMIZATION COMPLETE ===
                    if accepted_mpnn >= 1:
                        print("Found "+str(accepted_mpnn)+" MPNN designs passing filters")
                        print("")
                    else:
                        print("No accepted MPNN designs found for this trajectory.")
                        print("")

                else:
                    # No sequences survived filtering (all were duplicates or contained restricted AAs)
                    print('Duplicate MPNN designs sampled with different trajectory, skipping current trajectory optimisation')
                    print("")

                # === CLEANUP: REMOVE UNRELAXED TRAJECTORY PDB ===
                # Save disk space by removing the original unrelaxed trajectory structure
                if advanced_settings["remove_unrelaxed_trajectory"]:
                    os.remove(trajectory_pdb)

                # Calculate and print total time for MPNN optimization of this trajectory
                design_time = time.time() - design_start_time
                design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                print("Design and validation of trajectory "+design_name+" took: "+design_time_text)

            # === ACCEPTANCE RATE MONITORING ===
            # Check if the design process is too inefficient and should be stopped
            # Only starts monitoring after start_monitoring trajectories to allow initial exploration
            if trajectory_n >= advanced_settings["start_monitoring"] and advanced_settings["enable_rejection_check"]:
                acceptance = accepted_designs / trajectory_n  # Calculate success rate
                if not acceptance >= advanced_settings["acceptance_rate"]:
                    # Too many trajectories are failing - likely poor design settings
                    print("The ratio of successful designs is lower than defined acceptance rate! Consider changing your design settings!")
                    print("Script execution stopping...")
                    break

        # Increment trajectory counter and free memory
        trajectory_n += 1
        gc.collect()  # Force garbage collection to free GPU/CPU memory

### Script finished
elapsed_time = time.time() - script_start_time
elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
print("Finished all designs. Script execution for "+str(trajectory_n)+" trajectories took: "+elapsed_text)
