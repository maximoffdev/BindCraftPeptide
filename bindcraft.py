####################################
###################### BindCraft Run
####################################
### Import dependencies
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

args = parser.parse_args()

# perform checks of input setting files
settings_path, filters_path, advanced_path = perform_input_check(args)

### load settings from JSON
target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)

settings_file = os.path.basename(settings_path).split('.')[0]
filters_file = os.path.basename(filters_path).split('.')[0]
advanced_file = os.path.basename(advanced_path).split('.')[0]

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
final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

create_dataframe(trajectory_csv, trajectory_labels)
create_dataframe(mpnn_csv, design_labels)
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

####################################
# initialise counters
script_start_time = time.time()
trajectory_n = 1
accepted_designs = 0

### start design loop
while True:
    ### check if we have the target number of binders
    final_designs_reached = check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels)

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
            
            # === MPNN SEQUENCE OPTIMIZATION SECTION ===
            # If MPNN is enabled, use ProteinMPNN to generate alternative sequences for the designed backbone
            if advanced_settings["enable_mpnn"]:
                # Initialize counters for tracking MPNN designs
                mpnn_n = 1  # Counter for numbering each MPNN sequence (e.g., _mpnn1, _mpnn2, etc.)
                accepted_mpnn = 0  # Track how many MPNN designs pass all filters for this trajectory
                mpnn_dict = {}  # Dictionary to store MPNN sequence info (seq, score, seqid)
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
                    complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=advanced_settings["num_recycles_validation"], data_dir=advanced_settings["af_params_dir"], 
                                                                use_multimer=multimer_validation, use_initial_guess=advanced_settings["predict_initial_guess"], use_initial_atom_pos=advanced_settings["predict_bigbang"])
                    
                    # Choose template mode: use trajectory structure as template OR use original target structure
                    if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
                        # Use the designed trajectory structure as starting point (biases prediction toward designed pose)
                        complex_prediction_model.prep_inputs(pdb_filename=trajectory_pdb, chain='A', binder_chain='B', binder_len=length, use_binder_template=True, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                            rm_target_sc=advanced_settings["rm_template_sc_predict"], rm_template_ic=True)
                    else:
                        # Use only the original target structure (unbiased prediction - tests if sequence really folds to bind)
                        complex_prediction_model.prep_inputs(pdb_filename=target_settings["starting_pdb"], chain=target_settings["chains"], binder_len=length, rm_target_seq=advanced_settings["rm_template_seq_predict"],
                                                            rm_target_sc=advanced_settings["rm_template_sc_predict"])

                    # Compile binder-alone prediction model (tests if binder is stable without target)
                    # This checks for stability and detects if the binder requires the target to fold
                    binder_prediction_model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False, 
                                                                use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"], 
                                                                data_dir=advanced_settings["af_params_dir"], use_multimer=multimer_validation)
                    binder_prediction_model.prep_inputs(length=length)

                    # === ITERATE OVER MPNN SEQUENCES FOR VALIDATION ===
                    # For each MPNN-designed sequence, predict its structure and calculate quality metrics
                    for mpnn_sequence in mpnn_sequences:
                        mpnn_time = time.time()  # Start timer for this MPNN design

                        # Generate unique name for this MPNN design (e.g., PDL1_l10_s12345_mpnn1)
                        mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)
                        mpnn_score = round(mpnn_sequence['score'],2)  # MPNN confidence score
                        mpnn_seqid = round(mpnn_sequence['seqid'],2)  # Sequence identity to trajectory

                        # Store design info in dictionary for potential later use
                        mpnn_dict[mpnn_design_name] = {'seq': mpnn_sequence['seq'], 'score': mpnn_score, 'seqid': mpnn_seqid}

                        # Optionally save sequence in FASTA format
                        if advanced_settings["save_mpnn_fasta"] is True:
                            save_fasta(mpnn_design_name, mpnn_sequence['seq'], design_paths)
                        
                        ### === PREDICT BINDER-TARGET COMPLEX ===
                        # predict_binder_complex: Runs AF2 prediction on MPNN sequence bound to target
                        #   - Predicts with multiple AF2 model weights (usually models 1 and 2)
                        #   - Applies early AF2-based filters (pLDDT, pTM, iPTM, pAE, etc.)
                        #   - Returns statistics dict and pass/fail boolean
                        mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(complex_prediction_model,
                                                                                        mpnn_sequence['seq'], mpnn_design_name,
                                                                                        target_settings["starting_pdb"], target_settings["chains"],
                                                                                        length, trajectory_pdb, prediction_models, advanced_settings,
                                                                                        filters, design_paths, failure_csv)

                        # If basic AF2 quality filters failed, skip expensive interface scoring and move to next sequence
                        if not pass_af2_filters:
                            print(f"Base AF2 filters not passed for {mpnn_design_name}, skipping interface scoring")
                            mpnn_n += 1
                            continue

                        # === CALCULATE DETAILED INTERFACE METRICS FOR EACH AF2 MODEL ===
                        # For each AF2 model weight used in prediction, calculate detailed structural properties
                        for model_num in prediction_models:
                            # Paths to unrelaxed and relaxed (energy-minimized) structures
                            mpnn_design_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                            mpnn_design_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                            if os.path.exists(mpnn_design_pdb):
                                # Calculate atomic clashes (too-close atoms) before and after energy minimization
                                # Clashes indicate structural problems or packing defects
                                num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                                num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)

                                # score_interface: PyRosetta-based comprehensive interface analysis
                                #   Returns binding energy (dG), shape complementarity, hydrophobicity, hydrogen bonds, etc.
                                #   Also returns which amino acids are at the interface
                                mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_design_relaxed, binder_chain)

                                # calc_ss_percentage: Analyze secondary structure content (helix/sheet/loop percentages)
                                #   Uses DSSP to assign secondary structure
                                #   Returns overall percentages and interface-specific percentages, plus pLDDT values
                                mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt, mpnn_ss_plddt = calc_ss_percentage(mpnn_design_pdb, advanced_settings, binder_chain)
                                
                                # Calculate RMSD of binder to original trajectory (tests if it stayed in designed binding site)
                                # Unaligned RMSD measures overall structural similarity without fitting
                                rmsd_site = unaligned_rmsd(trajectory_pdb, mpnn_design_pdb, binder_chain, binder_chain)

                                # Calculate how much the target structure moved from its starting conformation
                                # Large values indicate the target is being distorted, which is usually bad
                                target_rmsd = target_pdb_rmsd(mpnn_design_pdb, target_settings["starting_pdb"], target_settings["chains"])

                                # Add all calculated metrics to the statistics dictionary for this model
                                mpnn_complex_statistics[model_num+1].update({
                                    'i_pLDDT': mpnn_i_plddt,  # Average pLDDT of interface residues
                                    'ss_pLDDT': mpnn_ss_plddt,  # pLDDT weighted by secondary structure
                                    'Unrelaxed_Clashes': num_clashes_mpnn,
                                    'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                                    'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                                    'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                                    'ShapeComplementarity': mpnn_interface_scores['interface_sc'],  # How well surfaces fit (0-1)
                                    'PackStat': mpnn_interface_scores['interface_packstat'],  # Packing quality (0-1)
                                    'dG': mpnn_interface_scores['interface_dG'],  # Binding energy (negative = favorable)
                                    'dSASA': mpnn_interface_scores['interface_dSASA'],  # Buried surface area
                                    'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],  # Energy per buried area
                                    'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                                    'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                                    'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                                    'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                                    'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                                    'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],  # Unsatisfied H-bonds (bad)
                                    'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                                    'InterfaceAAs': mpnn_interface_AA,  # Amino acid composition at interface
                                    'Interface_Helix%': mpnn_alpha_interface,
                                    'Interface_BetaSheet%': mpnn_beta_interface,
                                    'Interface_Loop%': mpnn_loops_interface,
                                    'Binder_Helix%': mpnn_alpha,
                                    'Binder_BetaSheet%': mpnn_beta,
                                    'Binder_Loop%': mpnn_loops,
                                    'Hotspot_RMSD': rmsd_site,  # Deviation from designed binding site
                                    'Target_RMSD': target_rmsd  # Target backbone deviation
                                })

                                # Clean up: Remove unrelaxed PDB to save disk space (keep only relaxed version)
                                if advanced_settings["remove_unrelaxed_complex"]:
                                    os.remove(mpnn_design_pdb)

                        # === CALCULATE AVERAGE METRICS ACROSS ALL AF2 MODELS ===
                        # calculate_averages: Averages metrics from all predicted models (typically 2 models)
                        #   Provides consensus estimate of binder quality
                        #   handle_aa=True means it also averages amino acid composition
                        mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)
                        
                        ### === PREDICT BINDER ALONE (WITHOUT TARGET) ===
                        # Tests if binder is stable on its own or requires target to fold properly
                        # predict_binder_alone: Runs AF2 in hallucination mode (no template) with just binder sequence
                        #   Returns structure quality metrics (pLDDT, pTM, pAE)
                        binder_statistics = predict_binder_alone(binder_prediction_model, mpnn_sequence['seq'], mpnn_design_name, length,
                                                                trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths)

                        # === CALCULATE BINDER-ALONE RMSD TO TRAJECTORY ===
                        # Check how much the binder structure changes when target is removed
                        # Large RMSD indicates binder is not stable alone (requires target to fold)
                        for model_num in prediction_models:
                            mpnn_binder_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")

                            if os.path.exists(mpnn_binder_pdb):
                                # Compare standalone binder structure to trajectory binder structure
                                rmsd_binder = unaligned_rmsd(trajectory_pdb, mpnn_binder_pdb, binder_chain, "A")

                            # Add RMSD to statistics
                            binder_statistics[model_num+1].update({
                                    'Binder_RMSD': rmsd_binder
                                })

                            # Clean up: Remove binder monomer PDBs to save space
                            if advanced_settings["remove_binder_monomer"]:
                                os.remove(mpnn_binder_pdb)

                        # Calculate average binder-alone metrics across models
                        binder_averages = calculate_averages(binder_statistics)

                        # === SEQUENCE VALIDATION ===
                        # validate_design_sequence: Checks sequence for potential experimental issues:
                        #   - Contains residues that absorb UV (for concentration measurement: W, Y, F)
                        #   - Avoids cysteines if specified (can form unwanted disulfides)
                        #   - Checks for unusual amino acid patterns
                        seq_notes = validate_design_sequence(mpnn_sequence['seq'], mpnn_complex_averages.get('Relaxed_Clashes', None), advanced_settings)

                        # Calculate time spent on this MPNN design
                        mpnn_end_time = time.time() - mpnn_time
                        elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"


                        # === PREPARE DATA FOR CSV OUTPUT ===
                        # Build a comprehensive row of data with averages and individual model results
                        # Insert statistics about MPNN design into CSV, will return None if corresponding model does note exist
                        model_numbers = range(1, 6)  # Support up to 5 AF2 models (usually only 2 are used)
                        statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                                            'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                                            'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                                            'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

                        # Start with basic design metadata
                        mpnn_data = [mpnn_design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], mpnn_sequence['seq'], mpnn_interface_residues, mpnn_score, mpnn_seqid]

                        # Add complex statistics: first average, then individual models (1-5)
                        for label in statistics_labels:
                            mpnn_data.append(mpnn_complex_averages.get(label, None))  # Average value
                            for model in model_numbers:
                                mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))  # Model-specific value

                        # Add binder-alone statistics: average + individual models
                        for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:  # These are the labels for binder alone
                            mpnn_data.append(binder_averages.get(label, None))
                            for model in model_numbers:
                                mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

                        # Add metadata: timing, sequence notes, settings used
                        mpnn_data.extend([elapsed_mpnn_text, seq_notes, settings_file, filters_file, advanced_file])

                        # Write data row to CSV
                        insert_data(mpnn_csv, mpnn_data)

                        # === SELECT BEST MODEL BY PLDDT ===
                        # Find which AF2 model produced the highest-confidence prediction
                        # CSV indices 11-14 contain pLDDT values for models 1-4 (after Average_pLDDT at index 10)
                        plddt_values = {i: mpnn_data[i] for i in range(11, 15) if mpnn_data[i] is not None}

                        # Find the index with the highest pLDDT value
                        highest_plddt_key = int(max(plddt_values, key=plddt_values.get))

                        # Convert index to model number (11->1, 12->2, etc.)
                        best_model_number = highest_plddt_key - 10
                        best_model_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{best_model_number}.pdb")

                        # === APPLY QUALITY FILTERS ===
                        # check_filters: Compares all calculated metrics against user-defined thresholds
                        #   Returns True if all filters passed, or list of failed filters
                        #   Filters can include thresholds for pLDDT, binding energy, clashes, etc.
                        filter_conditions = check_filters(mpnn_data, design_labels, filters)
                        if filter_conditions == True:
                            # === DESIGN PASSED ALL FILTERS - ACCEPT IT ===
                            print(mpnn_design_name+" passed all filters")
                            accepted_mpnn += 1  # Increment counter for this trajectory
                            accepted_designs += 1  # Increment global counter
                            
                            # Copy best model to Accepted folder for easy access
                            shutil.copy(best_model_pdb, design_paths["Accepted"])

                            # Add to final designs CSV (with empty first column for notes/ranking)
                            final_data = [''] + mpnn_data
                            insert_data(final_csv, final_data)

                            # Copy trajectory animation to Accepted folder (if enabled and not already copied)
                            if advanced_settings["save_design_animations"]:
                                accepted_animation = os.path.join(design_paths["Accepted/Animation"], f"{design_name}.html")
                                if not os.path.exists(accepted_animation):
                                    shutil.copy(os.path.join(design_paths["Trajectory/Animation"], f"{design_name}.html"), accepted_animation)

                            # Copy trajectory loss plots to Accepted folder
                            plot_files = os.listdir(design_paths["Trajectory/Plots"])
                            plots_to_copy = [f for f in plot_files if f.startswith(design_name) and f.endswith('.png')]
                            for accepted_plot in plots_to_copy:
                                source_plot = os.path.join(design_paths["Trajectory/Plots"], accepted_plot)
                                target_plot = os.path.join(design_paths["Accepted/Plots"], accepted_plot)
                                if not os.path.exists(target_plot):
                                    shutil.copy(source_plot, target_plot)

                        else:
                            # === DESIGN FAILED FILTERS - REJECT IT ===
                            print(f"Unmet filter conditions for {mpnn_design_name}")
                            
                            # Update failure statistics CSV to track which filters are failing most often
                            failure_df = pd.read_csv(failure_csv)
                            
                            # Handle filter column names that may have model-specific prefixes (Average_, 1_, 2_, etc.)
                            special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
                            incremented_columns = set()  # Track which base columns we've already incremented

                            # For each failed filter, increment its failure count (only once per base metric)
                            for column in filter_conditions:
                                base_column = column
                                # Strip model prefix to get base metric name
                                for prefix in special_prefixes:
                                    if column.startswith(prefix):
                                        base_column = column.split('_', 1)[1]

                                # Only increment each base metric once (even if multiple models failed)
                                if base_column not in incremented_columns:
                                    failure_df[base_column] = failure_df[base_column] + 1
                                    incremented_columns.add(base_column)

                            # Save updated failure counts
                            failure_df.to_csv(failure_csv, index=False)
                            
                            # Move rejected design to Rejected folder for later review
                            shutil.copy(best_model_pdb, design_paths["Rejected"])
                        
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
