import os
import time
import shutil
import pandas as pd

# Import specific utilities needed by this module
from .colabdesign_utils import predict_binder_complex, predict_binder_alone
from .biopython_utils import calculate_clash_score, validate_design_sequence, target_pdb_rmsd, calc_ss_percentage
from .pyrosetta_utils import score_interface, unaligned_rmsd
from .generic_utils import calculate_averages, check_filters, insert_data, save_fasta


def filter_design(sequence, basis_design_name, design_paths, trajectory_pdb, length, helicity_value,
                  seed, prediction_models, binder_chain, filters, design_labels, target_settings, 
                  advanced_settings, advanced_file, settings_file, filters_file, stats_csv, 
                  failure_csv, final_csv, complex_prediction_model, binder_prediction_model, 
                  is_mpnn_model=False, mpnn_n=None):
    

    start_time = time.time()  # Start timer for this design

    # Initialize variables that will be used later
    interface_residues = ""
    mpnn_score = ""
    mpnn_seqid = ""

    if is_mpnn_model:
        if not isinstance(sequence, dict) or 'seq' not in sequence:
            raise ValueError("For MPNN models, sequence must be a dictionary with at least a 'seq' key.")
        if mpnn_n is None:
            raise ValueError("mpnn_n must be provided when is_mpnn_model is True")
        mpnn_sequence = sequence  # For MPNN, sequence is already a dict with 'seq', 'score', 'seqid'

        # Generate unique name for this MPNN design (e.g., PDL1_l10_s12345_mpnn1)
        design_name = basis_design_name + "_mpnn" + str(mpnn_n)
        mpnn_score = round(mpnn_sequence['score'],2)  # MPNN confidence score
        mpnn_seqid = round(mpnn_sequence['seqid'],2)  # Sequence identity to trajectory

        # Optionally save sequence in FASTA format
        if advanced_settings["save_mpnn_fasta"] is True:
            save_fasta(design_name, mpnn_sequence['seq'], design_paths)
    else:
        if not isinstance(sequence, str):
            raise ValueError("For AF2-only models, sequence must be a string.")
        sequence = {'seq': sequence}  # For AF2-only, wrap sequence string in dict for consistency
        mpnn_score = sequence.get('score', '')  # May be empty string if not provided
        mpnn_seqid = sequence.get('seqid', '')  # May be empty string if not provided
        design_name = basis_design_name  # Use base name directly for AF2-only designs

    ### === PREDICT BINDER-TARGET COMPLEX ===
    # predict_binder_complex: Runs AF2 prediction on MPNN sequence bound to target
    #   - Predicts with multiple AF2 model weights (usually models 1 and 2)
    #   - Applies early AF2-based filters (pLDDT, pTM, iPTM, pAE, etc.)
    #   - Returns statistics dict and pass/fail boolean
    complex_statistics, pass_af2_filters = predict_binder_complex(complex_prediction_model,
                                                                    sequence['seq'], design_name,
                                                                    target_settings["starting_pdb"], target_settings["chains"],
                                                                    length, trajectory_pdb, prediction_models, advanced_settings,
                                                                    filters, design_paths, failure_csv, design_path_key="MPNN" if is_mpnn_model else "AF2")

    # If basic AF2 quality filters failed, skip expensive interface scoring and move to next sequence
    if not pass_af2_filters:
        print(f"Base AF2 filters not passed for {design_name}, skipping interface scoring")
        return False, stats_csv, failure_csv, final_csv

    # === CALCULATE DETAILED INTERFACE METRICS FOR EACH AF2 MODEL ===
    # For each AF2 model weight used in prediction, calculate detailed structural properties
    for model_num in prediction_models:
        # Paths to unrelaxed and relaxed (energy-minimized) structures
        if is_mpnn_model:
            design_pdb = os.path.join(design_paths["MPNN"], f"{design_name}_model{model_num+1}.pdb")
            design_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{design_name}_model{model_num+1}.pdb")
        else:
            design_pdb = os.path.join(design_paths["AF2"], f"{design_name}_model{model_num+1}.pdb")
            design_relaxed = os.path.join(design_paths["AF2/Relaxed"], f"{design_name}_model{model_num+1}.pdb")

        if os.path.exists(design_pdb):
            # Calculate atomic clashes (too-close atoms) before and after energy minimization
            # Clashes indicate structural problems or packing defects
            num_clashes = calculate_clash_score(design_pdb)
            num_clashes_relaxed = calculate_clash_score(design_relaxed)

            # score_interface: PyRosetta-based comprehensive interface analysis
            #   Returns binding energy (dG), shape complementarity, hydrophobicity, hydrogen bonds, etc.
            #   Also returns which amino acids are at the interface
            interface_scores, interface_AA, interface_residues = score_interface(design_relaxed, binder_chain)

            # calc_ss_percentage: Analyze secondary structure content (helix/sheet/loop percentages)
            #   Uses DSSP to assign secondary structure
            #   Returns overall percentages and interface-specific percentages, plus pLDDT values
            alpha, beta, loops, alpha_interface, beta_interface, loops_interface, i_plddt, ss_plddt = calc_ss_percentage(design_pdb, advanced_settings, binder_chain)
            
            # Calculate RMSD of binder to original trajectory (tests if it stayed in designed binding site)
            # Unaligned RMSD measures overall structural similarity without fitting
            rmsd_site = unaligned_rmsd(trajectory_pdb, design_pdb, binder_chain, binder_chain)

            # Calculate how much the target structure moved from its starting conformation
            # Large values indicate the target is being distorted, which is usually bad
            target_rmsd = target_pdb_rmsd(design_pdb, target_settings["starting_pdb"], target_settings["chains"])

            # Add all calculated metrics to the statistics dictionary for this model
            complex_statistics[model_num+1].update({
                'i_pLDDT': i_plddt,  # Average pLDDT of interface residues
                'ss_pLDDT': ss_plddt,  # pLDDT weighted by secondary structure
                'Unrelaxed_Clashes': num_clashes,
                'Relaxed_Clashes': num_clashes_relaxed,
                'Binder_Energy_Score': interface_scores['binder_score'],
                'Surface_Hydrophobicity': interface_scores['surface_hydrophobicity'],
                'ShapeComplementarity': interface_scores['interface_sc'],  # How well surfaces fit (0-1)
                'PackStat': interface_scores['interface_packstat'],  # Packing quality (0-1)
                'dG': interface_scores['interface_dG'],  # Binding energy (negative = favorable)
                'dSASA': interface_scores['interface_dSASA'],  # Buried surface area
                'dG/dSASA': interface_scores['interface_dG_SASA_ratio'],  # Energy per buried area
                'Interface_SASA_%': interface_scores['interface_fraction'],
                'Interface_Hydrophobicity': interface_scores['interface_hydrophobicity'],
                'n_InterfaceResidues': interface_scores['interface_nres'],
                'n_InterfaceHbonds': interface_scores['interface_interface_hbonds'],
                'InterfaceHbondsPercentage': interface_scores['interface_hbond_percentage'],
                'n_InterfaceUnsatHbonds': interface_scores['interface_delta_unsat_hbonds'],  # Unsatisfied H-bonds (bad)
                'InterfaceUnsatHbondsPercentage': interface_scores['interface_delta_unsat_hbonds_percentage'],
                'InterfaceAAs': interface_AA,  # Amino acid composition at interface
                'Interface_Helix%': alpha_interface,
                'Interface_BetaSheet%': beta_interface,
                'Interface_Loop%': loops_interface,
                'Binder_Helix%': alpha,
                'Binder_BetaSheet%': beta,
                'Binder_Loop%': loops,
                'Hotspot_RMSD': rmsd_site,  # Deviation from designed binding site
                'Target_RMSD': target_rmsd  # Target backbone deviation
            })

            # Clean up: Remove unrelaxed PDB to save disk space (keep only relaxed version)
            if advanced_settings["remove_unrelaxed_complex"]:
                os.remove(design_pdb)

    # === CALCULATE AVERAGE METRICS ACROSS ALL AF2 MODELS ===
    # calculate_averages: Averages metrics from all predicted models (typically 2 models)
    #   Provides consensus estimate of binder quality
    #   handle_aa=True means it also averages amino acid composition
    complex_averages = calculate_averages(complex_statistics, handle_aa=True)
    
    ### === PREDICT BINDER ALONE (WITHOUT TARGET) ===
    # Tests if binder is stable on its own or requires target to fold properly
    # predict_binder_alone: Runs AF2 in hallucination mode (no template) with just binder sequence
    #   Returns structure quality metrics (pLDDT, pTM, pAE)
    binder_statistics = predict_binder_alone(binder_prediction_model, sequence['seq'], design_name, length,
                                            trajectory_pdb, binder_chain, prediction_models, advanced_settings, 
                                            design_paths, design_path_key="MPNN" if is_mpnn_model else "AF2")

    # === CALCULATE BINDER-ALONE RMSD TO TRAJECTORY ===
    # Check how much the binder structure changes when target is removed
    # Large RMSD indicates binder is not stable alone (requires target to fold)
    for model_num in prediction_models:
        if is_mpnn_model:
            binder_pdb = os.path.join(design_paths["MPNN/Binder"], f"{design_name}_model{model_num+1}.pdb")
        else:
            binder_pdb = os.path.join(design_paths["AF2/Binder"], f"{design_name}_model{model_num+1}.pdb")

        rmsd_binder = None
        if os.path.exists(binder_pdb):
            # Compare standalone binder structure to trajectory binder structure
            rmsd_binder = unaligned_rmsd(trajectory_pdb, binder_pdb, binder_chain, "A")

        # Add RMSD to statistics
        if rmsd_binder is not None:
            binder_statistics[model_num+1].update({
                    'Binder_RMSD': rmsd_binder
                })
        else:
            raise ValueError(f"RMSD calculation failed for {design_name} model {model_num+1}")

        # Clean up: Remove binder monomer PDBs to save space
        if advanced_settings["remove_binder_monomer"]:
            os.remove(binder_pdb)

    # Calculate average binder-alone metrics across models
    binder_averages = calculate_averages(binder_statistics)

    # === SEQUENCE VALIDATION ===
    # validate_design_sequence: Checks sequence for potential experimental issues:
    #   - Contains residues that absorb UV (for concentration measurement: W, Y, F)
    #   - Avoids cysteines if specified (can form unwanted disulfides)
    #   - Checks for unusual amino acid patterns
    seq_notes = validate_design_sequence(sequence['seq'], complex_averages.get('Relaxed_Clashes', None), advanced_settings)

    # Calculate time spent on this MPNN design
    end_time = time.time() - start_time
    elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(end_time // 3600), int((end_time % 3600) // 60), int(end_time % 60))}"


    # === PREPARE DATA FOR CSV OUTPUT ===
    # Build a comprehensive row of data with averages and individual model results
    # Insert statistics about design into CSV, will return None if corresponding model does note exist
    model_numbers = range(1, 6)  # Support up to 5 AF2 models (usually only 2 are used)
    statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                        'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                        'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                        'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

    # Start with basic design metadata
    data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], sequence['seq'], interface_residues, mpnn_score, mpnn_seqid]

    # Add complex statistics: first average, then individual models (1-5)
    for label in statistics_labels:
        data.append(complex_averages.get(label, None))  # Average value
        for model in model_numbers:
            data.append(complex_statistics.get(model, {}).get(label, None))  # Model-specific value

    # Add binder-alone statistics: average + individual models
    for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:  # These are the labels for binder alone
        data.append(binder_averages.get(label, None))
        for model in model_numbers:
            data.append(binder_statistics.get(model, {}).get(label, None))

    # Add metadata: timing, sequence notes, settings used
    data.extend([elapsed_text, seq_notes, settings_file, filters_file, advanced_file])

    # Write data row to CSV
    insert_data(stats_csv, data)

    # === SELECT BEST MODEL BY PLDDT ===
    # Find which AF2 model produced the highest-confidence prediction
    # CSV indices 11-14 contain pLDDT values for models 1-4 (after Average_pLDDT at index 10)
    plddt_values = {i: data[i] for i in range(11, 15) if data[i] is not None}

    # Find the index with the highest pLDDT value
    highest_plddt_key = int(max(plddt_values, key=plddt_values.get))

    # Convert index to model number (11->1, 12->2, etc.)
    best_model_number = highest_plddt_key - 10
    if is_mpnn_model:
        best_model_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{design_name}_model{best_model_number}.pdb")
    else:
        best_model_pdb = os.path.join(design_paths["AF2/Relaxed"], f"{design_name}_model{best_model_number}.pdb")

    # === APPLY QUALITY FILTERS ===
    # check_filters: Compares all calculated metrics against user-defined thresholds
    #   Returns True if all filters passed, or list of failed filters
    #   Filters can include thresholds for pLDDT, binding energy, clashes, etc.
    filter_conditions = check_filters(data, design_labels, filters)


    if filter_conditions == True:
        # === DESIGN PASSED ALL FILTERS - ACCEPT IT ===
        print(design_name+" passed all filters")
        
        # Copy best model to Accepted folder for easy access
        shutil.copy(best_model_pdb, design_paths["Accepted"])

        # Add to final designs CSV (with empty first column for notes/ranking)
        final_data = [''] + data
        insert_data(final_csv, final_data)

        # Copy trajectory animation to Accepted folder (if enabled and not already copied)
        if advanced_settings["save_design_animations"]:
            accepted_animation = os.path.join(design_paths["Accepted/Animation"], f"{basis_design_name}.html")
            if not os.path.exists(accepted_animation):
                shutil.copy(os.path.join(design_paths["Trajectory/Animation"], f"{basis_design_name}.html"), accepted_animation)

        # Copy trajectory loss plots to Accepted folder
        plot_files = os.listdir(design_paths["Trajectory/Plots"])
        plots_to_copy = [f for f in plot_files if f.startswith(basis_design_name) and f.endswith('.png')]
        for accepted_plot in plots_to_copy:
            source_plot = os.path.join(design_paths["Trajectory/Plots"], accepted_plot)
            target_plot = os.path.join(design_paths["Accepted/Plots"], accepted_plot)
            if not os.path.exists(target_plot):
                shutil.copy(source_plot, target_plot)

    else:
        # === DESIGN FAILED FILTERS - REJECT IT ===
        print(f"Unmet filter conditions for {basis_design_name}")
        
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


    return filter_conditions, stats_csv, failure_csv, final_csv


def init_prediction_models(trajectory_pdb, length, mk_afdesign_model, multimer_validation, target_settings, advanced_settings):
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

    return complex_prediction_model, binder_prediction_model

