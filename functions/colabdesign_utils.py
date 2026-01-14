####################################
############## ColabDesign functions
####################################
### Import dependencies
import os, re, shutil, math, pickle, random, pprint
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import dynamic_slice
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.shared.utils import copy_dict, categorical
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages, get_chain_length
from .pyrosetta_utils import pr_relax, align_pdbs, pr_staple_disulfides
from . import pyrosetta_utils as pr_utils
from .generic_utils import update_failures
from Bio.PDB import PDBParser
from colabdesign.af.prep import prep_pdb, prep_pos, make_fixed_size, get_multi_id
from colabdesign.af.model import mk_af_model
from colabdesign.af.alphafold.common import residue_constants

# hallucinate a binder
def binder_hallucination(design_name, starting_pdb, protocol, chain, target_hotspot_residues, pos, length, seed, ds_pairs, helicity_value, design_models, advanced_settings, design_paths, failure_csv):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")

    # clear GPU memory for new trajectory
    clear_mem()

    # initialise binder hallucination model
    af_model = mk_af_model_advanced(protocol=protocol, debug=False, data_dir=advanced_settings["af_params_dir"], 
                                use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss')

    # sanity check for hotspots
    if target_hotspot_residues == "":
        target_hotspot_residues = None

    if protocol == "binder":
        af_model.prep_inputs(pdb_filename=starting_pdb, 
                             chain=chain, 
                             binder_len=length, 
                             hotspot=target_hotspot_residues, 
                             seed=seed, 
                             rm_aa=advanced_settings["omit_AAs"],
                            rm_target_seq=advanced_settings["rm_template_seq_design"], 
                            rm_target_sc=advanced_settings["rm_template_sc_design"])
        
    elif protocol == "binder_advanced":

        # === ADVANCED POSITION CONTROL ===
        # Users can specify position control directly via advanced_settings:
        #
        # 1. fixed_positions:     Keep sequence AND structure (e.g., "1-3,10")
        # 2. template_positions:  Keep structure, redesign sequence (e.g., "15-20")
        # 3. sequence_positions:  Keep sequence, allow movement (e.g., "5-8")
        # 4. redesign_positions:  Full freedom (default: all positions not specified above)
        #
        # Disulfide pairs are automatically handled:
        # - Always locked in sequence
        # - Locked in structure if floating=False, allowed to move if floating=True

        # Get user-specified position controls from advanced_settings
        fixed_positions = advanced_settings.get("fixed_positions", None)
        template_positions = advanced_settings.get("template_positions", None)
        sequence_positions = advanced_settings.get("sequence_positions", None)

        # Structural fixation (template CA restraint):
        # If using a template binder scaffold, default-enable a CA restraint when any
        # structure-related positions are specified, unless the user explicitly sets weight 0.
        _sentinel = object()
        _w_ca = advanced_settings.get("weights_template_ca", _sentinel)
        if _w_ca is _sentinel:
            _w_ca = advanced_settings.get("weights_template_ca_rmsd", _sentinel)
        if _w_ca is _sentinel:
            if bool(advanced_settings.get("use_template_binder", False)) and (fixed_positions or template_positions):
                _w_ca = 1.0
                if bool(advanced_settings.get("debug_positions", False)):
                    print("[DEBUG positions] weights_template_ca_rmsd not set; defaulting to 1.0 due to use_template_binder + fixed/template positions")
            else:
                _w_ca = 0.0
        try:
            template_ca_weight = float(_w_ca)
        except Exception:
            template_ca_weight = 0.0
        
        # Handle disulfide pairs automatically
        if ds_pairs and len(ds_pairs) > 0:
            # Build comma-separated string of disulfide positions (1-based)
            ds_pos_str = ','.join([f"{i+1},{j+1}" for (i, j) in ds_pairs])
            
            if advanced_settings.get("floating", False):
                # FLOATING MODE: Disulfides can move but keep sequence
                print(f"Floating mode: Disulfide pairs locked in sequence only (can move)")
                if sequence_positions:
                    sequence_positions = f"{sequence_positions},{ds_pos_str}"
                else:
                    sequence_positions = ds_pos_str
                for (i, j) in ds_pairs:
                    print(f"  Disulfide pair ({i+1}, {j+1}): sequence locked, structure flexible")
            else:
                # FIXED MODE: Disulfides locked in both sequence and structure
                print(f"Fixed mode: Disulfide pairs locked in sequence AND structure")
                if fixed_positions:
                    fixed_positions = f"{fixed_positions},{ds_pos_str}"
                else:
                    fixed_positions = ds_pos_str
                for (i, j) in ds_pairs:
                    print(f"  Disulfide pair ({i+1}, {j+1}): sequence AND structure locked")

        af_model.prep_inputs(
            pdb_filename=starting_pdb,
            target_chain="A",
            binder_chain="B",
            binder_len=length,
            fixed_positions=fixed_positions,
            template_positions=template_positions,
            sequence_positions=sequence_positions,
            seed=seed,
            init_binder_seq=advanced_settings.get("init_binder_seq_runtime"),
            init_binder_locked_map=advanced_settings.get("init_binder_locked_map_runtime"),
            scaffold_pos_map=advanced_settings.get("scaffold_pos_map_runtime"),
            hotspot=target_hotspot_residues,
            rm_target_sc=advanced_settings["rm_template_sc_design"],
            rm_target_seq=advanced_settings["rm_template_seq_design"],
            use_template_binder=bool(advanced_settings.get("use_template_binder", False)),
            debug_positions=bool(advanced_settings.get("debug_positions", False)),
            template_ca_weight=template_ca_weight,
        )      


    ### Update weights based on specified settings
    af_model.opt["weights"].update({"pae":advanced_settings["weights_pae_intra"],
                                    "plddt":advanced_settings["weights_plddt"],
                                    "i_pae":advanced_settings["weights_pae_inter"],
                                    "con":advanced_settings["weights_con_intra"],
                                    "i_con":advanced_settings["weights_con_inter"],
                                    })

    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"],"cutoff":advanced_settings["intra_contact_distance"],"binary":False,"seqsep":9})
    af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"],"cutoff":advanced_settings["inter_contact_distance"],"binary":False})
        

    ### additional loss functions
    if advanced_settings["use_rg_loss"]:
        # radius of gyration loss
        add_rg_loss(af_model, advanced_settings["weights_rg"])

    if advanced_settings["use_i_ptm_loss"]:
        # interface pTM loss
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        # termini distance loss
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])

    # optionally add disulfide promotion loss and seed cysteines
    if advanced_settings.get("use_disulfide_loss", False): # and protocol == "binder":
        # ToDo: Add option to generate terminal disulfide pattern
        # determine disulfide pairs (binder-local 0-based indices)
        if not ds_pairs or len(ds_pairs) == 0:
            # auto-generate pattern
            ds_pairs, sequence_pattern, _ = generate_disulfide_pattern(length, 
                                                                       advanced_settings.get("disulfide_num", 1), 
                                                                       advanced_settings.get("disulfide_min_sep", 5))
        else:
            # keep provided pattern; actual cysteine enforcement is done via bias below
            sequence_pattern = None

        if ds_pairs and len(ds_pairs) > 0:
            # Enforce cysteines at disulfide sites via bias (avoid restart())
            try:
                restype_order = getattr(residue_constants, "restype_order", None)
                if restype_order is None:
                    restype_order = {aa: i for i, aa in enumerate(list("ARNDCQEGHILKMFPSTWYV"))}
                c_idx = restype_order.get("C")
                if c_idx is not None:
                    bias = af_model._inputs.get("bias")
                    if bias is None:
                        bias = np.zeros((1, af_model._len, 20), dtype=np.float32)
                    elif bias.ndim == 2:
                        bias = bias[None, ...]

                    T = int(getattr(af_model, "_target_len", 0))
                    for i, j in ds_pairs:
                        for idx0 in (int(i), int(j)):
                            if 0 <= idx0 < int(length):
                                g = T + idx0
                                bias[0, g, :] = -1e3
                                bias[0, g, c_idx] = 1e3
                    af_model._inputs["bias"] = bias
            except Exception:
                pass
            # store pattern for loss and downstream stapling
            af_model.opt["disulfide_pattern"] = ds_pairs
            # register loss
            add_disulfide_loss(af_model, 
                               weight=advanced_settings.get("weights_disulfide", 1.0), 
                               cutoff=advanced_settings.get("disulfide_cutoff", 7.0))

    if ds_pairs and len(ds_pairs) > 0:
        # add here for case that disulfide loss not used but pairs provided
        advanced_settings["disulfide_pairs_runtime"] = ds_pairs

    # add the helicity loss
    add_helix_loss(af_model, helicity_value)

    ### Additional amino acid composition losses
    if advanced_settings.get("use_polar_bias", False):
        # Polar amino acid bias (S, T, N, Q, Y, C)
        add_polar_bias_loss(af_model, 
                           weight=advanced_settings.get("weights_polar_bias", 0.1),
                           polar_preference=advanced_settings.get("polar_preference", 0.5))
        print(f"Added polar bias loss: weight={advanced_settings.get('weights_polar_bias', 0.1)}, target={advanced_settings.get('polar_preference', 0.5)}")

    if advanced_settings.get("use_charged_bias", False):
        # Charged amino acid bias (D, E, K, R, H)
        add_charged_bias_loss(af_model,
                             weight=advanced_settings.get("weights_charged_bias", 0.1),
                             charged_preference=advanced_settings.get("charged_preference", 0.3))
        print(f"Added charged bias loss: weight={advanced_settings.get('weights_charged_bias', 0.1)}, target={advanced_settings.get('charged_preference', 0.3)}")

    if advanced_settings.get("use_hydrophilic_bias", False):
        # Combined hydrophilic bias (polar + charged)
        add_hydrophilic_bias_loss(af_model,
                                  weight=advanced_settings.get("weights_hydrophilic_bias", 0.1),
                                  hydrophilic_preference=advanced_settings.get("hydrophilic_preference", 0.6))
        print(f"Added hydrophilic bias loss: weight={advanced_settings.get('weights_hydrophilic_bias', 0.1)}, target={advanced_settings.get('hydrophilic_preference', 0.6)}")

    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### start design algorithm based on selection
    if advanced_settings["design_algorithm"] == '2stage':
        # uses gradient descend to get a PSSM profile and then uses PSSM to bias the sampling of random mutations to decrease loss
        af_model.design_pssm_semigreedy(soft_iters=advanced_settings["soft_iterations"], hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

    elif advanced_settings["design_algorithm"] == '3stage':
        # 3 stage design using logits, softmax, and one hot encoding
        af_model.design_3stage(soft_iters=advanced_settings["soft_iterations"], temp_iters=advanced_settings["temporary_iterations"], hard_iters=advanced_settings["hard_iterations"], 
                                num_models=1, models=design_models, sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'greedy':
        # design by using random mutations that decrease loss
        af_model.design_semigreedy(advanced_settings["greedy_iterations"], tries=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'mcmc':
        # design by using random mutations that decrease loss
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == '4stage':
        # initial logits to prescreen trajectory
        print("Stage 1: Test Logits")
        af_model.design_logits(iters=advanced_settings.get("logits_iterations", 50), e_soft=0.9, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)

        # determine pLDDT of best iteration according to lowest 'loss' value
        initial_plddt = get_best_plddt(af_model, length)
        
        # if best iteration has high enough confidence then continue
        if initial_plddt > advanced_settings.get("initial_plddt", 0.65):
            print("Initial trajectory pLDDT good, continuing: "+str(initial_plddt))
            if advanced_settings["optimise_beta"]:
                # temporarily dump model to assess secondary structure
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
                os.remove(model_pdb_path)

                # if beta sheeted trajectory is detected then choose to optimise
                if float(beta) > 15:
                    advanced_settings["soft_iterations"] = advanced_settings["soft_iterations"] + advanced_settings["optimise_beta_extra_soft"]
                    advanced_settings["temporary_iterations"] = advanced_settings["temporary_iterations"] + advanced_settings["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                    print("Beta sheeted trajectory detected, optimising settings")

            # how many logit iterations left
            # logits_iter = advanced_settings["soft_iterations"] - 50
            logits_iter = advanced_settings.get("soft_iterations", 0)
            if logits_iter > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
                                    ramp_recycles=False, save_best=True)
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                print("Optimised logit trajectory pLDDT: "+str(logit_plddt))
            else:
                logit_plddt = initial_plddt

            # perform softmax trajectory design
            if advanced_settings["temporary_iterations"] > 0:
                print("Stage 2: Softmax Optimisation")
                af_model.clear_best()
                af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
                softmax_plddt = get_best_plddt(af_model, length)
            else:
                softmax_plddt = logit_plddt

            # perform one hot encoding
            if softmax_plddt > advanced_settings.get("softmax_plddt", 0.65):
                print("Softmax trajectory pLDDT good, continuing: "+str(softmax_plddt))
                onehot_plddt = softmax_plddt
                if advanced_settings["hard_iterations"] > 0:
                    af_model.clear_best()
                    print("Stage 3: One-hot Optimisation")
                    af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True)
                    onehot_plddt = get_best_plddt(af_model, length)

                if onehot_plddt > advanced_settings.get("onehot_plddt", 0.65):
                    # perform greedy mutation optimisation
                    # ToDo: solve bias issue to enable greedy opt for disulfide generation (maybe reinitialise model with best sequence to avoid bias issues)
                    print("One-hot trajectory pLDDT good, continuing: "+str(onehot_plddt))
                    if advanced_settings["greedy_iterations"] > 0:# and advanced_settings.get("use_disulfide_loss", False) == False:
                        print("Stage 4: PSSM Semigreedy Optimisation")
                        current_seq = af_model.get_seq()[0]
                        af_model.set_seq(seq=current_seq)
                        
                        af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)
                    # elif advanced_settings.get("use_disulfide_loss", False):
                    #     print("Disulfide loss active, skipping semigreedy optimisation")

                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    print("One-hot trajectory pLDDT too low to continue: "+str(onehot_plddt))

            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax trajectory pLDDT too low to continue: "+str(softmax_plddt))

        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial trajectory pLDDT too low to continue: "+str(initial_plddt))

    else:
        print("ERROR: No valid design model selected")
        exit()
        return

    ### save trajectory PDB
    final_plddt = get_best_plddt(af_model, length)
    final_iptm = get_best_iptm(af_model)
    af_model.save_pdb(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    # let's check whether the trajectory is worth optimising by checking confidence, clashes, and contacts
    # check clashes
    #clash_interface = calculate_clash_score(model_pdb_path, 2.4)
    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

    #if clash_interface > 25 or ca_clashes > 0:
    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        update_failures(failure_csv, 'Trajectory_Clashes')
        print("Severe clashes detected, skipping analysis and MPNN optimisation")
        print("")
    else:
        # check if low quality prediction
        if final_plddt < advanced_settings.get("final_plddt", 0.70):
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_final_pLDDT')
            print(f"Trajectory starting confidence low, final pLDDT: {round(final_plddt, 2)}, threshold: {advanced_settings.get('final_plddt', 0.70)}, skipping analysis and MPNN optimisation")
            print("")
        elif final_iptm < advanced_settings.get("final_iptm", 0.50):
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_final_iPTM')
            print(f"Trajectory starting interface confidence low, final iPTM: {round(final_iptm, 2)}, threshold: {advanced_settings.get('final_iptm', 0.50)}, skipping analysis and MPNN optimisation")
            print("")
        else:
            # does it have enough contacts to consider?
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            # if less than 3 contacts then protein is floating above and is not binder
            if binder_contacts_n < 3:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                update_failures(failure_csv, 'Trajectory_Contacts')
                print("Too few contacts at the interface, skipping analysis and MPNN optimisation")
                print("")
            else:
                # phew, trajectory is okay! We can continue
                af_model.aux["log"]["terminate"] = ""
                print("Trajectory successful, final pLDDT: "+str(final_plddt))

    # move low quality prediction:
    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])

    ### get the sampled sequence for plotting
    af_model.get_seqs()
    if advanced_settings["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    ### save the hallucination trajectory animation
    if advanced_settings["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    if advanced_settings["save_trajectory_pickle"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name+".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)


    return af_model

# run prediction for binder with masked template target
def predict_binder_complex(prediction_model, binder_sequence, mpnn_design_name, target_pdb, chain, length, trajectory_pdb, prediction_models, advanced_settings, filters, design_paths, failure_csv, seed=None, design_path_key="MPNN",):
    prediction_stats = {}

    # clean sequence
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    
    # # Clear any existing bias that might have incompatible dimensions
    # # This is critical when the binder sequence length differs from the design trajectory length
    # if "bias" in prediction_model._inputs:
    #     del prediction_model._inputs["bias"]
    
    # Explicitly set the sequence to ensure proper model reinitialization
    # This ensures the model's internal state (including sequence length) is correctly updated
    prediction_model.set_seq(binder_sequence)

    # reset filtering conditionals
    pass_af2_filters = True
    filter_failures = {}

    # start prediction per AF2 model, 2 are used by default due to masked templates
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        complex_pdb = os.path.join(design_paths[design_path_key], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(complex_pdb):
            # predict model (no need to pass seq= since we already called set_seq above)
            prediction_model.predict(models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(complex_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'i_pTM': round(prediction_metrics['i_ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2), 
                'i_pAE': round(prediction_metrics['i_pae'], 2)
            }
            prediction_stats[model_num+1] = stats

            # List of filter conditions and corresponding keys
            filter_conditions = [
                (f"{model_num+1}_pLDDT", 'plddt', '>='),
                (f"{model_num+1}_pTM", 'ptm', '>='),
                (f"{model_num+1}_i_pTM", 'i_ptm', '>='),
                (f"{model_num+1}_pAE", 'pae', '<='),
                (f"{model_num+1}_i_pAE", 'i_pae', '<='),
            ]

            # perform initial AF2 values filtering to determine whether to skip relaxation and interface scoring
            if filters is not None:
                for filter_name, metric_key, comparison in filter_conditions:
                    threshold = filters.get(filter_name, {}).get("threshold")
                    if threshold is not None:
                        if comparison == '>=' and prediction_metrics[metric_key] < threshold:
                            pass_af2_filters = False
                            filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                            print(f"Filter failed: Model {model_num+1} {metric_key} = {prediction_metrics[metric_key]} (threshold: {threshold})")
                        elif comparison == '<=' and prediction_metrics[metric_key] > threshold:
                            pass_af2_filters = False
                            filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                            print(f"Filter failed: Model {model_num+1} {metric_key} = {prediction_metrics[metric_key]} (threshold: {threshold})")

                if not pass_af2_filters:
                    break

    # Update the CSV file with the failure counts
    if filter_failures:
        update_failures(failure_csv, filter_failures)

    # AF2 filters passed, contuing with relaxation
    for model_num in prediction_models:
        complex_pdb = os.path.join(design_paths[design_path_key], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if pass_af2_filters:
            mpnn_relaxed = os.path.join(design_paths[design_path_key + "/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            # pairs = (advanced_settings.get("disulfide_pairs_runtime")
            #          or advanced_settings.get("disulfide_pairs")) if advanced_settings.get("use_disulfide_loss", False) else None
            pairs = advanced_settings.get("disulfide_pairs_runtime") if advanced_settings.get("use_disulfide_loss", False) else None
            pr_relax(complex_pdb, 
                     mpnn_relaxed, 
                     disulfide=advanced_settings.get("use_disulfide_loss", False), 
                     binder_chain="B", 
                     binder_local_pairs=pairs)
            # optional: staple disulfides in the binder chain after relaxation
            # if advanced_settings.get("use_disulfide_loss", False):
            #     binder_chain = advanced_settings.get("binder_chain", "B")
            #     pairs = advanced_settings.get("disulfide_pairs_runtime") or advanced_settings.get("disulfide_pairs")
            #     if pairs:
            #         stapled_out = os.path.join(design_paths[design_path_key + "/Relaxed"], f"{mpnn_design_name}_model{model_num+1}_stapled.pdb")
            #         try:
            #             _staple = getattr(pr_utils, "pr_staple_disulfides", None)
            #             if _staple:
            #                 _staple(mpnn_relaxed, stapled_out, binder_chain, pairs)
            #         except Exception:
            #             pass
        else:
            if os.path.exists(complex_pdb):
                os.remove(complex_pdb)

    return prediction_stats, pass_af2_filters

# run prediction for binder alone
def predict_binder_alone(prediction_model, binder_sequence, mpnn_design_name, length, trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths, seed=None, design_path_key="MPNN"):
    binder_stats = {}

    # prepare sequence for prediction
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    
    # # Clear any existing bias that might have incompatible dimensions
    # if "bias" in prediction_model._inputs:
    #     del prediction_model._inputs["bias"]
    
    prediction_model.set_seq(binder_sequence)

    # predict each model separately
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        binder_alone_pdb = os.path.join(design_paths[design_path_key + "/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(binder_alone_pdb):
            # predict model
            prediction_model.predict(models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(binder_alone_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, pae

            # align binder model to trajectory binder
            align_pdbs(trajectory_pdb, binder_alone_pdb, binder_chain, "A")

            # optional: staple disulfides on binder-alone models
            if advanced_settings.get("use_disulfide_loss", False):
                pairs = advanced_settings.get("disulfide_pairs_runtime") #or advanced_settings.get("disulfide_pairs")
                if pairs:
                    stapled_out = os.path.join(design_paths[design_path_key + "/Binder"], f"{mpnn_design_name}_model{model_num+1}_stapled.pdb")
                    pr_staple_disulfides(binder_alone_pdb, stapled_out, binder_chain, pairs)
                    print(f"Stapled disulfides on binder-alone model {model_num+1}")
                    # try:
                    #     _staple = getattr(pr_utils, "pr_staple_disulfides", None)
                    #     if _staple:
                    #         _staple(binder_alone_pdb, stapled_out, binder_chain, pairs)
                    #         print(f"Stapled disulfides on binder-alone model {model_num+1}")
                    # except Exception:
                    #     pass

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2)
            }
            binder_stats[model_num+1] = stats

    return binder_stats

# run MPNN to generate sequences for binders
def mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, fixed_mpnn_scaffold, advanced_settings):
    # clear GPU memory
    clear_mem()

    # initialise MPNN model
    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"], model_name=advanced_settings["model_path"], weights=advanced_settings["mpnn_weights"])

    # check whether keep the interface generated by the trajectory or whether to redesign with MPNN
    design_chains = 'A,' + binder_chain

    # Build MPNN fixed-position string:
    # - Always include target chain "A"
    # - Always include fixed_mpnn_scaffold (already in "B{resnum}" format)
    # - Optionally include interface residues when mpnn_fix_interface=True
    fixed_tokens = {"A"}

    if fixed_mpnn_scaffold:
        fixed_tokens |= set(fixed_mpnn_scaffold)

    if advanced_settings["mpnn_fix_interface"]:
        if trajectory_interface_residues:
            fixed_tokens |= {x for x in str(trajectory_interface_residues).split(",") if x}
            print("Fixing interface residues: " + str(trajectory_interface_residues))

    # Ensure deterministic ordering (A first, then the rest)
    fixed_positions = "A"
    other_tokens = sorted(t for t in fixed_tokens if t != "A")
    if other_tokens:
        fixed_positions = fixed_positions + "," + ",".join(other_tokens)

    # prepare inputs for MPNN
    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])

    # sample MPNN sequences in parallel
    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"], num=1, batch=advanced_settings["num_seqs"])

    return mpnn_sequences

# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)

# Get iptm of best model
def get_best_iptm(af_model):
    return round(af_model._tmp["best"]["aux"]["i_ptm"],2)

# Define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1):
    '''add radius of gyration loss'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight

# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}
    
    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            idx = inputs["residue_index"].flatten()
            offset = idx[:, None] - idx[None, :]

        # define distogram
        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)
        mask_2d = np.outer(
            np.append(np.zeros(self._target_len), np.ones(self._binder_len)),
            np.append(np.zeros(self._target_len), np.ones(self._binder_len)),
        )

        x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
        if offset is None:
            if mask_2d is None:
                helix_loss = jnp.diagonal(x, 3).mean()
            else:
                helix_loss = jnp.diagonal(x * mask_2d, 3).sum() + (jnp.diagonal(mask_2d, 3).sum() + 1e-8)
        else:
            mask = offset == 3
            # ensure boolean mask
            mask = jnp.asarray(mask, dtype=bool)
            if mask_2d is not None:
                mask2 = jnp.asarray(mask_2d.astype(bool))
                mask = jnp.logical_and(mask2, mask)
            x_arr = jnp.asarray(x)
            x_masked = jnp.where(mask, x_arr, 0.0)
            helix_loss = jnp.sum(x_masked) / (jnp.sum(mask) + 1e-8)

        return {"helix": helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    '''Add loss penalizing the distance between N and C termini'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

# generate disulfide pattern and sequence pattern for binder
def generate_disulfide_pattern(L, disulfide_num, min_sep=5):
    disulfide_pattern = []
    positions = list(range(L))
    trials = 0
    while len(disulfide_pattern) < disulfide_num and trials < 1000:
        trials += 1
        if len(positions) < 2:
            break
        i, j = sorted(random.sample(positions, k=2))
        if abs(i - j) < min_sep:
            continue
        positions.remove(i)
        positions.remove(j)
        disulfide_pattern.append((i, j))
    sequence_pattern = list('X' * L)
    for (i, j) in disulfide_pattern:
        sequence_pattern[i] = 'C'; sequence_pattern[j] = 'C'
    return disulfide_pattern, ''.join(sequence_pattern), L

# add disulfide promotion loss
def add_disulfide_loss(self, weight=1.0, cutoff=7.0):
    def loss_fn(inputs, outputs):
        pairs = self.opt.get("disulfide_pattern", [])
        if pairs is None or len(pairs) == 0:
            return {"disulfide": jnp.array(0.0)}

        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)

        loss_val = 0.0
        n = 0
        for (i, j) in pairs:
            # map binder-local to global indices (target first, binder last)
            gi = int(self._target_len + i)
            gj = int(self._target_len + j)
            # symmetric pair distance histograms (1x1xB)
            dg_ij = dynamic_slice(dgram, (gi, gj, 0), (1, 1, dgram.shape[-1]))
            dg_ji = dynamic_slice(dgram, (gj, gi, 0), (1, 1, dgram.shape[-1]))
            pair_dgram = dg_ij + dg_ji
            # convert to contact loss at cutoff
            x = _get_con_loss(pair_dgram, dgram_bins, cutoff=cutoff, binary=False)
            loss_val += x.sum()
            n += 1

        return {"disulfide": loss_val / (n + 1e-8)}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["disulfide"] = weight
    self.opt["disulfide_cutoff"] = cutoff

# add polar amino acid bias loss
def add_polar_bias_loss(self, weight=0.1, polar_preference=0.5):
    '''
    Add loss to bias amino acid composition toward polar residues.
    
    Polar residues: S, T, N, Q, Y, C (uncharged polar amino acids)
    Uses AlphaFold's standard amino acid ordering: ARNDCQEGHILKMFPSTWYV
    
    Args:
        weight: Loss weight (default 0.1)
        polar_preference: Target fraction of polar residues (0.0-1.0, default 0.5 = 50%)
    '''
    
    # AlphaFold ordering: A R N D C Q E G H I L K M F P S T W Y V
    # Polar amino acids: C(4), N(2), Q(5), S(15), T(16), Y(18) but exclude C to avoid conflict with disulfide loss
    polar_indices = jnp.array([2, 5, 15, 16, 18])
    
    def loss_fn(inputs, outputs):
        if "seq" not in inputs:
            return {"polar": jnp.array(0.0)}
        
        seq_input = inputs["seq"]
        
        # Use logits directly (the actual optimization variable)
        if isinstance(seq_input, dict) and "logits" in seq_input:
            seq_logits = seq_input["logits"]  # Shape: [batch, complex_len, 20] or [complex_len, 20]
        else:
            return {"polar": jnp.array(0.0)}
        
        # Remove batch dimension if present
        if seq_logits.ndim == 3:
            seq_logits = seq_logits[0]
        
        # Convert logits to probabilities
        aa_probs = jax.nn.softmax(seq_logits, axis=-1)  # Shape: [complex_len, 20]
        
        # Extract binder portion (last _binder_len residues)
        binder_probs = aa_probs[-self._binder_len:, :]  # Shape: [binder_len, 20]
        
        # Calculate polar residue probability at each position
        polar_probs = binder_probs[:, polar_indices].sum(axis=-1)  # Shape: [binder_len]
        mean_polar_fraction = polar_probs.mean()
        
        # Loss: softer one-sided penalty
        deviation = mean_polar_fraction - polar_preference
        loss = jnp.square(jax.nn.relu(-deviation))
        
        return {"polar": loss}
    
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["polar"] = weight

# add charged amino acid bias loss
def add_charged_bias_loss(self, weight=0.1, charged_preference=0.3):
    '''
    Add loss to bias amino acid composition toward charged residues.
    
    Charged residues: D, E, K, R, H (acidic and basic amino acids)
    Uses AlphaFold's standard amino acid ordering: ARNDCQEGHILKMFPSTWYV
    
    Args:
        weight: Loss weight (default 0.1)
        charged_preference: Target fraction of charged residues (0.0-1.0, default 0.3 = 30%)
    '''
    
    # AlphaFold ordering: A R N D C Q E G H I L K M F P S T W Y V
    # Charged amino acids: R(1), D(3), E(6), H(8), K(11)
    charged_indices = jnp.array([1, 3, 6, 8, 11])

    def loss_fn(inputs, outputs):
        if "seq" not in inputs:
            return {"charged": jnp.array(0.0)}
        
        seq_input = inputs["seq"]
        
        # Use logits directly (the actual optimization variable)
        if isinstance(seq_input, dict) and "logits" in seq_input:
            seq_logits = seq_input["logits"]
        else:
            return {"charged": jnp.array(0.0)}
        
        # Remove batch dimension if present
        if seq_logits.ndim == 3:
            seq_logits = seq_logits[0]
        
        # Convert logits to probabilities
        aa_probs = jax.nn.softmax(seq_logits, axis=-1)
        
        # Extract binder portion
        binder_probs = aa_probs[-self._binder_len:, :]
        
        # Calculate charged residue probability
        charged_probs = binder_probs[:, charged_indices].sum(axis=-1)
        mean_charged_fraction = charged_probs.mean()
        
        # Loss
        deviation = mean_charged_fraction - charged_preference
        loss = jnp.square(jax.nn.relu(-deviation))
        
        return {"charged": loss}
    
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["charged"] = weight

# add hydrophilic amino acid bias loss
def add_hydrophilic_bias_loss(self, weight=0.1, hydrophilic_preference=0.6):
    '''
    Add loss to bias amino acid composition toward hydrophilic residues.
    
    Hydrophilic = Polar + Charged: C, N, Q, S, T, Y, D, E, H, K, R
    Uses AlphaFold's standard amino acid ordering: ARNDCQEGHILKMFPSTWYV
    
    Args:
        weight: Loss weight (default 0.1)
        hydrophilic_preference: Target fraction of hydrophilic residues (0.0-1.0, default 0.6 = 60%)
    '''
    
    # AlphaFold ordering: A R N D C Q E G H I L K M F P S T W Y V
    # Hydrophilic: R(1), N(2), D(3), C(4), Q(5), E(6), H(8), K(11), S(15), T(16), Y(18) but exclude C to avoid conflict with disulfide loss
    hydrophilic_indices = jnp.array([1, 2, 3, 5, 6, 8, 11, 15, 16, 18])
    
    def loss_fn(inputs, outputs):
        if "seq" not in inputs:
            return {"hydrophilic": jnp.array(0.0)}
        
        seq_input = inputs["seq"]
        
        # Use logits directly (the actual optimization variable)
        if isinstance(seq_input, dict) and "logits" in seq_input:
            seq_logits = seq_input["logits"]
        else:
            return {"hydrophilic": jnp.array(0.0)}
        
        # Remove batch dimension if present
        if seq_logits.ndim == 3:
            seq_logits = seq_logits[0]
        
        # Convert logits to probabilities
        aa_probs = jax.nn.softmax(seq_logits, axis=-1)
        
        # Extract binder portion
        binder_probs = aa_probs[-self._binder_len:, :]
        
        # Calculate hydrophilic residue probability
        hydrophilic_probs = binder_probs[:, hydrophilic_indices].sum(axis=-1)
        mean_hydrophilic_fraction = hydrophilic_probs.mean()
        
        # Loss
        deviation = mean_hydrophilic_fraction - hydrophilic_preference
        loss = jnp.square(jax.nn.relu(-deviation))
        
        return {"hydrophilic": loss}
    
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["hydrophilic"] = weight

# plot design trajectory losses
def plot_trajectory(af_model, design_name, design_paths):
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'NC', 'helix', 'disulfide', 'polar', 'charged', 'hydrophilic', 'mpnn']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # Create a new figure for each metric
            plt.figure()

            loss = af_model.get_loss(metric)
            # Create an x axis for iterations
            iterations = range(1, len(loss) + 1)

            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])

            # Add labels and a legend
            plt.xlabel('Iterations')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)

            # Save the plot
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], design_name+"_"+metric+".png"), dpi=150)
            
            # Close the figure
            plt.close()


# Extended ColabDesign model with binder_advanced protocol
class mk_af_model_advanced(mk_af_model):
    """
    Extended AlphaFold model with custom binder_advanced protocol.
    
    This subclass adds support for position-specific redesign control,
    allowing you to:
    - Fix certain positions (keep original sequence and structure)
    - Template-fix positions (keep structure, allow sequence redesign)
    - Redesign positions (full freedom)
    
    Usage:
        from functions.colabdesign_utils import mk_af_model_advanced
        
        model = mk_af_model_advanced(protocol="binder_advanced", ...)
        model.prep_inputs(
            pdb_filename="complex.pdb",
            target_chain="A",
            binder_chain="B",
            fixed_positions="1-3",
            template_positions="10-15",
            hotspot="50,55-60"
        )
    """
    
    def __init__(self,
                 protocol="fixbb",
                 use_multimer=False,
                 use_templates=False,
                 debug=False,
                 data_dir=".",
                 **kwargs):
        
        # Validate protocol - add our new one to allowed protocols
        if protocol not in ["fixbb", "hallucination", "binder", "partial", "binder_advanced"]:
            raise ValueError(f"protocol must be one of: fixbb, hallucination, binder, partial, binder_advanced")
        
        # Handle binder_advanced as a variant of binder
        if protocol == "binder_advanced":
            # Initialize as standard binder protocol
            super().__init__(protocol="binder",
                           use_multimer=use_multimer,
                           use_templates=use_templates,
                           debug=debug,
                           data_dir=data_dir,
                           **kwargs)
            
            # Override the protocol name
            self.protocol = "binder_advanced"
            
            # Override prep_inputs to use our custom function
            # Create a bound method that passes self as af_model
            def _prep_binder_advanced_method(pdb_filename,
                                             target_chain="A",
                                             binder_chain="B",
                                             fixed_positions=None,
                                             template_positions=None,
                                             sequence_positions=None,
                                             redesign_positions=None,
                                             hotspot=None,
                                             rm_target_seq=False,
                                             rm_target_sc=False,
                                             ignore_missing=True,
                                             **method_kwargs):
                """Bound method wrapper for prep_binder_advanced"""
                return prep_binder_advanced(
                    af_model=self,
                    pdb_filename=pdb_filename,
                    target_chain=target_chain,
                    binder_chain=binder_chain,
                    fixed_positions=fixed_positions,
                    template_positions=template_positions,
                    sequence_positions=sequence_positions,
                    redesign_positions=redesign_positions,
                    hotspot=hotspot,
                    rm_target_seq=rm_target_seq,
                    rm_target_sc=rm_target_sc,
                    ignore_missing=ignore_missing,
                    **method_kwargs
                )
            
            # Assign as instance method
            self.prep_inputs = _prep_binder_advanced_method
            
            # Keep the same loss function as standard binder (already set by parent __init__)
            # self._get_loss is already _loss_binder
            
        else:
            # Standard protocols - just call parent __init__
            super().__init__(protocol=protocol,
                           use_multimer=use_multimer,
                           use_templates=use_templates,
                           debug=debug,
                           data_dir=data_dir,
                           **kwargs)

    def _binder_offset(self, sequence_length):
        """Compute the starting index of the binder segment within a full sequence."""
        binder_len = getattr(self, "_binder_len", sequence_length)
        target_len = getattr(self, "_target_len", 0)
        expected_total = target_len + binder_len

        if sequence_length == binder_len:
            return 0
        if expected_total > 0 and sequence_length == expected_total:
            return target_len
        if sequence_length >= binder_len:
            return max(0, sequence_length - binder_len)
        return 0

    def _apply_seq_grad_mask(self):
        """Zero sequence gradients for positions that must remain fixed.

        Notes:
        - ColabDesign gradients can be JAX arrays; do not rely on in-place mutation.
        - We assign masked arrays back into the grad tree to ensure the optimizer sees it.
        """
        if not hasattr(self, "aux") or "grad" not in self.aux:
            return

        grad_root = self.aux["grad"]
        if not isinstance(grad_root, dict):
            return

        grad_seq = grad_root.get("seq")
        if grad_seq is None:
            return

        mask_candidates = []
        if "seq_grad_mask_binder" in self.opt:
            mask_candidates.append(np.asarray(self.opt["seq_grad_mask_binder"], dtype=np.float32))
        if "seq_grad_mask_global" in self.opt:
            mask_candidates.append(np.asarray(self.opt["seq_grad_mask_global"], dtype=np.float32))

        if not mask_candidates:
            return

        def _pick_mask(arr):
            if arr is None:
                return None
            ndim = getattr(arr, "ndim", None)
            shape = getattr(arr, "shape", None)
            if ndim is None or shape is None:
                return None
            for mask in mask_candidates:
                if mask.ndim != 1:
                    continue
                # grad arrays can be [N, L, 20] or [L, 20] or [L]
                if ndim == 3 and mask.shape[0] == shape[1]:
                    return mask
                if ndim == 2 and mask.shape[0] == shape[0]:
                    return mask
                if ndim == 1 and mask.shape[0] == shape[0]:
                    return mask
            return None

        def _apply_mask(arr):
            mask = _pick_mask(arr)
            if mask is None:
                return arr
            # Prefer JAX for JAX arrays
            try:
                m = jnp.asarray(mask, dtype=getattr(arr, "dtype", jnp.float32))
                if getattr(arr, "ndim", 0) == 3:
                    return arr * m[None, :, None]
                if getattr(arr, "ndim", 0) == 2:
                    return arr * m[:, None]
                if getattr(arr, "ndim", 0) == 1:
                    return arr * m
                return arr
            except Exception:
                m = np.asarray(mask, dtype=np.float32)
                a = np.asarray(arr)
                if a.ndim == 3:
                    return a * m[None, :, None]
                if a.ndim == 2:
                    return a * m[:, None]
                if a.ndim == 1:
                    return a * m
                return a

        if isinstance(grad_seq, dict):
            for key in ("logits", "pseudo", "onehot", "pssm"):
                if key in grad_seq and grad_seq[key] is not None:
                    grad_seq[key] = _apply_mask(grad_seq[key])
            grad_root["seq"] = grad_seq
        else:
            grad_root["seq"] = _apply_mask(grad_seq)

        self.aux["grad"] = grad_root

    def step(self, lr_scale=1.0, num_recycles=None,
             num_models=None, sample_models=None, models=None, backprop=True,
             callback=None, save_best=False, verbose=1):
        """Override step to enforce gradient masking before applying updates."""

        self.run(num_recycles=num_recycles, num_models=num_models, sample_models=sample_models,
                 models=models, backprop=backprop, callback=callback)

        self._apply_seq_grad_mask()

        if self.opt["norm_seq_grad"]:
            self._norm_seq_grad()

        self._state, self.aux["grad"] = self._optimizer(self._state, self.aux["grad"], self._params)

        lr = self.opt["learning_rate"] * lr_scale
        self._params = jax.tree_map(lambda x, g: x - lr * g, self._params, self.aux["grad"])

        self._save_results(save_best=save_best, verbose=verbose)
        self._k += 1

    def _mutate(self, seq, plddt=None, logits=None, mutation_rate=1):
        """Override mutate to sample exclusively from designable binder positions."""
        seq = np.array(seq)
        N, L = seq.shape

        binder_len = getattr(self, "_binder_len", L)
        offset = self._binder_offset(L)
        binder_window = min(binder_len, max(0, L - offset))

        i_prob = np.zeros(L, dtype=np.float32)
        if binder_window > 0:
            i_prob[offset:offset + binder_window] = 1.0

        if plddt is not None:
            binder_plddt = np.maximum(1 - np.array(plddt), 0)
            binder_plddt[np.isnan(binder_plddt)] = 0
            usable = min(binder_window, binder_plddt.shape[-1])
            if usable > 0:
                i_prob[offset:offset + usable] = binder_plddt[:usable]
                if usable < binder_window:
                    remainder_slice = slice(offset + usable, offset + binder_window)
                    i_prob[remainder_slice] = np.maximum(i_prob[remainder_slice], 1e-6)

        i_prob[np.isnan(i_prob)] = 0

        if "fix_pos" in self.opt:
            if "pos" in self.opt:
                p = self.opt["pos"][self.opt["fix_pos"]]
                seq[..., p] = self._wt_aatype_sub
            else:
                p = self.opt["fix_pos"]
                seq[..., p] = self._wt_aatype[..., p]
            i_prob[p] = 0

        binder_designable = np.asarray(self.opt.get("binder_designable", self.opt.get("pos", [])), dtype=int)
        if binder_designable.size > 0:
            allowed_mask = np.zeros(L, dtype=np.float32)
            for idx in binder_designable:
                pos = offset + int(idx)
                if 0 <= pos < L:
                    allowed_mask[pos] = 1.0
            i_prob *= allowed_mask

        forbidden_positions = np.asarray(self.opt.get("forbidden_positions", []), dtype=int)
        if forbidden_positions.size > 0:
            for idx in forbidden_positions:
                pos = offset + int(idx)
                if 0 <= pos < L:
                    i_prob[pos] = 0

        if i_prob.sum() <= 0:
            return seq

        for m in range(mutation_rate):
            total_prob = i_prob.sum()
            if total_prob <= 0:
                break

            probs = i_prob / total_prob
            i = np.random.choice(np.arange(L), p=probs)

            logits_array = np.array(0 if logits is None else logits)
            if logits_array.ndim == 3:
                logits_slice = logits_array[:, i]
            elif logits_array.ndim == 2:
                logits_slice = logits_array[i]
            else:
                logits_slice = logits_array

            alphabet_size = self._args.get("alphabet_size", 20)
            exclusion = np.eye(alphabet_size)[seq[:, i]] * 1e8
            a_logits = logits_slice - exclusion
            a = categorical(softmax(a_logits, axis=-1))

            seq[:, i] = a

        return seq
            
    def save_pdb_sanitized(self, filename, get_best=True, renum_pdb=True, aux=None):
        """
        Saves a PDB file with minimal ATOM records, stripping potential metadata.
        """
        if aux is None:
            aux = self._tmp["best"]["aux"] if (get_best and "aux" in self._tmp["best"]) else self.aux
        
        # Use only the first model's output for a clean, single-model PDB
        aux = jax.tree_map(lambda x: x[0], aux["all"])

        p = {
            "aatype": aux["aatype"],
            "residue_index": aux["residue_index"],
            "atom_positions": aux["atom_positions"],
            "atom_mask": aux["atom_mask"],
            "b_factors": 100 * aux["atom_mask"] * aux["plddt"][..., None]
        }

        # Standard atom names and residue names
        restypes = "ACDEFGHIKLMNPQRSTVWYX"
        res_3_to_1 = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
            "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
            "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y", "UNK": "X"
        }
        res_1_to_3 = {v: k for k, v in res_3_to_1.items()}
        
        atom_types = [
            "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
            "CD1", "CD2", "ND", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2",
            "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH",
            "CZ", "CZ2", "CZ3", "NZ", "OXT"
        ]
        
        pdb_lines = []
        atom_idx = 1
        
        # Determine chain boundaries from _lengths
        chain_breaks = np.cumsum(self._lengths)
        
        for i in range(p["aatype"].shape[0]):  # Iterate through residues
            res_idx = p["residue_index"][i]
            aa_type_idx = p["aatype"][i]
            res_name = res_1_to_3.get(restypes[aa_type_idx], "UNK")
            
            # Determine chain ID
            chain_idx = np.where(i < chain_breaks)[0][0]
            chain_id = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[chain_idx]

            for j in range(p["atom_positions"].shape[1]):  # Iterate through atoms
                if p["atom_mask"][i, j] > 0:
                    atom_name = atom_types[j]
                    x, y, z = p["atom_positions"][i, j]
                    b_factor = p["b_factors"][i, j]
                    
                    line = (
                        f"ATOM  {atom_idx:5d} {atom_name:<4s} {res_name:<3s} {chain_id}{res_idx:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}           {atom_name[0]:>1s}  "
                    )
                    pdb_lines.append(line)
                    atom_idx += 1
        
        with open(filename, 'w') as f:
            f.write("\n".join(pdb_lines))
            f.write("\nEND\n")



# Custom prep function for advanced binder redesign with selective control
def prep_binder_advanced(af_model, pdb_filename, 
                        target_chain="A", 
                        binder_chain="B",
                        binder_len=None,           # For hallucination: length of new binder to create
                        fixed_positions=None,      # Positions to keep original sequence and structure (1-based)
                        template_positions=None,   # Positions to fix in space (1-based)
                        sequence_positions=None,   # Positions to fix sequence (1-based)
                        redesign_positions=None,   # Positions to redesign (if None, redesign all except fixed)
                        hotspot=None,
                        rm_target_seq=False,
                        rm_target_sc=False,
                        ignore_missing=True,
                        seed=None,
                        use_template_binder=False,
                        debug_positions=False,
                        template_ca_weight=0.0,
                        **kwargs):
    """
    Advanced binder prep with fine-grained positional control.
    
    This function follows the same pattern as ColabDesign's _prep_binder method,
    but adds fine-grained control over which positions to fix/redesign.
    
    Supports two modes:
    1. **Redesign mode**: Redesign an existing binder with position control
       - Triggered when binder_chain exists in PDB and binder_len=None
    2. **Hallucination mode**: Create a new binder with position control
       - Triggered when binder_len is specified
    
    Three levels of position control:
    1. Binder redesign (like standard _prep_binder with redesign=True)
    2. Selective sequence fixation (exclude certain positions from redesign)
    3. Selective template fixation (fix certain positions in space)
    
    Args:
        af_model: The AF model instance to prepare
        pdb_filename: Path to PDB file containing target-binder complex
        target_chain: Chain ID(s) of the target protein (comma-separated for multiple)
        binder_chain: Chain ID of the binder to redesign (or create in hallucination mode)
        binder_len: Length of binder to hallucinate (None for redesign mode)
        fixed_positions: Binder positions to keep original sequence (e.g., "1,5-10,15")
                        These positions will NOT be redesigned at all
        template_positions: Binder positions to fix in space (e.g., "1-5,20-25")
                           These positions keep their coordinates but can be redesigned
        sequence_positions: Binder positions to fix sequence only (e.g., "30-40")
                            These positions can move but keep original sequence
        redesign_positions: Binder positions to redesign (e.g., "6-19,26-50")
                           If None, redesigns all positions except fixed_positions
        hotspot: Target hotspot residues (e.g., "10,15-20")
        rm_target_seq: Remove target sequence from template
        rm_target_sc: Remove target sidechain from template
        ignore_missing: Skip positions with missing density
        seed: Random seed for hallucination mode
        **kwargs: Additional arguments passed to _prep_model
        
    Example usage:
        # Redesign existing binder
        prep_binder_advanced(
            model, 
            "complex.pdb",
            target_chain="A",
            binder_chain="B",  # Exists in PDB
            fixed_positions="1-3",      # Keep CYS-CYS-PRO motif
            template_positions="10-15",  # Fix alpha helix in space
            hotspot="50,55-60"
        )
        
        # Hallucinate new binder
        prep_binder_advanced(
            model,
            "target.pdb",
            target_chain="A", 
            binder_chain="B",
            binder_len=50,              # Create new 50-residue binder
            fixed_positions="5,25",     # Fix these positions after creation
            hotspot="50,55-60"
        )
    """
    
    # Parse the PDB structure.
    # In hallucination mode (binder_len provided), we normally do NOT load the existing binder chain.
    # For peptide motif scaffolding we optionally load it (use_template_binder=True) so template
    # coordinates are available for fixed/template positions.
    if binder_len is not None:
        chains_to_load = (
            f"{target_chain},{binder_chain}" if (use_template_binder and binder_chain) else target_chain
        )
    else:
        chains_to_load = f"{target_chain},{binder_chain}" if binder_chain else target_chain
    im = [True] * len(chains_to_load.split(","))
    af_model._pdb = prep_pdb(pdb_filename, chain=chains_to_load, ignore_missing=im)

    if debug_positions:
        print("[DEBUG positions] prep_binder_advanced: chains_to_load:", chains_to_load)
        print("[DEBUG positions] prep_binder_advanced: use_template_binder:", bool(use_template_binder))
        print("[DEBUG positions] prep_binder_advanced: binder_len arg:", binder_len)
    
    # Check if binder exists in PDB (only meaningful when we loaded it)
    available_chains = set(af_model._pdb["idx"]["chain"])
    has_binder_in_pdb = binder_chain in available_chains
    
    # Determine mode: redesign vs hallucination
    if has_binder_in_pdb and binder_len is None:
        # REDESIGN MODE: Existing binder in PDB
        redesign = True
        print(f"🔄 Redesign mode: Using existing binder from chain {binder_chain}")
    elif binder_len is not None:
        # HALLUCINATION MODE: Create new binder
        redesign = False
        print(f"✨ Hallucination mode: Creating new binder of length {binder_len}")
        if has_binder_in_pdb and not use_template_binder:
            print(f"   Note: Ignoring existing chain {binder_chain} in PDB")
        if has_binder_in_pdb and use_template_binder:
            print(f"   Using template binder chain {binder_chain} as structural scaffold")
    else:
        raise ValueError(
            f"Must specify either binder_len (hallucination) or have binder_chain='{binder_chain}' in PDB (redesign)"
        )
    
    # Update protocol-specific args
    af_model._args.update({"redesign": redesign})
    
    # Calculate target and binder lengths
    af_model._target_len = sum([(af_model._pdb["idx"]["chain"] == c).sum() 
                                for c in target_chain.split(",")])
    
    template_binder_len = 0
    if has_binder_in_pdb:
        template_binder_len = int(sum([(af_model._pdb["idx"]["chain"] == c).sum() for c in binder_chain.split(",")]))

    if redesign:
        # REDESIGN MODE: Get binder length from PDB
        af_model._binder_len = sum([(af_model._pdb["idx"]["chain"] == c).sum() 
                                    for c in binder_chain.split(",")])
        res_idx = af_model._pdb["residue_index"]
    else:
        # HALLUCINATION MODE: Use specified length
        if binder_len is None:
            raise ValueError("binder_len must be provided in hallucination mode")
        af_model._binder_len = int(binder_len)

        # Do NOT crop the scaffold. If template binder is loaded and longer than the sampled length,
        # bump binder_len up to the scaffold length.
        if use_template_binder and has_binder_in_pdb and template_binder_len > 0 and int(af_model._binder_len) < int(template_binder_len):
            if debug_positions:
                print(
                    "[DEBUG positions] sampled binder_len",
                    int(af_model._binder_len),
                    "< template_binder_len",
                    int(template_binder_len),
                    "-> using template_binder_len",
                )
            af_model._binder_len = int(template_binder_len)

        if use_template_binder and has_binder_in_pdb and template_binder_len > 0:
            template_keep = int(template_binder_len)

            # Build residue_index using the full scaffold residue_index,
            # and extend sequentially if binder_len exceeds scaffold length.
            target_res_idx = np.asarray(af_model._pdb["residue_index"][:af_model._target_len], dtype=int)
            binder_res_idx_template = np.asarray(
                af_model._pdb["residue_index"][af_model._target_len:af_model._target_len + template_keep],
                dtype=int,
            )
            if int(af_model._binder_len) <= int(template_keep):
                binder_res_idx = binder_res_idx_template[: int(af_model._binder_len)]
            else:
                extra = int(af_model._binder_len) - int(template_keep)
                last = int(binder_res_idx_template[-1]) if binder_res_idx_template.size > 0 else int(target_res_idx[-1]) + 50
                binder_res_idx = np.concatenate([
                    binder_res_idx_template,
                    last + np.arange(1, extra + 1, dtype=int),
                ])
            res_idx = np.concatenate([target_res_idx, binder_res_idx])
        else:
            # Fallback: synthetic binder residue_index with chain break
            target_res_idx = np.asarray(af_model._pdb["residue_index"][:af_model._target_len], dtype=int)
            res_idx = np.append(target_res_idx, target_res_idx[-1] + np.arange(int(af_model._binder_len), dtype=int) + 50)
    
    # CRITICAL: _len must be FULL complex length for proper sequence initialization
    af_model._lengths = [af_model._target_len, af_model._binder_len]
    af_model._len = sum(af_model._lengths)

    # Sanity check: residue_index should match full complex length
    if len(res_idx) != af_model._len:
        raise ValueError(
            f"prep_binder_advanced: residue_index length {len(res_idx)} != full length {af_model._len}. "
            f"target_len={af_model._target_len}, binder_len={af_model._binder_len}, redesign={redesign}"
        )
    
    print(f"Target length: {af_model._target_len}, Binder length: {af_model._binder_len}")

    if debug_positions:
        print("[DEBUG positions] prep_binder_advanced: redesign:", bool(redesign))
        print("[DEBUG positions] prep_binder_advanced: template_binder_len:", int(template_binder_len))
        if not redesign and use_template_binder and template_binder_len > 0:
            if int(af_model._binder_len) > int(template_binder_len):
                print("[DEBUG positions] binder_len exceeds template binder; extension will be hallucinated")
    
    # Gather hotspot info (following _prep_binder pattern)
    if hotspot is not None:
        af_model.opt["hotspot"] = prep_pos(hotspot, **af_model._pdb["idx"])["pos"]
    
    # Extract wild-type binder sequence and set weights based on mode
    if redesign:
        # REDESIGN MODE: Extract existing binder sequence
        af_model._wt_aatype = af_model._pdb["batch"]["aatype"][af_model._target_len:]
        af_model.opt["weights"].update({
            "dgram_cce": 0.1,
            "rmsd": 0.0,
            "fape": 0.0,
            "con": 0.0,
            "i_con": 0.0,
            "i_pae": 0.0
        })
    else:
        # HALLUCINATION MODE: Ensure batch matches full complex length.
        # If template binder is loaded, we already cropped to the overlap above; now we pad to binder_len.
        af_model._pdb["batch"] = make_fixed_size(af_model._pdb["batch"], num_res=sum(af_model._lengths))
        af_model.opt["weights"].update({
            "plddt": 0.1,
            "con": 0.0,
            "i_con": 1.0,
            "i_pae": 0.0
        })
        
        # Initialize binder sequence randomly if seed provided
        if seed is not None:
            af_model.opt["weights"]["soft"] = 0.0
            af_model._binder_seq_init = seed
    
    # === Process position specifications for advanced control ===
    # Convert position strings to arrays (0-based binder-local indices)
    fixed_pos_array = np.array([], dtype=int)
    template_pos_array = np.array([], dtype=int)
    sequence_pos_array = np.array([], dtype=int)
    redesign_pos_array = np.arange(af_model._binder_len)  # Default: all positions
    
    init_binder_seq = kwargs.pop("init_binder_seq", None)
    init_binder_locked_map = kwargs.pop("init_binder_locked_map", None)
    scaffold_pos_map = kwargs.pop("scaffold_pos_map", None)

    if debug_positions:
        print("[DEBUG positions] raw fixed_positions spec:", fixed_positions)
        print("[DEBUG positions] raw template_positions spec:", template_positions)
        print("[DEBUG positions] raw sequence_positions spec:", sequence_positions)
        if init_binder_seq is not None:
            print("[DEBUG positions] init_binder_seq len:", len(str(init_binder_seq)))
        if init_binder_locked_map is not None:
            print("[DEBUG positions] init_binder_locked_map (1-based->AA):", init_binder_locked_map)

    def _aa_idx_to_letter(idx: int) -> str:
        try:
            inv = {v: k for k, v in residue_constants.restype_order.items()}
            return inv.get(int(idx), "?")
        except Exception:
            return "?"

    def _debug_dump_positions(label: str, pos_arr: np.ndarray):
        if not debug_positions:
            return
        T = int(af_model._target_len)
        Lb = int(af_model._binder_len)
        tmpl = int(template_binder_len)
        pos_list = []
        try:
            pos_list = [int(x) for x in np.asarray(pos_arr).tolist()]
        except Exception:
            pos_list = []
        print(f"[DEBUG positions] {label} positions (binder-local 0-based):", sorted(set(pos_list)))
        for p in sorted(set(pos_list)):
            if p < 0 or p >= Lb:
                continue
            g = T + p
            within_template = bool(has_binder_in_pdb and tmpl > 0 and p < tmpl)
            aa_template = None
            try:
                aa_template = _aa_idx_to_letter(int(af_model._pdb["batch"]["aatype"][g]))
            except Exception:
                aa_template = None
            aa_init = None
            if init_binder_seq is not None:
                try:
                    aa_init = str(init_binder_seq)[p]
                except Exception:
                    aa_init = None
            print(
                f"[DEBUG positions]  pos {p+1} (0b={p}) global={g} within_template={within_template} "
                f"templateAA={aa_template} initAA={aa_init}"
            )

    # Helper function to parse positions with chain-aware handling
    def parse_binder_positions(pos_string, binder_chain):
        """
        Parse position string and convert to binder-local 0-based indices.
        Handles both chain-prefixed (e.g., "B116,B117") and simple (e.g., "1,2,3") formats.
        """
        if pos_string is None or pos_string == "":
            return np.array([], dtype=int)

        s = str(pos_string).replace(" ", "")
        if not s:
            return np.array([], dtype=int)

        tokens = [t for t in s.split(",") if t]
        def _looks_chain_aware(token: str) -> bool:
            return len(token) >= 2 and token[0].isalpha() and (token[1].isdigit() or token[1] == "-")

        # Hallucination mode: binder is not reliably addressable in PDB numbering
        # (even if we loaded the template binder). Treat non-prefixed specs as binder-local.
        chain_aware = any(_looks_chain_aware(t.split("-", 1)[0]) for t in tokens)
        if not redesign:
            chain_aware = False

        if chain_aware:
            # Chain-aware: use prep_pos and map global -> binder-local
            pos_dict = prep_pos(s, **af_model._pdb["idx"])
            pos_global = pos_dict["pos"]
            pos_local = np.array(
                [p - af_model._target_len for p in pos_global if p >= af_model._target_len],
                dtype=int,
            )
            pos_local = pos_local[(pos_local >= 0) & (pos_local < int(af_model._binder_len))]
            return np.array(sorted(set(pos_local.tolist())), dtype=int)

        # Binder-local numeric parsing (1-based -> 0-based)
        vals_1b = []
        for part in tokens:
            # strip optional leading chain letter defensively
            if _looks_chain_aware(part):
                part = part[1:]
            if "-" in part[1:]:
                a_str, b_str = part.split("-", 1)
                try:
                    a = int(a_str)
                    b = int(b_str)
                except ValueError:
                    continue
                step = 1 if b >= a else -1
                vals_1b.extend(list(range(a, b + step, step)))
            else:
                try:
                    vals_1b.append(int(part))
                except ValueError:
                    continue

        out = []
        for p1 in vals_1b:
            if p1 <= 0:
                continue
            p0 = p1 - 1
            if 0 <= p0 < int(af_model._binder_len):
                out.append(p0)
        return np.array(sorted(set(out)), dtype=int)
    
    # Parse each position specification
    if fixed_positions:
        fixed_pos_array = parse_binder_positions(fixed_positions, binder_chain)
        print(f"Fixed positions (binder-local 0-based): {fixed_pos_array}")
    
    if template_positions:
        template_pos_array = parse_binder_positions(template_positions, binder_chain)
        print(f"Template-fixed positions (binder-local 0-based): {template_pos_array}")

    if sequence_positions:
        sequence_pos_array = parse_binder_positions(sequence_positions, binder_chain)
        print(f"Sequence-fixed positions (binder-local 0-based): {sequence_pos_array}")

    _debug_dump_positions("fixed(raw)", fixed_pos_array)
    _debug_dump_positions("template(raw)", template_pos_array)
    _debug_dump_positions("sequence(raw)", sequence_pos_array)
    
    # Build mutually-exclusive position sets immediately so we can correctly
    # determine which positions are sequence-designable (template + redesign)
    binder_len_local = int(af_model._binder_len or 0)
    all_binder_pos = set(range(binder_len_local))
    fixed_set = set(fixed_pos_array)
    template_set = set(template_pos_array) - fixed_set
    sequence_set = set(sequence_pos_array) - fixed_set - template_set

    # Determine redesign set: if user provided explicit redesign_positions, honor them
    if redesign_positions:
        # ensure provided redesign positions do not overlap higher-priority sets
        redesign_set = set(redesign_pos_array) - fixed_set - template_set - sequence_set
    else:
        # default: all positions not assigned to fixed/template/sequence
        redesign_set = all_binder_pos - fixed_set - template_set - sequence_set

    # Convert sets back to sorted numpy arrays for downstream bookkeeping
    redesign_pos_array = np.array(sorted(redesign_set), dtype=int)
    template_pos_array = np.array(sorted(template_set), dtype=int)
    sequence_pos_array = np.array(sorted(sequence_set), dtype=int)
    fixed_pos_array = np.array(sorted(fixed_set), dtype=int)

    _debug_dump_positions("fixed(final)", fixed_pos_array)
    _debug_dump_positions("template(final)", template_pos_array)
    _debug_dump_positions("sequence(final)", sequence_pos_array)

    if debug_positions and (not redesign) and use_template_binder and template_binder_len > 0:
        struct_fixed = sorted(set(int(x) for x in fixed_pos_array.tolist()) | set(int(x) for x in template_pos_array.tolist()))
        oob = [p for p in struct_fixed if p >= int(template_binder_len)]
        if oob:
            print(
                "[DEBUG positions] WARNING: structural-fixed positions outside template binder overlap:",
                oob,
                "template_binder_len=", int(template_binder_len),
            )

    print(f"Redesign positions (binder-local): {redesign_pos_array}")

    # === Set up designable positions (CRITICAL for sequence optimization) ===
    # Sequence-designable positions are ONLY: template_positions (keep structure) and redesign_positions (full freedom)
    binder_designable = np.array(sorted(list(redesign_set | template_set)), dtype=int)
    designable_global = af_model._target_len + binder_designable

    # Exclude any positions that must keep sequence (fixed_set or sequence_set)
    forbidden_positions = np.array(sorted(list(fixed_set | sequence_set)), dtype=int)
    if forbidden_positions.size > 0:
        mask = np.isin(binder_designable, forbidden_positions, invert=True)
        binder_designable = binder_designable[mask]
        designable_global = af_model._target_len + binder_designable

    seq_grad_mask_global = np.zeros(sum(af_model._lengths), dtype=np.float32)
    seq_grad_mask_global[designable_global] = 1.0
    seq_grad_mask_binder = np.zeros(int(af_model._binder_len or 0), dtype=np.float32)
    seq_grad_mask_binder[binder_designable] = 1.0

    af_model.opt["pos"] = binder_designable
    af_model.opt["binder_designable"] = binder_designable
    af_model.opt["designable_global"] = designable_global
    af_model.opt["forbidden_positions"] = forbidden_positions
    af_model.opt["seq_grad_mask_global"] = seq_grad_mask_global
    af_model.opt["seq_grad_mask_binder"] = seq_grad_mask_binder

    print(f"Sequence-designable positions (binder-local): {binder_designable}")
    print(f"Sequence-designable positions (global): {designable_global}")

    # Configure input features (following _prep_binder pattern)
    af_model._inputs = af_model._prep_features(num_res=sum(af_model._lengths), num_seq=1)
    af_model._inputs["residue_index"] = res_idx
    af_model._inputs["batch"] = af_model._pdb["batch"]
    af_model._inputs.update(get_multi_id(af_model._lengths))

    # === Configure template masking (following _prep_binder pattern with advanced control) ===
    T = af_model._target_len
    L = sum(af_model._lengths)

    # Initialize template removal masks
    rm = {}
    rm["rm_template"] = np.full(L, False)
    rm["rm_template_seq"] = np.full(L, False)
    rm["rm_template_sc"] = np.full(L, False)

    # Target masking (user-controlled, following _prep_binder pattern)
    rm["rm_template"][:T] = False  # Keep target structure
    rm["rm_template_seq"][:T] = rm_target_seq
    rm["rm_template_sc"][:T] = rm_target_sc
    print(f"Target template masking - remove seq: {rm_target_seq}, remove sc: {rm_target_sc} for target residues 0 to {T-1}")

    # === BINDER MASKING - apply masks according to strict hierarchy ===
    print(f"\n=== Position Control Summary (binder-local 0-based) ===")
    print(f"Fixed (keep seq + struct):      {sorted(fixed_set) if fixed_set else 'None'}")
    print(f"Template (keep struct, change seq): {sorted(template_set) if template_set else 'None'}")
    print(f"Sequence (keep seq, move in space): {sorted(sequence_set) if sequence_set else 'None'}")
    print(f"Redesign (change seq + struct): {sorted(redesign_set) if redesign_set else 'None'}")
    print()

    for pos in range(binder_len_local):
        global_pos = T + pos
        if pos in fixed_set:
            rm["rm_template"][global_pos] = False
            rm["rm_template_seq"][global_pos] = False
            rm["rm_template_sc"][global_pos] = False
        elif pos in template_set:
            rm["rm_template"][global_pos] = False
            rm["rm_template_seq"][global_pos] = True
            rm["rm_template_sc"][global_pos] = True
        elif pos in sequence_set:
            rm["rm_template"][global_pos] = True
            rm["rm_template_seq"][global_pos] = False
            rm["rm_template_sc"][global_pos] = False
        else:
            # redesign positions
            rm["rm_template"][global_pos] = True
            rm["rm_template_seq"][global_pos] = True
            rm["rm_template_sc"][global_pos] = True
    
    # print(rm["rm_template_seq"][:T])
    # print(rm["rm_template_sc"][:T])

    # Set template options (following _prep_binder pattern)
    af_model.opt["template"] = {"rm_ic": False}
    af_model._inputs.update(rm)
    
    # Prepare the model (following _prep_binder pattern - this is the final step)
    af_model._prep_model(**kwargs)

    # CRITICAL FIX: Reapply positional metadata after model prep (restart resets opt)
    preserved_opt_fields = {
        "pos": binder_designable,
        "binder_designable": binder_designable,
        "designable_global": designable_global,
        "forbidden_positions": forbidden_positions,
        "seq_grad_mask_global": seq_grad_mask_global,
        "seq_grad_mask_binder": seq_grad_mask_binder,
    }

    for key, value in preserved_opt_fields.items():
        af_model.opt[key] = value
        if hasattr(af_model, "_opt"):
            af_model._opt[key] = value

    # === CRITICAL FIX: Lock target sequence with extreme bias ===
    # Prevent target sequence from being modified during optimization by applying
    # extreme bias to all target positions. This works by making the softmax
    # over sequence probabilities effectively deterministic for target residues.
    target_aatype = af_model._pdb["batch"]["aatype"][:af_model._target_len]

    existing_bias = af_model._inputs.get("bias")
    if existing_bias is None:
        bias = np.zeros((1, af_model._len, 20), dtype=np.float32)
    else:
        bias = existing_bias
        # Ensure bias has batch dimension (shape should be [1, L, 20])
        if bias.ndim == 2:
            bias = bias[None, ...]

    # Lock ALL target positions with extreme bias (±1000)
    for pos in range(af_model._target_len):
        wt_aa = int(target_aatype[pos])
        bias[0, pos, :] = -1e6  # Extremely strong penalty for all AAs
        bias[0, pos, wt_aa] = 1e6  # Extremely strong preference for wild-type AA

    af_model._inputs["bias"] = bias
    print(f"🔒 Target sequence locked (positions 0-{af_model._target_len-1})")

    # === Apply initial binder sequence / locks (works in redesign and hallucination) ===
    # Use bias seeding to avoid relying on restart(seq=...), which can reset opt fields.
    binder_restype_order = getattr(residue_constants, "restype_order", None)
    if binder_restype_order is None:
        binder_restype_order = {aa: i for i, aa in enumerate(list("ARNDCQEGHILKMFPSTWYV"))}

    if init_binder_seq:
        init_binder_seq = str(init_binder_seq).strip().upper()
        binder_len_int = int(af_model._binder_len or 0)
        if len(init_binder_seq) != binder_len_int:
            print(f"Warning: init_binder_seq length {len(init_binder_seq)} != binder_len {af_model._binder_len}; ignoring init_binder_seq")
        else:
            bias = af_model._inputs.get("bias")
            if bias is None:
                bias = np.zeros((1, af_model._len, 20), dtype=np.float32)
            elif bias.ndim == 2:
                bias = bias[None, ...]

            # Mildly bias every binder position toward the provided initial AA
            init_bias = float(kwargs.pop("init_binder_seq_bias", 5.0))
            T = int(af_model._target_len)
            for pos0, aa in enumerate(init_binder_seq):
                aa_idx = binder_restype_order.get(aa)
                if aa_idx is None:
                    continue
                bias[0, T + pos0, aa_idx] += init_bias

            af_model._inputs["bias"] = bias

    if init_binder_locked_map:
        bias = af_model._inputs.get("bias")
        if bias is None:
            bias = np.zeros((1, af_model._len, 20), dtype=np.float32)
        elif bias.ndim == 2:
            bias = bias[None, ...]

        T = int(af_model._target_len)
        for p1_str, aa in dict(init_binder_locked_map).items():
            try:
                p1 = int(p1_str)
            except Exception:
                continue
            if p1 <= 0:
                continue
            pos0 = p1 - 1
            if not (0 <= pos0 < int(af_model._binder_len or 0)):
                continue
            aa = str(aa).strip().upper()
            aa_idx = binder_restype_order.get(aa)
            if aa_idx is None:
                continue
            global_pos = T + pos0
            bias[0, global_pos, :] = -1e3
            bias[0, global_pos, aa_idx] = 1e3

        af_model._inputs["bias"] = bias
        print(f"🔒 Binder injected/locked residues applied: {len(init_binder_locked_map)}")

    # === Lock binder sequence-fixed positions ===
    # Apply strong bias to binder positions that should keep their sequence
    positions_to_lock = sorted(list(fixed_set | sequence_set))
    
    if len(positions_to_lock) > 0 and (redesign or (use_template_binder and template_binder_len > 0) or (init_binder_seq is not None) or (init_binder_locked_map is not None)):
        # In redesign mode, lock to the PDB wild-type.
        # In hallucination mode, lock to the intended designed sequence (init_binder_locked_map/init_binder_seq)
        # to avoid overwriting injected motifs and disulfide cysteines with template binder residues.
        wt_sequence = af_model._pdb["batch"]["aatype"][af_model._target_len:]

        max_lockable = int(af_model._binder_len or 0)
        # If we don't have an explicit intended sequence for hallucination and are using template binder,
        # restrict locking to the template overlap to avoid locking padded residues to arbitrary aatype.
        if (not redesign) and init_binder_seq is None and not init_binder_locked_map and use_template_binder and template_binder_len > 0:
            max_lockable = min(max_lockable, int(template_binder_len))
        
        # Use the bias matrix already created above
        bias = af_model._inputs["bias"]
        
        # Lock each binder position with extreme bias
        locked_debug = []

        lock_map = {}
        if init_binder_locked_map:
            # normalize keys to int (1-based)
            for k, v in dict(init_binder_locked_map).items():
                try:
                    k_int = int(k)
                except Exception:
                    continue
                lock_map[k_int] = str(v).strip().upper()
        for pos in positions_to_lock:
            if pos < 0 or pos >= max_lockable:
                continue
            global_pos = af_model._target_len + pos
            chosen_aa_idx = None
            chosen_src = None

            # 1) Explicit per-position lock map (1-based)
            aa_letter = lock_map.get(int(pos) + 1)
            if aa_letter is not None:
                chosen_aa_idx = binder_restype_order.get(aa_letter)
                chosen_src = "locked_map"

            # 2) Otherwise, use init_binder_seq if provided (hallucination intended sequence)
            if chosen_aa_idx is None and init_binder_seq is not None:
                try:
                    aa_letter2 = str(init_binder_seq)[int(pos)].upper()
                except Exception:
                    aa_letter2 = None
                if aa_letter2:
                    chosen_aa_idx = binder_restype_order.get(aa_letter2)
                    chosen_src = "init_seq"

            # 3) Fallback: template/PDB wild-type
            if chosen_aa_idx is None:
                chosen_aa_idx = int(wt_sequence[pos])
                chosen_src = "template_wt"

            bias[0, global_pos, :] = -1e3
            bias[0, global_pos, chosen_aa_idx] = 1e3

            if debug_positions:
                locked_debug.append((int(pos), str(chosen_src), _aa_idx_to_letter(int(chosen_aa_idx))))
        
        af_model._inputs["bias"] = bias
        print(f"🔒 Binder sequence-fixed positions locked: {positions_to_lock}")
        if debug_positions:
            print("[DEBUG positions] binder seq-fixed lock AAs (0-based->src->AA):", locked_debug)
            skipped = [p for p in positions_to_lock if p < 0 or p >= max_lockable]
            if skipped:
                print("[DEBUG positions] binder seq-fixed skipped (no template overlap):", skipped)
    
    # Store position information for later reference
    af_model._custom_positions = {
        "fixed": list(fixed_set) if 'fixed_set' in locals() else list(fixed_pos_array),
        "template": list(template_set) if 'template_set' in locals() else list(template_pos_array),
        "sequence": list(sequence_set) if 'sequence_set' in locals() else list(sequence_pos_array),
        "redesign": list(redesign_set) if 'redesign_set' in locals() else list(redesign_pos_array)
    }

    # === Optional: Structural fixation via explicit CA-template restraint ===
    # Template masking alone often does not fully prevent structural drift in hallucination.
    # This adds a differentiable restraint on CA positions for fixed+template residues.
    try:
        template_ca_weight_f = float(template_ca_weight or 0.0)
    except Exception:
        template_ca_weight_f = 0.0

    if template_ca_weight_f > 0:
        # Only meaningful if we have template coordinates loaded.
        if has_binder_in_pdb and template_binder_len > 0:
            struct_fixed_local = sorted(set(fixed_set) | set(template_set))

            # In hallucination mode with a template binder scaffold, ONLY restrain positions that can be
            # referenced back to the ORIGINAL scaffold coordinates (pre-shift). That means:
            # - apply loss at shifted positions (in struct_fixed_local)
            # - reference coords at base positions using scaffold_pos_map (shifted0b -> base0b)
            if (not redesign) and use_template_binder:
                # keep apply positions within current binder length
                struct_fixed_local = [p for p in struct_fixed_local if 0 <= int(p) < int(af_model._binder_len)]

            if len(struct_fixed_local) == 0:
                if debug_positions:
                    print("[DEBUG positions] template_ca restraint: no positions to restrain")
            else:
                batch = af_model._pdb.get("batch", {})
                atom_pos = None
                atom_mask = None
                if isinstance(batch, dict):
                    atom_pos = batch.get("all_atom_positions", None)
                    atom_mask = batch.get("all_atom_mask", None)
                    if atom_pos is None:
                        atom_pos = batch.get("atom_positions", None)
                    if atom_mask is None:
                        atom_mask = batch.get("atom_mask", None)

                if atom_pos is None or atom_mask is None:
                    print(
                        "Warning: template_ca restraint requested but template atom coordinates not found in prep_pdb batch; skipping restraint"
                    )
                else:
                    ca_idx = int(residue_constants.atom_order["CA"])
                    T_int = int(af_model._target_len)

                    # Build aligned (apply_global_idx, ref_global_idx) pairs
                    pairs = []
                    map_dict = scaffold_pos_map if isinstance(scaffold_pos_map, dict) else {}

                    for p_shift in struct_fixed_local:
                        p_shift_i = int(p_shift)
                        # Reference: base position (unshifted scaffold) if provided, else identity
                        p_base_i = int(map_dict.get(p_shift_i, p_shift_i))

                        # Only restrain if we actually have template coordinates for the reference
                        if 0 <= p_base_i < int(template_binder_len):
                            pairs.append((T_int + p_shift_i, T_int + p_base_i))

                    if len(pairs) == 0:
                        if debug_positions:
                            print("[DEBUG positions] template_ca restraint: no valid reference positions in template overlap; skipping")
                    else:
                        apply_global_idx = np.asarray([a for (a, _) in pairs], dtype=int)
                        ref_global_idx = np.asarray([r for (_, r) in pairs], dtype=int)

                        # Filter to positions with CA present in template reference
                        try:
                            ca_present = np.asarray(atom_mask)[ref_global_idx, ca_idx] > 0.5
                        except Exception:
                            ca_present = np.ones((len(ref_global_idx),), dtype=bool)

                        apply_global_idx = apply_global_idx[ca_present]
                        ref_global_idx = ref_global_idx[ca_present]

                        if apply_global_idx.size == 0:
                            if debug_positions:
                                print("[DEBUG positions] template_ca restraint: all reference CA atoms missing; skipping")
                        else:
                            template_ca = np.asarray(atom_pos)[ref_global_idx, ca_idx, :].astype(np.float32)
                            template_ca_mask = np.asarray(atom_mask)[ref_global_idx, ca_idx].astype(np.float32)

                            pos_apply_jnp = jnp.asarray(apply_global_idx.astype(np.int32))
                            template_ca_jnp = jnp.asarray(template_ca)
                            template_ca_mask_jnp = jnp.asarray(template_ca_mask)

                            def _template_ca_loss_fn(inputs, outputs):
                                xyz = outputs["structure_module"]["final_atom_positions"]
                                ca = xyz[:, residue_constants.atom_order["CA"]]
                                pred = ca[pos_apply_jnp]
                                diff = pred - template_ca_jnp
                                per = jnp.sum(diff * diff, axis=-1)
                                w = template_ca_mask_jnp
                                denom = jnp.maximum(jnp.sum(w), 1e-6)
                                loss_val = jnp.sum(per * w) / denom
                                return {"template_ca": loss_val}

                            af_model._callbacks["model"]["loss"].append(_template_ca_loss_fn)
                            af_model.opt["weights"]["template_ca"] = template_ca_weight_f
                            if debug_positions:
                                print(
                                    "[DEBUG positions] template_ca restraint enabled (shifted->base mapping applied):",
                                    "n=", int(apply_global_idx.size),
                                    "weight=", float(template_ca_weight_f),
                                )
                                # Print a small sample mapping for sanity
                                try:
                                    sample_n = int(min(10, apply_global_idx.size))
                                    print(
                                        "[DEBUG positions] template_ca apply_global->ref_global sample:",
                                        list(zip(apply_global_idx[:sample_n].tolist(), ref_global_idx[:sample_n].tolist())),
                                    )
                                except Exception:
                                    pass
        else:
            if debug_positions:
                print("[DEBUG positions] template_ca restraint requested but no template binder loaded; skipping")
    
    print("Advanced binder prep complete!")
    print(f"  Num Fixed sequence positions: {len(fixed_pos_array)}")
    print(f"  Num Template-fixed positions: {len(template_pos_array)}")
    print(f"  Num Redesign positions: {len(redesign_pos_array)}")

    return af_model

