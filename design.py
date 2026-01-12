#!/usr/bin/env python3
"""Design pipeline: AF2 design + optional MPNN redesign.
Outputs trajectory PDBs and metrics CSVs (trajectory_stats.csv, mpnn_design_stats.csv, AF2_design_stats.csv, final_design_stats.csv, failure_csv.csv).
This is a split-out version of the original bindcraft.py focused on the design stage only.
"""
import os
import sys
import argparse
import time
import numpy as np
from functions import *


def extract_chain_sequence_from_pdb(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"Chain '{chain_id}' not found in {pdb_path}")
    chain = model[chain_id]
    seq = []
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        seq.append(three_to_one_map.get(residue.get_resname(), 'X'))
    return ''.join(seq)


def main():
    # Ensure GPU is available (halts otherwise)
    check_jax_gpu()

    parser = argparse.ArgumentParser(description="BindCraft design stage (AF2 + optional MPNN)")
    parser.add_argument("--settings", "-s", required=True, help="Path to basic settings.json")
    parser.add_argument("--filters", "-f", default="./settings_filters/default_filters.json", help="Path to filters.json")
    parser.add_argument("--advanced", "-a", default="./settings_advanced/default_4stage_multimer.json", help="Path to advanced settings json")
    parser.add_argument("--prefilters", "-p", default=None, help="Optional prefilters json for trajectory pre-filtering")
    args = parser.parse_args()

    settings_path, filters_path, advanced_path, prefilters_path = perform_input_check(args)
    target_settings, advanced_settings, filters, prefilters = load_json_settings(settings_path, filters_path, advanced_path, prefilters_path)

    # Preserve unshifted base specs for binder_advanced so runtime shifting doesn't compound across trajectories.
    # Prefer specs provided in settings.json; fall back to advanced.json for backward compatibility.
    if target_settings.get("protocol", "binder") == "binder_advanced":
        for key in ("fixed_positions", "template_positions", "sequence_positions"):
            base_key = f"{key}_base"
            # Always prefer the value from settings.json when present; else fall back to advanced.json
            base_val = target_settings.get(key, advanced_settings.get(key))
            advanced_settings[base_key] = base_val
        # Prefer disulfide_num from settings.json; fallback to advanced.json, default 1
        advanced_settings["disulfide_num"] = int(target_settings.get("disulfide_num", advanced_settings.get("disulfide_num", 1)))

    # if target settings contains max_trajectories, override advanced setting
    if "max_trajectories" in target_settings:
        advanced_settings["max_trajectories"] = int(target_settings["max_trajectories"])

    settings_file = os.path.basename(settings_path).split(".")[0]
    filters_file = os.path.basename(filters_path).split(".")[0]
    advanced_file = os.path.basename(advanced_path).split(".")[0]
    prefilters_file = os.path.basename(prefilters_path).split(".")[0] if prefilters_path else None

    design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])
    bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

    design_paths = generate_directories_isolated(target_settings["design_path"])
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv = os.path.join(target_settings["design_path"], "trajectory_stats.csv")
    # mpnn_csv = os.path.join(target_settings["design_path"], "mpnn_design_stats.csv")
    # AF2_design_csv = os.path.join(target_settings["design_path"], "AF2_design_stats.csv")
    # final_csv = os.path.join(target_settings["design_path"], "final_design_stats.csv")
    failure_csv = os.path.join(target_settings["design_path"], "failure_csv.csv")

    create_dataframe(trajectory_csv, trajectory_labels)
    # create_dataframe(mpnn_csv, design_labels)
    # create_dataframe(AF2_design_csv, design_labels)
    # create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, args.filters)

    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
    print(f"Running binder design for target {settings_file}")
    print(f"Design settings used: {settings_file}")
    # print(f"Filtering designs based on {filters_file}")
    # if prefilters_file is not None and advanced_settings.get("pre_filter_trajectory", False):
    #     print(f"Pre-filtering trajectories based on {prefilters_file}")
    # elif prefilters_file is None and advanced_settings.get("pre_filter_trajectory", False):
    #     print("-------------------------------------------------------------------------------------")
    #     print("Warning: pre_filter_trajectory is enabled but no prefilters file provided, exiting...")
    #     print("-------------------------------------------------------------------------------------")
    #     sys.exit(1)

    script_start_time = time.time()
    trajectory_n = 1
    accepted_designs = 0

    # Optional starting PDB helix check
    if advanced_settings.get("initial_helix_check_cutoff", 100.0) < 100.0:
        starting_pdb = target_settings["starting_pdb"]
        binder_chain_id = "B"
        helix_pct, sheet_pct, loop_pct, *_ = calc_ss_percentage(starting_pdb, advanced_settings, chain_id=binder_chain_id, sec_struct_only=True)
        if helix_pct > advanced_settings["initial_helix_check_cutoff"]:
            print("-------------------------------------------------------------------------------------")
            print(f"ERROR: Helical secondary structure detected in binder chain {binder_chain_id} ({helix_pct}% helix)")
            print("The binder_advanced protocol does not support helical starting structures.")
            print(f"Starting PDB: {starting_pdb}")
            print("Program terminated.")
            print("-------------------------------------------------------------------------------------")
            sys.exit(1)
        print(f"Starting binder secondary structure check passed: {sheet_pct}% sheet, {loop_pct}% loop")

    while True:

        if check_n_trajectories(design_paths, advanced_settings):
            break

        trajectory_start_time = time.time()
        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

        def _parse_position_spec_to_1based_set(spec):
            """Parse binder-local position specs into a set of 1-based integers.

            Supports:
            - None / "" -> empty
            - int -> {int}
            - list/tuple/set of ints/strings
            - strings like "1,8,13" or "1-3,10" or "B12,B13" (chain prefix stripped)

            Notes:
            - Negative 1-based values are accepted (e.g., -1) but are NOT expanded without length.
            """
            if spec is None:
                return set()

            tokens = []
            if isinstance(spec, (list, tuple, set)):
                for item in spec:
                    if item is None:
                        continue
                    tokens.append(str(item))
            else:
                tokens.append(str(spec))

            out = set()
            for raw in tokens:
                s = raw.strip()
                if not s:
                    continue
                for part in s.replace(" ", "").split(","):
                    if not part:
                        continue
                    # strip an optional leading chain letter, e.g. "B12" -> "12"
                    if len(part) >= 2 and part[0].isalpha() and (part[1].isdigit() or part[1] == '-'):
                        part = part[1:]

                    if "-" in part[1:]:
                        # range; keep sign on the first number
                        a_str, b_str = part.split("-", 1)
                        try:
                            a = int(a_str)
                            b = int(b_str)
                        except ValueError:
                            continue
                        step = 1 if b >= a else -1
                        for v in range(a, b + step, step):
                            out.add(v)
                    else:
                        try:
                            out.add(int(part))
                        except ValueError:
                            continue
            return out

        def _required_min_len_from_specs_1based(pos_set_1based):
            """Infer a minimum binder length from 1-based position specs.

            Positive positions require length >= max(pos).
            Negative positions (Python-style from end) require length >= abs(neg).
            """
            if not pos_set_1based:
                return 0
            req = 0
            for p in pos_set_1based:
                if p > 0:
                    req = max(req, p)
                elif p < 0:
                    req = max(req, abs(p))
            return req

        def _required_min_len_from_disulfide_pairs(ds_pairs_spec):
            """Infer a minimum binder length from 0-based disulfide pair indices.

            Positive idx requires length >= idx+1.
            Negative idx (Python-style from end) requires length >= abs(idx).
            """
            if not ds_pairs_spec:
                return 0
            req = 0
            for (i, j) in ds_pairs_spec:
                for idx in (i, j):
                    if idx is None:
                        continue
                    try:
                        idx_int = int(idx)
                    except Exception:
                        continue
                    if idx_int >= 0:
                        req = max(req, idx_int + 1)
                    else:
                        req = max(req, abs(idx_int))
            return req

        if target_settings.get("protocol", "binder") == "binder_advanced":
            # Reset to base specs each trajectory before shifting/runtime edits
            advanced_settings["fixed_positions"] = advanced_settings.get("fixed_positions_base")
            advanced_settings["template_positions"] = advanced_settings.get("template_positions_base")
            advanced_settings["sequence_positions"] = advanced_settings.get("sequence_positions_base")

            if "lengths" in target_settings and target_settings["lengths"]:
                min_length = min(target_settings["lengths"])
                max_length = max(target_settings["lengths"]) if max(target_settings["lengths"]) > min_length else min_length
                samples = np.arange(min_length, max_length + 1)
                length = int(np.random.choice(samples))
            else:
                # Fallback to the starting binder chain length if no sampling range was provided
                length = get_chain_length(target_settings["starting_pdb"], "B")

            fixed_pos_1b = _parse_position_spec_to_1based_set(advanced_settings.get("fixed_positions_base"))
            template_pos_1b = _parse_position_spec_to_1based_set(advanced_settings.get("template_positions_base"))
            sequence_pos_1b = _parse_position_spec_to_1based_set(advanced_settings.get("sequence_positions_base"))
            required_min_length = max(
                _required_min_len_from_specs_1based(fixed_pos_1b | template_pos_1b | sequence_pos_1b),
                _required_min_len_from_disulfide_pairs(advanced_settings.get("disulfide_pairs")),
            )

            if length < required_min_length:
                length = int(required_min_length)
        elif target_settings.get("protocol", "binder") == "binder":
            min_length = min(target_settings["lengths"])
            max_length = max(target_settings["lengths"]) if max(target_settings["lengths"]) > min_length else min_length
            samples = np.arange(min_length, max_length + 1)
            length = np.random.choice(samples)
        else:
            print(f"Invalid protocol specified in settings.json, {target_settings.get('protocol', 'None')}, exiting...")
            sys.exit(1)

        advanced_settings["binder_len_runtime"] = int(length)

        ds_pairs = []
        # Prefer disulfide specs from settings.json; fall back to advanced.json for compatibility
        ds_pairs_pre = target_settings.get("disulfide_pairs", advanced_settings.get("disulfide_pairs"))
        if ds_pairs_pre:
            valid = True
            seen = set()
            min_sep = int(advanced_settings.get("disulfide_min_sep", 5))

            def _norm_ds_idx(idx, L):
                idx_int = int(idx)
                if idx_int < 0:
                    idx_int = L + idx_int
                return idx_int

            try:
                for (i_raw, j_raw) in ds_pairs_pre:
                    i = _norm_ds_idx(i_raw, length)
                    j = _norm_ds_idx(j_raw, length)
                    if i == j or i < 0 or j < 0 or i >= length or j >= length:
                        valid = False
                        break
                    if abs(i - j) < min_sep:
                        valid = False
                        break
                    if i in seen or j in seen:
                        valid = False
                        break
                    seen.add(i)
                    seen.add(j)
                    ds_pairs.append((i, j))
            except Exception:
                valid = False

            if not valid:
                ds_pairs = []

        # Store normalized (0-based) disulfide indices for downstream use
        advanced_settings["disulfide_pairs_runtime"] = ds_pairs

        helicity_value = load_helicity(advanced_settings)

        # === binder_advanced: optional shifted motif + random initial binder sequence ===
        if target_settings.get("protocol", "binder") == "binder_advanced":
            binder_chain_id = target_settings.get("binder_chain", "B")

            # Use starting binder sequence as the conserved pattern source (if present)
            try:
                _base_binder_seq = extract_chain_sequence_from_pdb(target_settings["starting_pdb"], binder_chain_id)
            except Exception:
                _base_binder_seq = ""

            fixed_pos_1b = _parse_position_spec_to_1based_set(advanced_settings.get("fixed_positions_base"))
            template_pos_1b = _parse_position_spec_to_1based_set(advanced_settings.get("template_positions_base"))
            sequence_pos_1b = _parse_position_spec_to_1based_set(advanced_settings.get("sequence_positions_base"))

            # Only shift positive 1-based positions (negative 1-based can't be shifted without a defined length mapping)

            motif_pos_1b = sorted({p for p in (fixed_pos_1b | template_pos_1b | sequence_pos_1b) if p > 0})

            offset = 0
            if motif_pos_1b:
                motif_pos_0b = [p - 1 for p in motif_pos_1b]
                base_min = min(motif_pos_0b)
                base_max = max(motif_pos_0b)
                # Right-aligned random motif placement: last motif residue must be at or before last binder position
                max_offset = (length - 1) - base_max
                min_offset = -base_min
                if max_offset >= min_offset:
                    rng = np.random.default_rng(seed)
                    offset = int(rng.integers(min_offset, max_offset + 1))
            advanced_settings["positions_shift_offset_runtime"] = int(offset)

            def _shift_1b_set(pos_set_1b, off, L):
                out = set()
                for p in pos_set_1b:
                    if p <= 0:
                        # Keep negative/zero values unchanged
                        out.add(p)
                        continue
                    p0 = p - 1
                    p0s = p0 + off
                    if 0 <= p0s < L:
                        out.add(p0s + 1)
                return out

            fixed_shift_1b = _shift_1b_set(fixed_pos_1b, offset, length)
            template_shift_1b = _shift_1b_set(template_pos_1b, offset, length)
            sequence_shift_1b = _shift_1b_set(sequence_pos_1b, offset, length)

            # Write shifted runtime positions back to advanced_settings (comma-separated ints)
            if fixed_shift_1b:
                advanced_settings["fixed_positions"] = ",".join(str(x) for x in sorted(fixed_shift_1b) if x)
                advanced_settings["fixed_positions_runtime"] = advanced_settings["fixed_positions"]
            else:
                advanced_settings["fixed_positions_runtime"] = ""

            if template_shift_1b:
                advanced_settings["template_positions"] = ",".join(str(x) for x in sorted(template_shift_1b) if x)
                advanced_settings["template_positions_runtime"] = advanced_settings["template_positions"]
            else:
                advanced_settings["template_positions_runtime"] = ""

            if sequence_shift_1b:
                advanced_settings["sequence_positions"] = ",".join(str(x) for x in sorted(sequence_shift_1b) if x)
                advanced_settings["sequence_positions_runtime"] = advanced_settings["sequence_positions"]
            else:
                advanced_settings["sequence_positions_runtime"] = ""

            # Initialize a random binder sequence and inject conserved residues from the starting binder
            allowed = [aa for aa in "ACDEFGHIKLMNPQRSTVWY" if aa not in set(str(advanced_settings.get("omit_AAs", "")).replace(" ", "").split(","))]
            if not allowed:
                allowed = list("ACDEFGHIKLMNPQRSTVWY")
            rng = np.random.default_rng(seed)
            init_seq_list = [str(rng.choice(allowed)) for _ in range(length)]

            # Inject conserved residues at shifted motif positions (union of fixed/template/sequence)
            injected_map = {}
            motif_shift_union_1b = sorted({p for p in (fixed_shift_1b | template_shift_1b | sequence_shift_1b) if p > 0})
            for p1 in motif_shift_union_1b:
                src_p0 = (p1 - 1) - offset
                if 0 <= src_p0 < len(_base_binder_seq):
                    aa = _base_binder_seq[src_p0]
                    init_seq_list[p1 - 1] = aa
                    injected_map[int(p1)] = aa

            # Inject disulfide cysteines AFTER length is finalized (ds_pairs are 0-based runtime indices)
            for (i, j) in ds_pairs:
                for idx0 in (i, j):
                    if 0 <= idx0 < length:
                        init_seq_list[idx0] = "C"
                        injected_map[int(idx0 + 1)] = "C"

            advanced_settings["init_binder_seq_runtime"] = "".join(init_seq_list)
            advanced_settings["init_binder_locked_map_runtime"] = injected_map
        design_name = f"{target_settings['binder_name']}_l{length}_s{seed}"
        trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
        trajectory_exists = any(os.path.exists(os.path.join(design_paths[td], design_name + ".pdb")) for td in trajectory_dirs)

        if trajectory_exists:
            trajectory_n += 1
            continue

        print(f"Starting trajectory: {design_name}")
        trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings.get("protocol", "binder"), target_settings["chains"],
                                           target_settings.get("target_hotspot_residues", None), target_settings.get("pos", None), length, seed, ds_pairs, helicity_value,
                                           design_models, advanced_settings, design_paths, failure_csv)
        trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"])
        trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
        trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

        # Save the (target+binder) trajectory sequence as FASTA alongside MPNN FASTAs
        try:
            trajectory_fastas_dir = os.path.join(target_settings["design_path"], "trajectory_fastas")
            os.makedirs(trajectory_fastas_dir, exist_ok=True)
            target_chain_ids = [c.strip() for c in str(target_settings.get("chains", "A")).split(',') if c.strip()]
            binder_chain_id = target_settings.get("binder_chain", "B")
            chain_ids_in_order = target_chain_ids + [binder_chain_id]
            full_seq = "/".join(extract_chain_sequence_from_pdb(trajectory_pdb, c) for c in chain_ids_in_order)
            fasta_name = os.path.join(trajectory_fastas_dir, f"{design_name}_complete.fasta")
            with open(fasta_name, "w") as fh:
                fh.write(f">{design_name}\n{full_seq}\n")
        except Exception as exc:
            print(f"Warning: could not write trajectory FASTA for {design_name}: {exc}")

        trajectory_time = time.time() - trajectory_start_time
        trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
        print(f"Starting trajectory took: {trajectory_time_text}\n")

        if trajectory.aux["log"].get("terminate", "") == "":
            trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
            pr_relax(trajectory_pdb, trajectory_relaxed)
            binder_chain = "B"

            num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
            num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)
            (trajectory_alpha, trajectory_beta, trajectory_loops,
             trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface,
             trajectory_i_plddt, trajectory_ss_plddt) = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

            trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain)
            trajectory_sequence = trajectory.get_seq(get_best=True)[0]
            traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings)
            trajectory_target_rmsd = target_pdb_rmsd(trajectory_pdb, target_settings["starting_pdb"], target_settings["chains"])

            trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings.get("target_hotspot_residues", ""), trajectory_sequence, trajectory_interface_residues,
                               trajectory_metrics.get('plddt'), trajectory_metrics.get('ptm'), trajectory_metrics.get('i_ptm'), trajectory_metrics.get('pae'), trajectory_metrics.get('i_pae'),
                               trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed, trajectory_interface_scores['binder_score'],
                               trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'], trajectory_interface_scores['interface_packstat'],
                               trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'], trajectory_interface_scores['interface_dG_SASA_ratio'],
                               trajectory_interface_scores['interface_fraction'], trajectory_interface_scores['interface_hydrophobicity'], trajectory_interface_scores['interface_nres'], trajectory_interface_scores['interface_interface_hbonds'],
                               trajectory_interface_scores['interface_hbond_percentage'], trajectory_interface_scores['interface_delta_unsat_hbonds'], trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                               trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA, trajectory_target_rmsd,
                               trajectory_time_text, traj_seq_notes, settings_file, filters_file, advanced_file]
            insert_data(trajectory_csv, trajectory_data)

            print(advanced_settings.get("fixed_positions", None))
            print(advanced_settings.get("sequence_positions", None))
            print(trajectory_interface_residues)
            print(ds_pairs)

            # Build fixed MPNN scaffold positions in the same "B{resnum}" format as trajectory_interface_residues
            def _parse_pos_list(v):
                if v is None:
                    return set()
                if isinstance(v, (list, tuple, set)):
                    return {int(x) for x in v}
                if isinstance(v, str):
                    v = v.strip()
                    if not v:
                        return set()
                    return {int(x) for x in v.replace(" ", "").split(",") if x}
                return {int(v)}

            fixed_positions_set = _parse_pos_list(advanced_settings.get("fixed_positions"))
            sequence_positions_set = _parse_pos_list(advanced_settings.get("sequence_positions"))
            ds_positions_set = {idx + 1 for (i, j) in ds_pairs for idx in (i, j)}  # ds_pairs are 0-based -> convert to 1-based

            fixed_mpnn_scaffold = {
                f"{binder_chain}{pos}"
                for pos in (fixed_positions_set | sequence_positions_set | ds_positions_set)
            }

            print(fixed_mpnn_scaffold)

            if not trajectory_interface_residues:
                print(f"No interface residues found for {design_name}, skipping MPNN optimization")

            elif advanced_settings["enable_mpnn"]:
                mpnn_n = 1
                accepted_mpnn = 0
                design_start_time = time.time()

                mpnn_trajectories = mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, fixed_mpnn_scaffold, advanced_settings)

                # Save complete (target+binder) sequences for all sampled MPNN outputs
                mpnn_complete_dir = os.path.join(target_settings["design_path"], "trajectory_fastas")
                os.makedirs(mpnn_complete_dir, exist_ok=True)
                for n in range(advanced_settings["num_seqs"]):
                    full_seq = mpnn_trajectories['seq'][n]
                    fasta_name = os.path.join(mpnn_complete_dir, f"{design_name}_mpnn{n+1}_complete.fasta")
                    with open(fasta_name, "w") as fh:
                        fh.write(f">{design_name}_mpnn{n+1}\n{full_seq}\n")

                # existing_mpnn_sequences = set(pd.read_csv(mpnn_csv, usecols=['Sequence'])['Sequence'].values)
                restricted_AAs = set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(',')) if advanced_settings.get("force_reject_AA") else set()

                mpnn_sequences = sorted({
                    mpnn_trajectories['seq'][n][-length:]: {
                        'seq': mpnn_trajectories['seq'][n][-length:],
                        'score': mpnn_trajectories['score'][n],
                        'seqid': mpnn_trajectories['seqid'][n]
                    } for n in range(advanced_settings["num_seqs"])
                    if (not restricted_AAs or not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
                    and mpnn_trajectories['seq'][n][-length:] #not in existing_mpnn_sequences
                }.values(), key=lambda x: x['score'])
                # del existing_mpnn_sequences

                # if mpnn_sequences:
                #     if advanced_settings["save_mpnn_fasta"] is True:
                #         save_fasta(design_name, mpnn_trajectories['seq'][n][-length:], design_paths)
                #     if advanced_settings.get("optimise_beta") and float(trajectory_beta) > 15:
                #         advanced_settings["num_recycles_validation"] = advanced_settings["optimise_beta_recycles_valid"]

                #     clear_mem()
                #     complex_prediction_model, binder_prediction_model = init_prediction_models(
                #         trajectory_pdb=trajectory_pdb,
                #         length=length,
                #         mk_afdesign_model=mk_afdesign_model,
                #         multimer_validation=multimer_validation,
                #         target_settings=target_settings,
                #         advanced_settings=advanced_settings
                #     )

                #     for mpnn_sequence in mpnn_sequences:
                #         filter_conditions, mpnn_csv_out, failure_csv_out, final_csv_out = filter_design(
                #             sequence=mpnn_sequence,
                #             basis_design_name=design_name,
                #             design_paths=design_paths,
                #             trajectory_pdb=trajectory_pdb,
                #             length=length,
                #             helicity_value=helicity_value,
                #             seed=seed,
                #             prediction_models=prediction_models,
                #             binder_chain=binder_chain,
                #             filters=filters,
                #             design_labels=design_labels,
                #             target_settings=target_settings,
                #             advanced_settings=advanced_settings,
                #             advanced_file=advanced_file,
                #             settings_file=settings_file,
                #             filters_file=filters_file,
                #             stats_csv=mpnn_csv,
                #             failure_csv=failure_csv,
                #             final_csv=final_csv,
                #             complex_prediction_model=complex_prediction_model,
                #             binder_prediction_model=binder_prediction_model,
                #             is_mpnn_model=True,
                #             mpnn_n=mpnn_n
                #         )
                #         mpnn_n += 1
                #         if filter_conditions is True:
                #             accepted_mpnn += 1
                #         if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
                #             break

                #     design_time = time.time() - design_start_time
                #     design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                #     print(f"MPNN redesign for trajectory {design_name} took: {design_time_text}\n")
                # else:
                #     print('No unique MPNN sequences after filtering omit_AAs/duplicates')

            # acceptance monitoring
            # if trajectory_n >= advanced_settings.get("start_monitoring", 1000) and advanced_settings.get("enable_rejection_check", False):
            #     acceptance = accepted_designs / trajectory_n if trajectory_n else 0
            #     if acceptance < advanced_settings.get("acceptance_rate", 0):
            #         print("Acceptance rate below threshold, stopping early")
            #         break

        trajectory_n += 1
        gc.collect()

    elapsed_time = time.time() - script_start_time
    elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
    print(f"Finished all designs. Ran {trajectory_n} trajectories in {elapsed_text}")


if __name__ == "__main__":
    main()
