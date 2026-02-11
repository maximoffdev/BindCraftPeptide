#!/usr/bin/env python3
"""Repredict and rescore existing designs.
Takes existing trajectory PDBs (or other complexes), extracts binder sequence, repredicts with AF2, and writes repredict_stats.csv plus repredicted PDBs.
"""
import os
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import sys
import glob
import argparse
import json
from pathlib import Path
import pyrosetta as pr
from Bio.PDB import PDBParser
import gc
from functions import *


def read_fasta_sequence(fasta_path):
    """Return concatenated sequence from a single-entry FASTA file."""
    with open(fasta_path, "r") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    seq_lines = [line for line in lines if not line.startswith(">")]
    return "".join(seq_lines)


def extract_chain_sequence(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"Chain '{chain_id}' not found in {pdb_path}")
    chain = model[chain_id]
    three_to_one = {
        'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',
        'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
    }
    seq = []
    for res in chain:
        if res.id[0] != ' ':
            continue
        resname = res.get_resname()
        if resname in three_to_one:
            seq.append(three_to_one[resname])
        else:
            seq.append('X')
    return ''.join(seq)


def main():
    parser = argparse.ArgumentParser(description="Repredict existing designs and rescore interfaces")
    parser.add_argument("--settings", "-s", required=True, help="Path to basic settings.json")
    parser.add_argument("--filters", "-f", default="./settings_filters/default_filters.json", help="Path to filters.json")
    parser.add_argument("--advanced", "-a", default="./settings_advanced/default_4stage_multimer.json", help="Path to advanced settings json")
    parser.add_argument("--prefilters", "-p", default=None, help="Unused placeholder for compatibility")
    args = parser.parse_args()

    settings_path, filters_path, advanced_path, prefilters_path = perform_input_check(args)
    target_settings, advanced_settings, filters, prefilters = load_json_settings(settings_path, filters_path, advanced_path, prefilters_path)
    binder_chain = target_settings.get("binder_chain", "B")
    target_settings["binder_chain"] = binder_chain

    design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])
    bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

    # Initialize PyRosetta once for interface scoring/alignment used downstream
    _has_is_init = hasattr(pr, "is_initialized")
    try:
        if not _has_is_init or not pr.is_initialized():
            pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
            # pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all')
    except AttributeError:
        # Older PyRosetta builds may not expose is_initialized; fall back to plain init
            pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')

    design_paths = generate_directories_isolated(target_settings["design_path"])

    # Reprediction outputs (used for both trajectory PDBs and MPNN FASTA inputs)
    repred_base = os.path.join(target_settings["design_path"], "Repredicted")
    repred_relaxed = os.path.join(repred_base, "Relaxed")
    repred_binder = os.path.join(repred_base, "Binder")
    repred_best = os.path.join(repred_base, "Best")
    for path in [repred_base, repred_relaxed, repred_binder, repred_best]:
        os.makedirs(path, exist_ok=True)
    design_paths.setdefault("Repredicted", repred_base)
    design_paths.setdefault("Repredicted/Relaxed", repred_relaxed)
    design_paths.setdefault("Repredicted/Binder", repred_binder)
    design_paths.setdefault("Repredicted/Best", repred_best)
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    repredict_csv = os.path.join(target_settings["design_path"], "repredict_stats.csv")
    repredict_final_csv = os.path.join(target_settings["design_path"], "repredict_final_stats.csv")
    create_dataframe(repredict_csv, design_labels)
    create_dataframe(repredict_final_csv, final_labels)
    repredict_failure_csv = None  # no filtering/failure CSV for repredict stage

    pdb_dir = Path(target_settings["design_path"]) / "Trajectory"
    pdb_paths = sorted(pdb_dir.glob("*.pdb"))
    print(f"Found {len(pdb_paths)} PDBs in {pdb_dir}")

    mpnn_fasta_dir = Path(target_settings["design_path"]) / "trajectory_fastas"
    mpnn_fasta_paths = sorted(
        p for p in mpnn_fasta_dir.glob("*.fasta")
        if "mpnn" in p.name.lower()
    )
    print(f"Found {len(mpnn_fasta_paths)} MPNN FASTAs in {mpnn_fasta_dir}")

    if not pdb_paths and not mpnn_fasta_paths:
        print("No inputs found in Trajectory/*.pdb or trajectory_fastas/*.fasta")
        sys.exit(1)

    for pdb_path in pdb_paths:
        print(f"Repredicting design from {pdb_path}")
        try:
            binder_seq = extract_chain_sequence(pdb_path, binder_chain)
        except Exception as exc:
            print(f"Skipping {pdb_path}: {exc}")
            continue

        length = len(binder_seq)
        design_name = os.path.splitext(os.path.basename(pdb_path))[0]
        helicity_value = load_helicity(advanced_settings)
        seed = 0  # seed not relevant here

        # free any leftover GPU/TPU memory before loading models
        clear_mem()

        complex_prediction_model, binder_prediction_model = init_prediction_models(
            trajectory_pdb=str(pdb_path),
            length=length,
            mk_afdesign_model=mk_afdesign_model,
            multimer_validation=multimer_validation,
            target_settings=target_settings,
            advanced_settings=advanced_settings
        )

        _filter_conditions, _stats_csv, _failure_csv, _final_csv = repredict_and_score_structure(
            sequence=binder_seq,
            basis_design_name=design_name,
            design_paths=design_paths,
            trajectory_pdb=str(pdb_path),
            length=length,
            helicity_value=helicity_value,
            seed=seed,
            prediction_models=prediction_models,
            binder_chain=binder_chain,
            filters=None,
            design_labels=design_labels,
            target_settings=target_settings,
            advanced_settings=advanced_settings,
            advanced_file=os.path.basename(advanced_path).split('.')[0],
            settings_file=os.path.basename(settings_path).split('.')[0],
            filters_file="",
            stats_csv=repredict_csv,
            failure_csv=repredict_failure_csv,
            final_csv=repredict_final_csv,
            complex_prediction_model=complex_prediction_model,
            binder_prediction_model=binder_prediction_model,
            is_mpnn_model=False,
            mpnn_n=None,
            design_path_key="Repredicted"
        )

        clear_mem()
        gc.collect()

    for fasta_path in mpnn_fasta_paths:
        print(f"Repredicting design from {fasta_path}")
        fasta_seq = read_fasta_sequence(fasta_path)
        fasta_stem = fasta_path.stem  # e.g., design_mpnn1_complete
        design_name = fasta_stem[:-9] if fasta_stem.endswith("_complete") else fasta_stem
        parent_design = design_name.split("_mpnn")[0]
        trajectory_pdb = pdb_dir / f"{parent_design}.pdb"
        if not trajectory_pdb.exists():
            print(f"Skipping {fasta_path}: trajectory PDB {trajectory_pdb} not found")
            continue

        try:
            binder_template_seq = extract_chain_sequence(str(trajectory_pdb), binder_chain)
        except Exception as exc:
            print(f"Skipping {fasta_path}: {exc}")
            continue

        length = len(binder_template_seq)
        if len(fasta_seq) < length:
            print(f"Skipping {fasta_path}: FASTA sequence shorter than binder length {length}")
            continue
        binder_seq = fasta_seq[-length:]
        helicity_value = load_helicity(advanced_settings)
        seed = 0

        clear_mem()

        complex_prediction_model, binder_prediction_model = init_prediction_models(
            trajectory_pdb=str(trajectory_pdb),
            length=length,
            mk_afdesign_model=mk_afdesign_model,
            multimer_validation=multimer_validation,
            target_settings=target_settings,
            advanced_settings=advanced_settings
        )

        _filter_conditions, _stats_csv, _failure_csv, _final_csv = repredict_and_score_structure(
            sequence=binder_seq,
            basis_design_name=design_name,
            design_paths=design_paths,
            trajectory_pdb=str(trajectory_pdb),
            length=length,
            helicity_value=helicity_value,
            seed=seed,
            prediction_models=prediction_models,
            binder_chain=binder_chain,
            filters=None,
            design_labels=design_labels,
            target_settings=target_settings,
            advanced_settings=advanced_settings,
            advanced_file=os.path.basename(advanced_path).split('.')[0],
            settings_file=os.path.basename(settings_path).split('.')[0],
            filters_file="",
            stats_csv=repredict_csv,
            failure_csv=repredict_failure_csv,
            final_csv=repredict_final_csv,
            complex_prediction_model=complex_prediction_model,
            binder_prediction_model=binder_prediction_model,
            is_mpnn_model=False,
            mpnn_n=None,
            design_path_key="Repredicted"
        )

        clear_mem()
        gc.collect()

    print(f"Reprediction complete. Results in {repredict_csv}")


if __name__ == "__main__":
    main()
