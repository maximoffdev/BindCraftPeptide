#!/usr/bin/env python3
"""Score existing (repredicted) complex PDBs without AF2 reprediction.

Scans a directory of PDBs (by default Repredicted/Relaxed), groups files by design name
using the common *_modelN.pdb naming, computes interface/secondary-structure metrics,
and writes repredict_stats.csv.

Hotspot RMSD is computed by comparing each scored model PDB to its corresponding
trajectory PDB in Trajectory/. For MPNN designs, the mapping strips the *_mpnnX
suffix to find the base trajectory name.
"""
import os
import sys
import argparse
import time
import re
from collections import defaultdict
from pathlib import Path
import pyrosetta as pr
from Bio.PDB import PDBParser
from functions import *


def extract_chain_sequence(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"Chain '{chain_id}' not found in {pdb_path}")
    chain = model[chain_id]
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    seq = []
    for res in chain:
        if res.id[0] != ' ':
            continue
        resname = res.get_resname()
        seq.append(three_to_one.get(resname, 'X'))
    return ''.join(seq)


_MODEL_RE = re.compile(r"^(?P<base>.+)_model(?P<model>\d+)$")


def parse_design_and_model(pdb_path: Path):
    """Return (design_name, model_number) from a *_modelN.pdb name.

    If the file does not match the pattern, model_number is set to 1.
    """
    stem = pdb_path.stem
    match = _MODEL_RE.match(stem)
    if not match:
        return stem, 1
    return match.group("base"), int(match.group("model"))


def trajectory_name_from_design(design_name: str):
    """Map repredicted design name to the base trajectory name.

    Example:
      scaffold_l13_s733810_mpnn8 -> scaffold_l13_s733810
    """
    return design_name.split("_mpnn")[0]


def collect_pdbs(scan_root: Path):
    """Collect PDBs under scan_root.

    If a Relaxed subfolder exists (or any file lives under Relaxed), prefer only Relaxed PDBs.
    Excludes Binder/ and Best/ by default to keep only complex predictions.
    """
    if not scan_root.exists():
        return []

    pdbs = sorted(scan_root.rglob("*.pdb"))
    if not pdbs:
        return []

    relaxed = [p for p in pdbs if "Relaxed" in p.parts]
    if relaxed:
        pdbs = relaxed

    pdbs = [p for p in pdbs if "Binder" not in p.parts and "Best" not in p.parts]
    return pdbs


def main():
    parser = argparse.ArgumentParser(description="Score existing PDBs without AF2 reprediction")
    parser.add_argument("--settings", "-s", required=True, help="Path to basic settings.json")
    parser.add_argument("--filters", "-f", default="./settings_filters/default_filters.json", help="Path to filters.json (unused, for compatibility)")
    parser.add_argument("--advanced", "-a", default="./settings_advanced/default_4stage_multimer.json", help="Path to advanced settings json")
    parser.add_argument("--prefilters", "-p", default=None, help="Unused placeholder for compatibility")
    parser.add_argument("--input-pdbs", default=None, help="Optional directory to scan for PDBs; defaults to design_path/Repredicted")
    args = parser.parse_args()

    settings_path, filters_path, advanced_path, prefilters_path = perform_input_check(args)
    target_settings, advanced_settings, filters, prefilters = load_json_settings(settings_path, filters_path, advanced_path, prefilters_path)
    binder_chain = target_settings.get("binder_chain", "B")
    target_settings["binder_chain"] = binder_chain

    bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

    # PyRosetta init for scoring
    _has_is_init = hasattr(pr, "is_initialized")
    try:
        if not _has_is_init or not pr.is_initialized():
            pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
    except AttributeError:
        pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')

    design_paths = generate_directories_isolated(target_settings["design_path"])
    _, design_labels, _ = generate_dataframe_labels()

    repredict_csv = os.path.join(target_settings["design_path"], "repredict_stats.csv")
    create_dataframe(repredict_csv, design_labels)

    scan_root = Path(args.input_pdbs) if args.input_pdbs else Path(target_settings["design_path"]) / "Repredicted"
    pdb_paths = collect_pdbs(scan_root)
    print(f"Found {len(pdb_paths)} PDBs under {scan_root}")
    if not pdb_paths:
        print("No PDBs to score; exiting")
        sys.exit(1)

    grouped = defaultdict(dict)
    for pdb_path in pdb_paths:
        design_name, model_num = parse_design_and_model(pdb_path)
        if model_num < 1 or model_num > 5:
            continue
        if model_num not in grouped[design_name]:
            grouped[design_name][model_num] = pdb_path

    if not grouped:
        print("No model PDBs matching *_model[1-5].pdb found; exiting")
        sys.exit(1)

    helicity_value = load_helicity(advanced_settings)
    seed = 0
    design_path = Path(target_settings["design_path"])

    for design_name in sorted(grouped.keys()):
        model_map = grouped[design_name]
        model_nums = sorted(model_map.keys())
        primary_pdb = model_map[model_nums[0]]

        print(f"Scoring {design_name} from {len(model_nums)} model PDB(s)")
        start_time = time.time()

        try:
            binder_seq = extract_chain_sequence(primary_pdb, binder_chain)
        except Exception as exc:
            print(f"Skipping {design_name}: {exc}")
            continue

        length = len(binder_seq)

        trajectory_base = trajectory_name_from_design(design_name)
        trajectory_pdb = design_path / "Trajectory" / f"{trajectory_base}.pdb"
        if not trajectory_pdb.exists():
            trajectory_pdb = None

        complex_statistics = {}
        interface_residues_for_row = ""

        for model_num in model_nums:
            pdb_path = model_map[model_num]
            try:
                num_clashes = calculate_clash_score(str(pdb_path))
                interface_scores, interface_AA, interface_residues = score_interface(str(pdb_path), binder_chain)
                alpha, beta, loops, alpha_interface, beta_interface, loops_interface, i_plddt, ss_plddt = calc_ss_percentage(str(pdb_path), advanced_settings, binder_chain)
                target_rmsd = target_pdb_rmsd(str(pdb_path), target_settings["starting_pdb"], target_settings["chains"])
                hotspot_rmsd = (
                    unaligned_rmsd(str(trajectory_pdb), str(pdb_path), binder_chain, binder_chain)
                    if trajectory_pdb is not None
                    else None
                )

                if not interface_residues_for_row:
                    interface_residues_for_row = interface_residues

                complex_statistics[model_num] = {
                    'i_pLDDT': i_plddt,
                    'ss_pLDDT': ss_plddt,
                    'Unrelaxed_Clashes': num_clashes,
                    'Relaxed_Clashes': num_clashes,
                    'Binder_Energy_Score': interface_scores['binder_score'],
                    'Surface_Hydrophobicity': interface_scores['surface_hydrophobicity'],
                    'ShapeComplementarity': interface_scores['interface_sc'],
                    'PackStat': interface_scores['interface_packstat'],
                    'dG': interface_scores['interface_dG'],
                    'dSASA': interface_scores['interface_dSASA'],
                    'dG/dSASA': interface_scores['interface_dG_SASA_ratio'],
                    'Interface_SASA_%': interface_scores['interface_fraction'],
                    'Interface_Hydrophobicity': interface_scores['interface_hydrophobicity'],
                    'n_InterfaceResidues': interface_scores['interface_nres'],
                    'n_InterfaceHbonds': interface_scores['interface_interface_hbonds'],
                    'InterfaceHbondsPercentage': interface_scores['interface_hbond_percentage'],
                    'n_InterfaceUnsatHbonds': interface_scores['interface_delta_unsat_hbonds'],
                    'InterfaceUnsatHbondsPercentage': interface_scores['interface_delta_unsat_hbonds_percentage'],
                    'InterfaceAAs': interface_AA,
                    'Interface_Helix%': alpha_interface,
                    'Interface_BetaSheet%': beta_interface,
                    'Interface_Loop%': loops_interface,
                    'Binder_Helix%': alpha,
                    'Binder_BetaSheet%': beta,
                    'Binder_Loop%': loops,
                    'Hotspot_RMSD': hotspot_rmsd,
                    'Target_RMSD': target_rmsd,
                }
            except Exception as exc:
                print(f"Skipping {design_name} model {model_num}: {exc}")
                continue

        if not complex_statistics:
            print(f"No models successfully scored for {design_name}; skipping")
            continue

        binder_statistics = {}

        complex_averages = calculate_averages(complex_statistics, handle_aa=True)
        binder_averages = calculate_averages(binder_statistics) if binder_statistics else {}

        statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                             'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                             'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                             'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

        model_numbers = range(1, 6)
        data = [design_name, advanced_settings.get("design_algorithm", ""), length, seed, helicity_value,
            target_settings.get("target_hotspot_residues", ""), binder_seq, interface_residues_for_row, "", ""]

        for label in statistics_labels:
            data.append(complex_averages.get(label, None))
            for model in model_numbers:
                data.append(complex_statistics.get(model, {}).get(label, None))

        for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:
            data.append(binder_averages.get(label, None))
            for model in model_numbers:
                data.append(binder_statistics.get(model, {}).get(label, None))

        end_time = time.time() - start_time
        elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(end_time // 3600), int((end_time % 3600) // 60), int(end_time % 60))}"
        clashes_for_notes = complex_averages.get('Relaxed_Clashes', 0) or 0
        seq_notes = validate_design_sequence(binder_seq, clashes_for_notes, advanced_settings)
        settings_file = os.path.basename(settings_path).split('.')[0]
        filters_file = os.path.basename(filters_path).split('.')[0]
        advanced_file = os.path.basename(advanced_path).split('.')[0]
        data.extend([elapsed_text, seq_notes, settings_file, filters_file, advanced_file])

        insert_data(repredict_csv, data)

    print(f"Scoring complete. Results in {repredict_csv}")


if __name__ == "__main__":
    main()
