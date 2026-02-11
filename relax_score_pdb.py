#!/usr/bin/env python3
"""Score existing (repredicted) complex PDBs with relaxation."""
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

# === PRESERVED HELPER FUNCTIONS ===
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
        if res.id[0] != ' ': continue
        resname = res.get_resname()
        seq.append(three_to_one.get(resname, 'X'))
    return ''.join(seq)

def parse_design_and_model(pdb_path: Path):
    _MODEL_RE = re.compile(r"^(?P<base>.+)_model(?P<model>\d+)$")
    stem = pdb_path.stem
    match = _MODEL_RE.match(stem)
    if not match: return stem, 1
    return match.group("base"), int(match.group("model"))

def trajectory_name_from_design(design_name: str):
    return design_name.split("_mpnn")[0]

def main():
    parser = argparse.ArgumentParser(description="Score existing PDBs with relaxation")
    parser.add_argument("--settings", "-s", required=True)
    parser.add_argument("--filters", "-f", default="./settings_filters/default_filters.json")
    parser.add_argument("--advanced", "-a", default="./settings_advanced/default_4stage_multimer.json")
    parser.add_argument("--prefilters", "-p", default=None) # ADDED TO FIX ATTRIBUTERROR
    parser.add_argument("--input-pdbs", default=None)
    parser.add_argument("--use_disulfide", action="store_true")
    args = parser.parse_args()

    # Load settings - perform_input_check requires 4 arguments in the namespace
    settings_path, filters_path, advanced_path, prefilters_path = perform_input_check(args)
    target_settings, advanced_settings, filters, prefilters = load_json_settings(settings_path, filters_path, advanced_path, prefilters_path)
    binder_chain = target_settings.get("binder_chain", "B")
    target_settings["binder_chain"] = binder_chain

    bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

    # PyRosetta Init
    try:
        if not pr.is_initialized():
            pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
    except:
        pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')

    design_paths = generate_directories_isolated(target_settings["design_path"])
    _, design_labels, _ = generate_dataframe_labels()

    repredict_csv = os.path.join(target_settings["design_path"], "repredict_stats.csv")
    create_dataframe(repredict_csv, design_labels)

    # Path logic
    scan_root = Path(args.input_pdbs) if args.input_pdbs else Path(target_settings["design_path"]) / "Repredicted"
    relaxed_dir = scan_root / "Relaxed"
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect only unrelaxed complex PDBs
    pdb_paths = [p for p in scan_root.glob("*.pdb") if "Relaxed" not in p.parts and "Binder" not in p.parts and "Best" not in p.parts]
    
    grouped = defaultdict(dict)
    for pdb_path in pdb_paths:
        design_name, model_num = parse_design_and_model(pdb_path)
        if 1 <= model_num <= 5:
            grouped[design_name][model_num] = pdb_path

    helicity_value = load_helicity(advanced_settings)
    seed = 0
    design_path_root = Path(target_settings["design_path"])

    for design_name in sorted(grouped.keys()):
        model_map = grouped[design_name]
        model_nums = sorted(model_map.keys())
        primary_pdb = model_map[model_nums[0]]

        print(f"Scoring {design_name}...")
        start_time = time.time()

        try:
            binder_seq = extract_chain_sequence(primary_pdb, binder_chain)
        except Exception as exc:
            print(f"Skipping {design_name}: {exc}"); continue

        length = len(binder_seq)
        trajectory_base = trajectory_name_from_design(design_name)
        trajectory_pdb = design_path_root / "Trajectory" / f"{trajectory_base}.pdb"
        
        complex_statistics = {}
        interface_residues_for_row = ""

        for model_num in model_nums:
            unrelaxed_pdb = model_map[model_num]
            relaxed_pdb = relaxed_dir / f"{design_name}_model{model_num}.pdb"

            try:
                # --- RELAXATION STEP ---
                if not relaxed_pdb.exists():
                    print(f"  Relaxing model {model_num}...")
                    pairs = advanced_settings.get("disulfide_pairs_runtime") if advanced_settings.get("use_disulfide_loss", False) else None
                    pr_relax(str(unrelaxed_pdb), str(relaxed_pdb), 
                             disulfide=advanced_settings.get("use_disulfide_loss", False), 
                             binder_chain=binder_chain, binder_local_pairs=pairs)

                # --- SCORING (on relaxed version) ---
                num_clashes_unrelaxed = calculate_clash_score(str(unrelaxed_pdb))
                num_clashes_relaxed = calculate_clash_score(str(relaxed_pdb))
                interface_scores, interface_AA, interface_residues = score_interface(str(relaxed_pdb), binder_chain)
                alpha, beta, loops, alpha_interface, beta_interface, loops_interface, i_plddt, ss_plddt = calc_ss_percentage(str(relaxed_pdb), advanced_settings, binder_chain)
                target_rmsd = target_pdb_rmsd(str(relaxed_pdb), target_settings["starting_pdb"], target_settings["chains"])
                hotspot_rmsd = unaligned_rmsd(str(trajectory_pdb), str(relaxed_pdb), binder_chain, binder_chain) if trajectory_pdb.exists() else None

                if not interface_residues_for_row: interface_residues_for_row = interface_residues

                complex_statistics[model_num] = {
                    'i_pLDDT': i_plddt, 'ss_pLDDT': ss_plddt,
                    'Unrelaxed_Clashes': num_clashes_unrelaxed, 'Relaxed_Clashes': num_clashes_relaxed,
                    'Binder_Energy_Score': interface_scores['binder_score'], 'Surface_Hydrophobicity': interface_scores['surface_hydrophobicity'],
                    'ShapeComplementarity': interface_scores['interface_sc'], 'PackStat': interface_scores['interface_packstat'],
                    'dG': interface_scores['interface_dG'], 'dSASA': interface_scores['interface_dSASA'],
                    'dG/dSASA': interface_scores['interface_dG_SASA_ratio'], 'Interface_SASA_%': interface_scores['interface_fraction'],
                    'Interface_Hydrophobicity': interface_scores['interface_hydrophobicity'], 'n_InterfaceResidues': interface_scores['interface_nres'],
                    'n_InterfaceHbonds': interface_scores['interface_interface_hbonds'], 'InterfaceHbondsPercentage': interface_scores['interface_hbond_percentage'],
                    'n_InterfaceUnsatHbonds': interface_scores['interface_delta_unsat_hbonds'], 'InterfaceUnsatHbondsPercentage': interface_scores['interface_delta_unsat_hbonds_percentage'],
                    'InterfaceAAs': interface_AA, 'Interface_Helix%': alpha_interface, 'Interface_BetaSheet%': beta_interface, 'Interface_Loop%': loops_interface,
                    'Binder_Helix%': alpha, 'Binder_BetaSheet%': beta, 'Binder_Loop%': loops, 'Hotspot_RMSD': hotspot_rmsd, 'Target_RMSD': target_rmsd,
                }
            except Exception as exc:
                print(f"Skipping model {model_num}: {exc}"); continue

        # === DATA ROW CONSTRUCTION ===
        if not complex_statistics: continue
        complex_averages = calculate_averages(complex_statistics, handle_aa=True)
        binder_statistics = {} 
        binder_averages = {}

        statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                             'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                             'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                             'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

        data = [design_name, advanced_settings.get("design_algorithm", ""), length, seed, helicity_value,
            target_settings.get("target_hotspot_residues", ""), binder_seq, interface_residues_for_row, "", ""]

        for label in statistics_labels:
            data.append(complex_averages.get(label, None))
            for model in range(1, 6):
                data.append(complex_statistics.get(model, {}).get(label, None))

        for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:
            data.append(binder_averages.get(label, None))
            for model in range(1, 6):
                data.append(binder_statistics.get(model, {}).get(label, None))

        end_time = time.time() - start_time
        elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(end_time // 3600), int((end_time % 3600) // 60), int(end_time % 60))}"
        data.extend([elapsed_text, validate_design_sequence(binder_seq, complex_averages.get('Relaxed_Clashes', 0), advanced_settings), 
                     os.path.basename(settings_path), os.path.basename(filters_path), os.path.basename(advanced_path)])

        insert_data(repredict_csv, data)

    print(f"Scoring complete. Results in {repredict_csv}")

if __name__ == "__main__":
    main()