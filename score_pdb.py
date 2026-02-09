#!/usr/bin/env python3
"""Score existing (repredicted) complex PDBs without AF2 reprediction.

Scans a directory of PDBs (by default Repredicted/Relaxed), groups files by design name
using the common *_modelN.pdb naming, computes interface/secondary-structure metrics,
and writes repredict_stats.csv.

Optionally, PDBs can be relaxed with PyRosetta before scoring (see --relax).

Hotspot RMSD is computed by comparing each scored model PDB to its corresponding
trajectory PDB in Trajectory/. For MPNN designs, the mapping strips the *_mpnnX
suffix to find the base trajectory name.
"""
import os
import sys
import argparse
import time
import re
import csv
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


def infer_single_disulfide_pair_from_binder_sequence(binder_seq: str):
    """Infer a single disulfide pair from binder sequence.

    Assumption (per workflow simplification): exactly two cysteines exist in every binder
    and they should be stapled together.

    Returns a list of one tuple with 0-based binder-local indices, or None if not possible.
    """
    cys_positions = [i for i, aa in enumerate(binder_seq) if aa == "C"]
    if len(cys_positions) != 2:
        return None
    return [(cys_positions[0], cys_positions[1])]


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
        # Prefer Relaxed PDBs, but keep unrelaxed ones that don't have a relaxed counterpart.
        relaxed_by_name = {p.name: p for p in relaxed}
        unrelaxed = [p for p in pdbs if "Relaxed" not in p.parts]
        unrelaxed_without_relaxed_copy = [p for p in unrelaxed if p.name not in relaxed_by_name]
        pdbs = list(relaxed) + unrelaxed_without_relaxed_copy

    pdbs = [p for p in pdbs if "Binder" not in p.parts and "Best" not in p.parts]
    return pdbs


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


_BOLTZ_MODEL_SUFFIX_RE = re.compile(r"^(?P<base>.+)_model(?P<model>\d+)$")


def load_boltz_repredict_stats(design_path: Path):
    """Load boltz_repredict_stats.csv if present.

    Returns a mapping keyed by (design_base_name, model_num_or_None) -> stats dict.
    If the CSV's design_name includes a *_modelN suffix, it is parsed and stored under that model.
    Otherwise it is stored under model None.
    """
    print(f"Looking for boltz_repredict_stats.csv under {design_path}")
    boltz_csv = design_path / "boltz_repredict_stats.csv"
    if not boltz_csv.exists():
        return {}

    by_design_model = {}
    with boltz_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_name = (row.get("design_name") or "").strip()
            if not raw_name:
                continue

            match = _BOLTZ_MODEL_SUFFIX_RE.match(raw_name)
            if match:
                design_base = match.group("base")
                model_num = int(match.group("model"))
            else:
                design_base = raw_name
                model_num = None

            stats = {k: _safe_float(v) if k != "design_name" else raw_name for k, v in row.items()}

            # Map boltz naming onto BindCraft's expected complex keys where possible
            mapped = {
                "pTM": stats.get("pTM_complex"),
                "i_pTM": stats.get("ipTM_complex"),
                "pLDDT": stats.get("pLDDT_complex"),
                "i_pLDDT": stats.get("ipLDDT_complex"),
                "pAE": stats.get("pDE_complex"),
                "i_pAE": stats.get("ipDE_complex"),
            }
            for k, v in mapped.items():
                if v is not None:
                    stats[k] = v

            by_design_model[(design_base, model_num)] = stats

    return by_design_model


def main():
    parser = argparse.ArgumentParser(description="Score existing PDBs without AF2 reprediction")
    parser.add_argument("--settings", "-s", required=True, help="Path to basic settings.json")
    parser.add_argument("--filters", "-f", default="./settings_filters/default_filters.json", help="Path to filters.json (unused, for compatibility)")
    parser.add_argument("--advanced", "-a", default="./settings_advanced/default_4stage_multimer.json", help="Path to advanced settings json")
    parser.add_argument("--prefilters", "-p", default=None, help="Unused placeholder for compatibility")
    parser.add_argument("--input-pdbs", default=None, help="Optional directory to scan for PDBs; defaults to design_path/Repredicted")
    parser.add_argument("--relax", action="store_true", help="Relax complex PDBs into Relaxed/ before scoring")
    parser.add_argument("--use_disulfide", action="store_true", help="Enable disulfide mode during relaxation")
    parser.add_argument("--design_path", "-d", default=None, help="Optional override for design_path in settings.json")
    args = parser.parse_args()

    settings_path, filters_path, advanced_path, prefilters_path = perform_input_check(args)
    target_settings, advanced_settings, filters, prefilters = load_json_settings(settings_path, filters_path, advanced_path, prefilters_path)
    binder_chain = target_settings.get("binder_chain", "B")
    target_settings["binder_chain"] = binder_chain
    if args.design_path is not None:
        target_settings["design_path"] = args.design_path
        print(f"Overriding design_path with CLI argument: {target_settings['design_path']}")

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
    relaxed_dir = None

    # Optional relaxation step: relax all non-relaxed complex PDBs into Relaxed/ and then score relaxed.
    if args.relax:
        relaxed_dir = scan_root / "Relaxed"
        relaxed_dir.mkdir(parents=True, exist_ok=True)

        # Only relax complex PDBs (exclude Binder/Best and anything already under Relaxed)
        unrelaxed_pdbs = [
            p for p in sorted(scan_root.rglob("*.pdb"))
            if "Relaxed" not in p.parts and "Binder" not in p.parts and "Best" not in p.parts
        ]
        print(f"Found {len(unrelaxed_pdbs)} unrelaxed complex PDB(s) under {scan_root} to relax")

        # Determine whether disulfides should be applied during relaxation.
        # CLI has highest priority; then target_settings (if set) overrides advanced_settings.
        disulfide_enabled = bool(
            args.use_disulfide
            or target_settings.get(
                "use_disulfide_loss",
                advanced_settings.get("use_disulfide_loss", False),
            )
        )

        for unrelaxed_pdb in unrelaxed_pdbs:
            design_base, model_num = parse_design_and_model(unrelaxed_pdb)
            if model_num < 1 or model_num > 5:
                continue
            out_relaxed = relaxed_dir / f"{design_base}_model{model_num}.pdb"
            if out_relaxed.exists():
                continue
            print(f"Relaxing {unrelaxed_pdb.name} -> {out_relaxed.name}")

            pairs = None
            disulfide_for_this = False
            if disulfide_enabled:
                try:
                    binder_seq = extract_chain_sequence(str(unrelaxed_pdb), binder_chain)
                    pairs = infer_single_disulfide_pair_from_binder_sequence(binder_seq)
                    if pairs is None:
                        cys_count = binder_seq.count("C")
                        raise ValueError(
                            f"binder sequence has {cys_count} cysteines (expected exactly 2)"
                        )
                    disulfide_for_this = True
                except Exception as exc:
                    print(
                        f"Warning: skipping relaxation for {unrelaxed_pdb.name} due to disulfide inference error: {exc}"
                    )
                    continue

            try:
                pr_relax(
                    str(unrelaxed_pdb),
                    str(out_relaxed),
                    disulfide=disulfide_for_this,
                    binder_chain=binder_chain,
                    binder_local_pairs=pairs,
                )
            except Exception as exc:
                print(f"Warning: skipping relaxation for {unrelaxed_pdb.name} due to relax error: {exc}")
                continue

    # Collect PDBs to score: prefer Relaxed/ if present (or if relax was enabled)
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
    boltz_stats_by_design_model = load_boltz_repredict_stats(design_path)
    if boltz_stats_by_design_model:
        print(f"Loaded boltz repredict stats from {design_path / 'boltz_repredict_stats.csv'}")
    else:
        print("No boltz_repredict_stats.csv found; continuing without boltz metrics")

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
                # If relaxation is enabled we want both unrelaxed + relaxed clash counts.
                # pdb_path will usually point at the relaxed structure due to collect_pdbs() preference.
                num_clashes_unrelaxed = None
                num_clashes_relaxed = None

                if args.relax:
                    # Derive the corresponding unrelaxed file path (same basename, outside Relaxed/).
                    if "Relaxed" in pdb_path.parts:
                        rel_idx = pdb_path.parts.index("Relaxed")
                        unrelaxed_candidate = Path(*pdb_path.parts[:rel_idx]) / pdb_path.name
                    else:
                        unrelaxed_candidate = pdb_path

                    num_clashes_unrelaxed = calculate_clash_score(str(unrelaxed_candidate)) if unrelaxed_candidate.exists() else None
                    num_clashes_relaxed = calculate_clash_score(str(pdb_path))
                else:
                    num_clashes_relaxed = calculate_clash_score(str(pdb_path))
                interface_scores, interface_AA, interface_residues = score_interface(str(pdb_path), binder_chain)
                alpha, beta, loops, alpha_interface, beta_interface, loops_interface, i_plddt, ss_plddt = calc_ss_percentage(str(pdb_path), advanced_settings, binder_chain)
                target_rmsd = target_pdb_rmsd(str(pdb_path), target_settings["starting_pdb"], target_settings["chains"])
                # New: align PDBs before calculating hotspot RMSD
                align_pdbs(str(trajectory_pdb), str(pdb_path), binder_chain, binder_chain)
                hotspot_rmsd = (
                    unaligned_rmsd(str(trajectory_pdb), str(pdb_path), binder_chain, binder_chain)
                    if trajectory_pdb is not None
                    else None
                )

                if not interface_residues_for_row:
                    interface_residues_for_row = interface_residues

                complex_statistics[model_num] = {
                    # Optional: boltz AF metrics (merged below if available)
                    'i_pLDDT': i_plddt,
                    'ss_pLDDT': ss_plddt,
                    'Unrelaxed_Clashes': num_clashes_unrelaxed if args.relax else num_clashes_relaxed,
                    'Relaxed_Clashes': num_clashes_relaxed,
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

                boltz_stats = (
                    boltz_stats_by_design_model.get((design_name, model_num))
                    or boltz_stats_by_design_model.get((design_name, None))
                )
                if boltz_stats:
                    complex_statistics[model_num].update(boltz_stats)
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
