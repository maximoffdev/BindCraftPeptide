#!/usr/bin/env python3
import os
import yaml
import json
import csv
import glob
import subprocess
import argparse
from pathlib import Path
import shutil
import numpy as np 
import time
from Bio import PDB



def _write_clean_template_pdb(
    template_pdb_path: Path,
    output_pdb_path: Path,
    chain_id: str | None,
) -> Path:
    """Write a cleaned template PDB to avoid Boltz template parsing crashes.

    The cleaning is intentionally conservative:
    - Keep only ATOM records
    - Keep only standard amino-acid residues
    - Optionally keep only a single chain
    - Renumber residues sequentially starting at 1 (removes insertion-code weirdness)

    Boltz's template parser can crash on some PDBs that contain non-standard residues,
    insertion codes, or mixed polymer records. Providing a minimal single-chain template
    is typically sufficient and more robust.
    """

    template_pdb_path = Path(template_pdb_path).expanduser().resolve()
    output_pdb_path = Path(output_pdb_path).expanduser().resolve()
    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("template", str(template_pdb_path))
    model = next(structure.get_models())

    if chain_id is not None and chain_id not in [c.id for c in model.get_chains()]:
        raise ValueError(
            f"Requested template chain '{chain_id}' not found in {template_pdb_path}"
        )

    # If the user requests a specific chain, write the cleaned template as a single
    # chain with ID 'A'. Boltz templates are easiest to work with when the template
    # chain ID matches the query chain ID used in the YAML (A).
    output_chain_id = "A" if chain_id is not None else None

    # Renumber residues in-place (on the parsed structure) for the selected chain(s).
    for chain in model.get_chains():
        if chain_id is not None and chain.id != chain_id:
            continue

        if output_chain_id is not None:
            chain.id = output_chain_id

        new_resseq = 1
        for residue in list(chain.get_residues()):
            if not PDB.is_aa(residue, standard=True):
                continue
            hetflag, _resseq, _icode = residue.id
            residue.id = (" ", int(new_resseq), " ")
            new_resseq += 1

    class _TemplateSelect(PDB.Select):
        def accept_chain(self, chain):
            if chain_id is None:
                return True
            return chain.id == (output_chain_id or chain_id)

        def accept_residue(self, residue):
            return PDB.is_aa(residue, standard=True)

        def accept_atom(self, atom):
            # Drop alternate locations other than blank or 'A'
            altloc = atom.get_altloc()
            return altloc in (" ", "A")

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb_path), select=_TemplateSelect())
    return output_pdb_path


def get_sequence_from_pdb(pdb_path, chain_id):
    """Extract sequence for a chain by iterating residues.

    Notes:
    - We intentionally do NOT use Bio.PDB.PPBuilder here because it can drop residues
      when backbone atoms are missing, which can make the YAML sequence shorter than
      the template residues and trigger Boltz template parsing/indexing errors.
    - We include only standard amino acids (plus a small set of common aliases).
    """

    three_to_one = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
        # common non-canonical residue names that should map cleanly
        "MSE": "M",
    }

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", str(pdb_path))
        model = structure[0]

        chain = None
        for c in model:
            if c.id == chain_id:
                chain = c
                break
        if chain is None:
            return None

        seq = []
        for residue in chain.get_residues():
            hetflag, _resseq, _icode = residue.id
            if hetflag != " ":
                continue
            resname = residue.get_resname().upper()
            aa = three_to_one.get(resname)
            if aa is None:
                # Skip non-standard residues (keeps consistency with template cleaning).
                continue
            seq.append(aa)

        return "".join(seq) if seq else None
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return None

def create_boltz_yaml(
    output_path,
    target_seq,
    binder_seq,
    auto_disulfide,
    is_cyclic,
    template_pdb_path,
    template_chain_id=None,
    templates_dir=None,
):
    """Generates a Boltz-1 YAML with optional cyclic and covalent constraints.

    A template for chain A is ALWAYS included (forced) using the provided template PDB.
    """
    disulf_res = []
    if auto_disulfide:
        for index, aa in enumerate(binder_seq):
            if aa == "C":
                disulf_res.append(index + 1)

    template_pdb_path = Path(template_pdb_path).expanduser().resolve()
    if not template_pdb_path.exists():
        raise FileNotFoundError(f"Template PDB does not exist: {template_pdb_path}")

    # Always provide a cleaned template to Boltz (more robust).
    if templates_dir is None:
        templates_dir = Path(output_path).parent / "templates"
    else:
        templates_dir = Path(templates_dir)

    cleaned_template = _write_clean_template_pdb(
        template_pdb_path=template_pdb_path,
        output_pdb_path=templates_dir / f"{Path(output_path).stem}_template.pdb",
        chain_id=template_chain_id,
    )

    # Defensive check: ensure the query sequence for chain A matches the template residue count.
    # If they differ (often due to missing backbone atoms and PPBuilder-style extraction),
    # Boltz may crash while indexing template residues against the query sequence.
    template_seq = get_sequence_from_pdb(cleaned_template, "A")
    if template_seq is not None and len(template_seq) != len(target_seq):
        print(
            f"[WARN] Template/query length mismatch for '{Path(output_path).name}': "
            f"len(template)={len(template_seq)} vs len(target_seq)={len(target_seq)}. "
            "Omitting template to avoid Boltz template parsing crash."
        )
        cleaned_template = None
    
    manifest = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": target_seq, "msa": "empty"}},
            {"protein": {"id": "B", "sequence": binder_seq, "msa": "empty", "cyclic": is_cyclic}}
        ],
        # Prefer a single-chain cleaned template when compatible; otherwise omit templates.
        "templates": [] if cleaned_template is None else [
            {
                # NOTE: We intentionally pass only a cleaned single-chain template PDB
                # and do NOT set force/chain_id/template_id here.
                "pdb": str(cleaned_template),
            }
        ],
    }
        
    if len(disulf_res) == 2:
        manifest["constraints"] = [
            {
                "bond": {
                    "atom1": ["B", int(disulf_res[0]), "SG"],
                    "atom2": ["B", int(disulf_res[1]), "SG"]
                }
            }
        ]
    
    with open(output_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

def run_boltz(yaml_dir, output_dir):
    """Executes Boltz CLI using the absolute path to the executable."""
    start_time =time.time()


    cmd = ['boltz', "predict", str(yaml_dir), "--out_dir", str(output_dir), "--use_potentials", '--output_format','pdb']
    print(f"Running Boltz: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    end_time = time.time()
    print(f"Prediction of {len(list(yaml_dir.glob('*')))} structures completed in {end_time - start_time:.2f} seconds")

def parse_results(boltz_out_dir, csv_path, structures_dir):
    boltz_out_dir = Path(boltz_out_dir)
    structures_dir.mkdir(parents=True, exist_ok=True)
    
    #Find prediction directories in the nested boltz output
    prediction_dirs = list(boltz_out_dir.glob("predictions/*/predictions/*"))
    all_data = []
    headers = [
        "Design", "confidence_score", "pTM_complex", "ipTM_complex", 
        "pLDDT_complex", "ipLDDT_complex", "pDE_complex", "ipDE_complex", 
        "pTM_target", "pTM_binder", "pLDDT_target", "pLDDT_binder"
    ]

    for pred_dir in prediction_dirs:
        struct_files = list(pred_dir.glob("*.pdb"))
        if not struct_files: continue
            
        src_struct = struct_files[0]
        shutil.copy2(src_struct, structures_dir / src_struct.name.replace('model_0',''))
        
        plddt_files = list(pred_dir.glob("plddt*"))
        if not plddt_files: continue
        
        data = np.load(plddt_files[0].resolve())
        arr = data[data.files[0]]

        # In BindCraft/Boltz YAML we use chain A=target and chain B=binder.
        target_seq = get_sequence_from_pdb(pred_dir / src_struct.name, "A")
        binder_seq = get_sequence_from_pdb(pred_dir / src_struct.name, "B")
        if not target_seq or not binder_seq:
            continue

        target_length = len(target_seq)
        binder_length = len(binder_seq)

        
        # Calculate mean pLDDT for target and binder, target residues are the first N residues (because chain A) and binder the last residues because chain B
        binder_slice = arr[target_length:target_length + binder_length]
        target_slice = arr[0:target_length]
        if len(binder_slice) == 0 or len(target_slice) == 0:
            continue
        binder_plddt = float(np.mean(binder_slice))
        target_plddt = float(np.mean(target_slice))

        json_files = list(pred_dir.glob("*.json"))
        if not json_files: continue

        with open(json_files[0], 'r') as f:
            m = json.load(f)
            row = {
                "Design": pred_dir.name.replace('model_0',''),
                "confidence_score": m.get("confidence_score"),
                "pTM_complex": m.get("ptm"),
                "ipTM_complex": m.get("iptm"),
                "pLDDT_complex": m.get("complex_plddt"),
                "ipLDDT_complex": m.get("complex_iplddt"),
                "pDE_complex": m.get("complex_pde"),
                "ipDE_complex": m.get("complex_ipde"),
                "pLDDT_target": target_plddt,
                "pLDDT_binder": binder_plddt,
                "pTM_target": m.get("chains_ptm", {}).get("0"),
                "pTM_binder": m.get("chains_ptm", {}).get("1"),
            }
            all_data.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_data)






def main():
    parser = argparse.ArgumentParser(description="Boltz-1 Reprediction Pipeline")
    parser.add_argument("--settings", "-s", required=True, help="Path to settings.json")
    parser.add_argument("--design_path", "-d", default=None, help="Optional override for design_path in settings.json")
    
  
    args = parser.parse_args()

  
    with open(args.settings, 'r') as f:
        settings = json.load(f)

    # Map settings to variables
    
    if args.design_path is not None:
        settings["design_path"] = args.design_path
    
    base_design_path = Path(settings["design_path"])
    input_pdbs_dir = base_design_path / "Trajectory" 
    input_fastas_dir = base_design_path / "trajectory_fastas" 
    cyclic = settings.get("cyclic", False) # for optional head to tail cyclised designs


    # If the MPNN folder doesn't exist, fall back to the base design path
    if not input_pdbs_dir.exists():
        input_pdbs_dir = base_design_path / "Repredicted" / "Relaxed"

    work_dir = base_design_path / "boltz_reprediction"
    target_chain = settings.get("chains", "A")
    binder_chain = "B" # Defaulting to B as per standard design outputs

    # Template chain: by default use the target chain. This produces a single-chain
    # template file which is more robust and still lets Boltz assign chains.
    template_chain = settings.get("template_chain", target_chain)

    # Directories
    yaml_dir = work_dir / "yaml_inputs"
    results_dir = work_dir / "predictions"
    structure_dir = base_design_path / "Repredicted"
    
    for d in [yaml_dir, results_dir, structure_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    #Process PDBs and Fastas from input dirs

    pdb_files = list(input_pdbs_dir.glob("*.pdb"))
    fasta_files = list(input_fastas_dir.glob("*.fasta"))
        
    
    if settings["disulfide_num"] == 1:
        use_disulfide = True
    else:
        use_disulfide = False

    # Creating Boltz input yamls
    if pdb_files:
        for pdb in pdb_files:
            name = pdb.stem
            t_seq = get_sequence_from_pdb(pdb, target_chain)
            b_seq = get_sequence_from_pdb(pdb, binder_chain)

            if t_seq and b_seq:
                create_boltz_yaml(
                    yaml_dir / f"{name}.yaml",
                    t_seq,
                    b_seq,
                    use_disulfide,
                    cyclic,
                    template_pdb_path=pdb,
                    template_chain_id=template_chain,
                    templates_dir=work_dir / "templates",
                )
    else:
        if not fasta_files:
            print(f"No PDBs found in {input_pdbs_dir} and no FASTA files found in {input_fastas_dir}")
            return

        for fasta in fasta_files:
            name = fasta.stem.replace("_complete", "")
            with open(fasta, 'r') as f:
                lines = f.readlines()
                seq = "".join([line.strip() for line in lines if not line.startswith(">")])
                t_seq = seq.split("/")[0]
                b_seq = seq.split("/")[1]

            if t_seq and b_seq:
                template_pdb = input_pdbs_dir / f"{name}.pdb"
                if not template_pdb.exists():
                    print(
                        f"[WARN] Skipping FASTA '{fasta.name}' because required template PDB was not found: {template_pdb}"
                    )
                    continue

                create_boltz_yaml(
                    yaml_dir / f"{name}.yaml",
                    t_seq,
                    b_seq,
                    use_disulfide,
                    cyclic,
                    template_pdb_path=template_pdb,
                    template_chain_id=template_chain,
                    templates_dir=work_dir / "templates",
                )



    # Run Prediction and Parse results
    try:
        run_boltz(yaml_dir, results_dir)
        parse_results(work_dir, base_design_path / "boltz_repredict_stats.csv", structure_dir)
    except Exception as e:
        print(f"Prediction or parsing failed: {e}")

if __name__ == "__main__":
    main()