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



def get_sequence_from_pdb(pdb_path, chain_id):
    """Extracts sequence from ATOM records of a specific chain."""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        ppb = PDB.PPBuilder()
        for chain in structure[0]:
            if chain.id == chain_id:
                return "".join([str(pp.get_sequence()) for pp in ppb.build_peptides(chain)])
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
    return None

def create_boltz_yaml(output_path, target_seq, binder_seq, auto_disulfide, is_cyclic):
    """Generates a Boltz-1 YAML with optional cyclic and covalent constraints."""
    disulf_res = []
    if auto_disulfide:
        for index, aa in enumerate(binder_seq):
            if aa == "C":
                disulf_res.append(index + 1)
    
    manifest = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": target_seq, "msa": "empty"}},
            {"protein": {"id": "B", "sequence": binder_seq, "msa": "empty", "cyclic": is_cyclic}}
        ]
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

        binder_length = len(get_sequence_from_pdb(pred_dir / src_struct.name, "A"))
        target_length = len(get_sequence_from_pdb(pred_dir / src_struct.name, "B"))

        
        # Calculate mean pLDDT for target and binder, target residues are the first N residues (because chain A) and binder the last residues because chain B
        binder_plddt = np.mean(arr[target_length:target_length + binder_length])
        target_plddt = np.mean(arr[0:target_length]) 

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

    # Directories
    yaml_dir = work_dir / "yaml_inputs"
    results_dir = work_dir / "predictions"
    structure_dir = base_design_path / "Repredicted"
    
    for d in [yaml_dir, results_dir, structure_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    #Process PDBs and Fastas from input dirs

    pdb_files = list(input_pdbs_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"No PDBs found in {input_pdbs_dir}")
        return
    
    fasta_files = list(input_fastas_dir.glob("*.fasta"))
    if not fasta_files:
        print(f"No FASTA files found in {input_fastas_dir}")
        
    
    if settings["disulfide_num"] == 1:
        use_disulfide = True
    else:
        use_disulfide = False

    # Creating Boltz input yamls
    for pdb in pdb_files:
        name = pdb.stem
        t_seq = get_sequence_from_pdb(pdb, target_chain)
        b_seq = get_sequence_from_pdb(pdb, binder_chain)

        if t_seq and b_seq:
            create_boltz_yaml(
                yaml_dir / f"{name}.yaml",
                t_seq, b_seq,use_disulfide, cyclic
            )

    for fasta in fasta_files:
        name = fasta.stem.replace("_complete","")
        with open(fasta, 'r') as f:
            lines = f.readlines()
            seq = "".join([line.strip() for line in lines if not line.startswith(">")])
            t_seq = seq.split("/")[0]
            b_seq = seq.split("/")[1]
         
           
        if t_seq and b_seq:
            create_boltz_yaml(
                yaml_dir / f"{name}.yaml",
                t_seq, b_seq, use_disulfide, cyclic
            )



    # Run Prediction and Parse results
    try:
        run_boltz(yaml_dir, results_dir)
        parse_results(work_dir, base_design_path / "boltz_repredict_stats.csv", structure_dir)
    except Exception as e:
        print(f"Prediction or parsing failed: {e}")

if __name__ == "__main__":
    main()