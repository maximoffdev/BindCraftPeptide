import os
import yaml
import time
import subprocess
import csv
import shutil
import json
import argparse
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, is_aa, PDBIO
from Bio.PDB.mmcifio import MMCIFIO
import gemmi
from Bio.SeqUtils import seq1
import io

# Required for sequence extraction in parse_results
three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
def save_pdb_with_seqres(pdb_path, output_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", str(pdb_path))
    
    seqres_lines = []
    
    for chain in structure.get_chains():
        chain_id = chain.id
        # Extract residues (ignoring HETATMs)
        residues = [res.get_resname() for res in chain.get_residues() if res.id[0] == " "]
        num_res = len(residues)
        
        # PDB SEQRES format: 13 residues per line
        # Format: SEQRES [Serial] [Chain] [NumRes] [Res1] [Res2]...
        for i in range(0, num_res, 13):
            line_res = residues[i:i+13]
            serial = (i // 13) + 1
            res_string = " ".join(f"{res:>3}" for res in line_res)
            # Line format according to PDB columns
            line = f"SEQRES {serial:>3} {chain_id} {num_res:>4}  {res_string:<52}\n"
            seqres_lines.append(line)

    # 1. Get the ATOM records using Biopython
    import io
    io_buffer = io.StringIO()
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(io_buffer)
    atom_data = io_buffer.getvalue()

    # 2. Combine SEQRES and ATOM data
    with open(output_path, "w") as f:
        f.writelines(seqres_lines)
        f.write(atom_data)




def create_boltz_yaml(output_path, target_seq, binder_seq, auto_disulfide, is_cyclic, use_template,template_struct):
    """Generates Boltz-1 YAML with cyclic/covalent constraints for BindCraft."""
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
        manifest["constraints"] = [{
            "bond": {
                "atom1": ["B", int(disulf_res[0]), "SG"],
                "atom2": ["B", int(disulf_res[1]), "SG"]
            }
        }]

    if use_template:
            template_dir = Path("templates")
            template_dir.mkdir(exist_ok=True)
            target_pdb = template_dir / "template.pdb"

            save_pdb_with_seqres(template_struct, target_pdb)

            manifest["templates"] = [
                {
                    "pdb": str(target_pdb.resolve())
                }
            ]
    # Ensure parent directory exists for the yaml
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def run_boltz(yaml_dir, output_dir, conda_env="boltz"):
    """Runs Boltz via 'conda run'."""
    start_time = time.time()
    cmd = [
        'conda', 'run', '-n', conda_env, 
        'boltz', 'predict', str(yaml_dir), 
        "--out_dir", str(output_dir), 
        "--use_potentials", 
        "--output_format", "pdb"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: Boltz failed. Error: {e}")
        raise
    print(f"Completed in {time.time() - start_time:.2f}s")

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

def parse_results(boltz_out_dir, csv_path, structures_dir, get_sequence_func):
    """Parses Boltz outputs and extracts pLDDT for target and binder."""
    boltz_out_dir = Path(boltz_out_dir)
    structures_dir.mkdir(parents=True, exist_ok=True)
    
    prediction_dirs = list(boltz_out_dir.glob("predictions/*"))
    all_data = []
    
    headers = [
        "Design",'binder_sequence' ,"boltz_confidence_score", "boltz_complex_pDE", "boltz_complex_iPDE", "boltz_pTM_complex", "boltz_ipTM_complex", 
        "boltz_pLDDT_complex", "boltz_ipLDDT_complex", "boltz_pLDDT_target", "boltz_pLDDT_binder"
    ]

    for pred_dir in prediction_dirs:
        struct_files = list(pred_dir.glob("*.pdb"))
        if not struct_files: continue
            
        src_struct = struct_files[0]
        dest_name = src_struct.name.replace('_model_0', '')
        shutil.copy2(src_struct, structures_dir / dest_name)
        
        plddt_files = list(pred_dir.glob("plddt*"))
        if not plddt_files: continue
        data = np.load(plddt_files[0].resolve())
        arr = data[data.files[0]]

        target_len = len(get_sequence_func(src_struct, "A"))
        binder_len = len(get_sequence_func(src_struct, "B"))

        target_plddt = np.mean(arr[0:target_len]) 
        binder_plddt = np.mean(arr[target_len:target_len + binder_len])

        json_files = list(pred_dir.glob("*.json"))
        if not json_files: continue

        with open(json_files[0], 'r') as f:
            m = json.load(f)
            all_data.append({
                "Design": dest_name.replace('.pdb', ''),
                "binder_sequence": get_sequence_func(src_struct, "B"),
                "boltz_confidence_score": m.get("confidence_score"),
                "boltz_pTM_complex": m.get("ptm"),
                "boltz_ipTM_complex": m.get("iptm"),
                "boltz_complex_pDE": m.get("complex_pde"),
                "boltz_complex_iPDE": m.get("complex_ipde"),
                "boltz_pLDDT_complex": m.get("complex_plddt"),
                "boltz_ipLDDT_complex": m.get("complex_iplddt"),
                "boltz_pLDDT_target": target_plddt,
                "boltz_pLDDT_binder": binder_plddt
            })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Boltz Prediction Tool")
    parser.add_argument("--template", help="Path to template PDB file")
    parser.add_argument("--out_dir", default="boltz_results", help="Directory for all outputs")
    parser.add_argument("--cyclic", action="store_true", help="Peptide is cyclic")
    parser.add_argument("--disulfide", action="store_true", help="Auto-detect disulfides (C-C)")

    args = parser.parse_args()

    # Define paths
    base_dir = Path(args.out_dir)
    yaml_dir = base_dir / "yamls"
    boltz_out = base_dir / "boltz_raw"
    final_pdbs = base_dir / "pdbs"
    summary_csv = base_dir / "summary.csv"
    input_yaml = yaml_dir / "input.yaml"

    target_seq = extract_chain_sequence_from_pdb(args.template, "A") if args.template else "PLACEHOLDER_TARGET_SEQ"
    binder_seq = extract_chain_sequence_from_pdb(args.template, "B") if args.template else "PLACEHOLDER_BINDER_SEQ"
    # Execution steps
    create_boltz_yaml(
        input_yaml, 
        target_seq, 
        binder_seq, 
        args.disulfide, 
        args.cyclic, 
        bool(args.template), 
        args.template
    )
    
    run_boltz(yaml_dir, boltz_out)
    
    parse_results(
        boltz_out, 
        summary_csv, 
        final_pdbs, 
        extract_chain_sequence_from_pdb
    )
    
    print(f"Done! Results in {args.out_dir}")