#!/usr/bin/env python3
"""Renumber and relabel chains in a PDB.

Given an input PDB path, writes a new PDB in the same directory named
"<stem>_renumbered.pdb" where:

- Chains are renamed sequentially: A, B, C, ...
- Residues within each chain are renumbered starting from 1.

Notes:
- The PDB format only supports single-character chain IDs. If the input
  contains more than 26 chains, this script will raise an error.
- Renumbering clears insertion codes (sets them to a blank space) to ensure
  strictly sequential residue numbers.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure


def _chain_id_from_index(idx0: int) -> str:
	"""0-based index -> PDB chain ID (A..Z).

	PDB chain IDs are single-character; we support up to 26 chains.
	"""

	if idx0 < 0:
		raise ValueError("chain index must be >= 0")
	if idx0 >= 26:
		raise ValueError(
			f"Too many chains ({idx0 + 1}); PDB only supports 26 single-letter chain IDs (A-Z)."
		)
	return chr(ord("A") + idx0)


def renumber_pdb(in_pdb: str | Path) -> Path:
	in_path = Path(in_pdb)
	if not in_path.exists():
		raise FileNotFoundError(str(in_path))
	if in_path.suffix.lower() != ".pdb":
		# still allow it, but keep output naming consistent
		pass

	out_path = in_path.with_name(f"{in_path.stem}_renumbered.pdb")

	parser = PDBParser(QUIET=True)
	structure = parser.get_structure("in", str(in_path))

	# Work on first model for PDB output.
	model0 = next(structure.get_models())

	new_structure = Structure("renumbered")
	new_model = Model(0)
	new_structure.add(new_model)

	# Preserve chain iteration order as provided by Biopython.
	chains = list(model0.get_chains())
	for chain_index, chain in enumerate(chains):
		new_chain_id = _chain_id_from_index(chain_index)
		new_chain = Chain(new_chain_id)

		new_resseq = 1
		for residue in chain.get_residues():
			# Copy residue and renumber.
			new_residue = copy.deepcopy(residue)
			# deepcopy can retain a copied parent/chain; detach so renumbering doesn't
			# collide with "sibling" residue IDs inside that copied parent.
			try:
				new_residue.detach_parent()
			except Exception:
				pass
			hetflag, _old_resseq, _old_icode = new_residue.id
			new_residue.id = (hetflag, int(new_resseq), " ")
			new_chain.add(new_residue)
			new_resseq += 1

		new_model.add(new_chain)

	io = PDBIO()
	io.set_structure(new_structure)
	io.save(str(out_path))
	return out_path


def main() -> None:
	ap = argparse.ArgumentParser(
		description=(
			"Renumber each chain starting at residue 1 and relabel chains A, B, C... "
			"Writes <input_stem>_renumbered.pdb next to the input."
		)
	)
	ap.add_argument("pdb", help="Path to input PDB file")
	args = ap.parse_args()

	out_path = renumber_pdb(args.pdb)
	print(str(out_path))


if __name__ == "__main__":
	main()
