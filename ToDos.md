Adjustable plddt and iptm filters during AF2 design protocol --> done


AF2 validation and filters outside MPNN loop --> done


Disulfide peptide generation
    - terminal cystein sequence init    --> done
    - disulfide loss    --> done
    - pyrosetta disulfide closure and relax --> done
    - mpnn fixation of C residues for disulfide cyclization --> done


Partial Hallucination for AF2 design protocol (to combine with RFDiffusion)


Filtering based on RMSD to designed structure


# Optimize ProteinMPNN usage (biases, ...)


Add pre-filtering step directly after AF2 design protocol (filtering out unwanted secondary structures etc.)


Select MPNN designs after generating and evaluating all of them.


Amino acid distribution losses during AF2 design protocol
    - simple distriution loss
    - ProteinMPNN distribution loss


Filters
    - disulfide cyclization filter
    - backbone filter
    - adjusted scores filters pLDDT, ipTM 
    - H-bonds, contact points filters


Analysis tools:
- binding site conservation
- binding site frustration
- rigidity