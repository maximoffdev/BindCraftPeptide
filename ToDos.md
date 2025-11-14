Adjustable plddt and iptm filters during AF2 design protocol --> done


AF2 validation and filters outside MPNN loop --> done


Disulfide peptide generation
    - terminal cystein sequence init    --> done
    - disulfide loss    --> done
    - pyrosetta disulfide closure and relax --> done
    - mpnn fixation of C residues for disulfide cyclization --> done


Partial Hallucination for AF2 design protocol (to combine with RFDiffusion)
    - protocol implementation   --> done
    - disulfide fixation (simplification of specification for head to tail cycl)    --> done
    - test  -->
    - ignore missing implementation


Motif sequence fixation
    - in advanced binder design protocol
    - for proteinMPNN optimization


disulfide check in filters if disulfide bridge has formed


+ length hallucination to provided binder motif in advanced binder design protocol


Add pre-filtering step directly after AF2 design protocol    --> done


Add pre pre filtering of input pdb directly after secondary structure (for RFDiffusion inputs)


Select MPNN designs after generating and evaluating all of them.


Amino acid distribution losses during AF2 design protocol
    - simple distriution loss   --> done
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