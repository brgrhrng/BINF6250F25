# Introduction
Project to implement a Profile Hidden Markov Model by extending the provided HMM.py code base to implement a full profile HMM for protein motifs using a multiple sequence alignment (MSA) in FASTA format as input.

Concept Focus
A profile HMM is a position-specific HMM architecture tailored to model a conserved motif or domain from an MSA. Each alignment column becomes a block of three states (Match (Mi), Insertion (Ii), Deletion (Di)), with transitions constrained to flow left-to-right through the motif. This architecture allows the model to capture both column-wise conservation and localized insertions and deletions in homologous sequences.

Key Features
    * Position-specific emissions:
        ** Match states Mi use column-specific amino acid distributions estimated from the MSA.
        ** Insertion states Ii share a background composition independent of position.
    * Explicit gap handling:
        ** Deletion states Di are silent and model gaps by skipping columns without emitting residues.
    * Left-to-right topology:
        ** Structured transitions between Mi,Ii,Di encode allowed motif-length variation.
    * Probabilistic scoring:
        ** Forward, Viterbi, and Forward–Backward algorithms from HMM.py are reused to score sequences and, if desired, retrain parameters.

# Pseudocode
Put pseudocode in this box:

```
Some pseudocode here
```

# Successes
Description of the team's learning points

# Struggles
Description of the stumbling blocks the team experienced

# Personal Reflections
## Brooks
Group leader's reflection on the project

## Jacque
Other members' reflections on the project

# Generative AI Appendix
As per the syllabus
