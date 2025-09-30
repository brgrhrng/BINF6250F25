# BINF6250F25

# Introduction
Description of the project

# Pseudocode
Put pseudocode in this box:

```
# Function to identify a motif with length k, using Gibbs sampling,
#     from a list of DNA sequences.
GibbsMotifFinder(sequences, length k)
    motif_positions = empty list
    k_mers = empty list
    for seq in sequences:
        add random start position to motif_positions
        add seq[start:start+k] to k_mers

    for 10000 iterations, or until information content plateaus:
        randomly select a seq from sequences
        construct PWM from the k_mer for every other sequence 
        score all k-length substrings of seq against PWM
        use PWM weights to randomly select a substring
        update motif position, k_mer for seq to new substring

# Driver code
import sequences from data/seq_file
import gff-annotations from data/gff_file
promoters = new list
for each seq:
    extract coding sequences encoded in gff-annotations
    save first 50 bp from each coding sequence to promoters
generate a motif using GibbsMotifFinder(promoters, k=10)
generate sequence logo from motif


    
    
```

# Successes
Description of the team's learning points

# Struggles
Description of the stumbling blocks the team experienced

# Personal Reflections
## Group Leader
Group leader's reflection on the project

## Other member
Other members' reflections on the project

# Generative AI Appendix
As per the syllabus
