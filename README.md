# BINF6250F25

# Introduction
Description of the project

# Pseudocode
Put pseudocode in this box:

```
1. import sequences from file

2. for each sequence, pick a random start index

3.

GibbsMotifFinder(sequences, k-length)
    # create a list of possible motifs from each sequence
    k_mers = empty list
    
    for seq in sequences
      get random index
      add seq[random_i:random_i+k] to k_mers
    
    for 10000 iterations, or until information content plateaus:
      randomly select a sequence from sequences
      construct PWM from the k_mer associated with every other sequence # Chris
      scores = [score each k_mer using PWM] # BAG
      select random k_mer weighted by scores
      
      check if new IC approx equals last ic # info plateau
    
    
    
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