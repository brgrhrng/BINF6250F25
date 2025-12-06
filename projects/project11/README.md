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
parse_MSA(list of seqs):
	----------------------------------------------
	given: list of strings, or list of lists of chars
	populate prof hmm based on a list of aligned seqs
	
	this is a wrapper function to call private methods
	doing the initialization stuff. Builds HMM internally
	from provided list of aligned seqs
	----------------------------------------------
	# 1
	col_classifiers = _msa_to_classifiers(seqs)
	hmm attribute length = count of "M" in col_classifiers
	
	# 2
	_build_topology(col_classifers)
	
	# 3
	_estimate_parameters(seqs, col_classifiers)
	

_msa_to_classifiers(list of seqs):
	----------------------------------------------
	given: list of strings, or list of lists of chars
	return: list of "M" / "I" classifiers for building profile HMM
	
	internal use of a numpy array makes column (ie cross-row) operations
	much more efficient/easy
	-------------------------------------------------
	convert strings to character lists
	bundle lists into an np.array # will error if lists are unequal
	
	intialize col_classifiers as empty list
	for sliced column in array:
		gaps = count of "-"
		
		if 	gaps / length of col < 0.5
			append "M" to col_classifiers
		else:
			append "I" to col_classifiers
			
	return col_classifiers

	
_build_topology(list of "M" / "I" classifiers)
	----------------------------------------------
	given: list of "M" / "I" classifiers
	populate hiddenstates list attribute with M0, M/I/D for i:len, M{len+1}
	populate transitions dict according to profile HMM topology
	-------------------------------------------------
	# Initial state
	append "M0" to hiddenstates
	append "I0" to hiddenstates
	
	# Middle states
	match_index = 1
	for classifier in list of classifers:
		if classifer is "M":
			append "M+match_index" to hiddenstates
			append "I+match_index" to hiddenstates
			append "D+match_index" to hiddenstates
			
			# encode transitions into current M state
			add Mmi-1 -> Mmi to transitions
			add Imi-1 -> Mmi to transitions
			add Dmi-1 -> Mmi to transitions if match_index > 1
			
			# encode transitions into current D state
			add Mmi-1 -> Dmi to transitions
			add Imi-1 -> Dmi to transitions
			add Dmi-1 -> Dmi to transitions if match_index > 1
			
			# encode insertion transitions
			add Mmi -> Imi to transitions
			add Imi -> Imi to transitions
			add Dmi -> Imi to transitions
			
			increment match_index by 1
	
	# Terminal state
	append "M+match_index" to hiddenstates
	add Mmi-1 -> Mmi to transitions
	add Imi-1 -> Mmi to transitions
	add Dmi-1 -> Mmi to transitions



_estimate_parameters(seqs, col_classifiers):
	----------------------------------------------
	given: list of seqs, list of "M" / "I" classifiers
	estimate HMM attributes:
		init_probs, trans_probs, and emit_probs 
	-------------------------------------------------
	bundle lists into an np.array # will error if lists are unequal
	
	get global frequency of each residue in array
	
	# Init and emission probs
	# Initial state
	p(init)_Mm0 = 1
	set Mm0 and Em0 emission dicts to None 
	
	# Middle states
	match_index = 1
	for classifier in list of classifers:
		p(init)_Mmi = 0	
		if classifer is "M":
			for each unique character in column mi of seq array:
				p(Mmi emits char) = (char in col + b)/(char in col + 20*b) # b>0; prevents div by 0				
		
		elif classifier is "I":
			for each unique character in column mi of seq array:
				p(Imi emits char) = global frequency of character
			
			set Dmi emission dict to None # 1 D for every I
		
		increment match_index by 1
	
	# Terminal state
	p(init)_Mmi = 0
	set Mmi emission dict to None
	
	
	# Transition probs
	build labels array:
		if match col:
			replace non-"-" with Mi
			replace "-" with Di
		if insert col:
			replace non-"-" with Ii
	
	use labels array to populate transition probabilities
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
