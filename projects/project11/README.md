# Introduction
Project to implement a Profile Hidden Markov Model by extending the provided HMM.py code base to implement a full profile HMM for protein motifs using a multiple sequence alignment (MSA) in FASTA format as input.

Concept Focus
A profile HMM is a position-specific HMM architecture tailored to model a conserved motif or domain from an MSA. Each alignment column becomes a block of three states (Match (Mi), Insertion (Ii), Deletion (Di)), with transitions constrained to flow left-to-right through the motif. This architecture allows the model to capture both column-wise conservation and localized insertions and deletions in homologous sequences.

Key Features
* Position-specific emissions:
<<<<<<< HEAD
...Match states Mi use column-specific amino acid distributions estimated from the MSA.
...Insertion states Ii share a background composition independent of position.
* Explicit gap handling:
...Deletion states Di are silent and model gaps by skipping columns without emitting residues.
* Left-to-right topology:
...Structured transitions between Mi,Ii,Di encode allowed motif-length variation.
* Probabilistic scoring:
... Forward, Viterbi, and Forward–Backward algorithms from HMM.py are reused to score sequences and, if desired, retrain parameters.
=======
** Match states Mi use column-specific amino acid distributions estimated from the MSA.
** Insertion states Ii share a background composition independent of position.
* Explicit gap handling:
** Deletion states Di are silent and model gaps by skipping columns without emitting residues.
* Left-to-right topology:
** Structured transitions between Mi,Ii,Di encode allowed motif-length variation.
* Probabilistic scoring:
** Forward, Viterbi, and Forward–Backward algorithms from HMM.py are reused to score sequences and, if desired, retrain parameters.
>>>>>>> f9d47a837e18819e8c958b4032137161ffb1878e

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
	
	num_hidden_states = length(self.hidden_states)
	L = length(seqs[0])
	
	get whole_alphabet for emissions with size
	get alphabet_size
	get_global_frequency of each unique residue in each seq of seqs (this function should normalize these)
	    return this in a list_of_all that matches length of alphabet or in dictionary
	            if we could return this with a minium value 
    if we only do this on the frequency of residues from our sequences, then...
        smallest_residue = find min of list of all / 2
	
	# Initial state
    Initialize init_probs M0 = 1.0 
        all other hiddenstates = 0
        init_state emission dict = 0 
	
	# Middle states
	## Emission Matrix
	create np.ndarray.fill(smallest_residue,hiddenstates)
	
	#now we will overwrite the matrix with our seq based residue "emissions"
	for seq_index, classifier in list of col_classifers:
        Get the residues in the seq_index column of sequences # for seq in seqs; residues+=seq[seq_index]
	    if classifer is "M":
			for each unique character/aa in residues: # set
			    if not gap:
			        key = aa
			        count = residues.count(key) # string.count(char)
			        value = (count + b)/(length(residues) + alphabet_size*b) # b>0;prevents div by 0		
		elif classifier is "I":
		
			for each unique character/aaa in residues: #set
			    if not gap:
			        key = aa
			        value = lookup in global residue dictionary from above
			
		#else:	set Dmi emission dict to None 
	Normalize emissions matrix after finishing
		
    ## Transition Matrix - first build the labels array ; 
                            count them ; then calc probs
	build labels array:
        From labeled paths (seq_labels)
            count states observed
        create paired list from labeled paths
            count state_transitions observed (this is c(s,s') across all sequences)
        Convert counts to probabilities with pseudocounts:
            a(s,s') = c(s,s') + b / summation states[c(s,s') + b*size_of_alphabet]
          ie: A_m or the transition matrix comes from counting the state_transitions
        Each row of the transition matrix must sum to 1.0 I.e NORMALIZED
        put matrix into dictionary
        return transition probs in dictionary
			
	# Terminal state
	end probabilities = 0
	set end state emission dict to None


```

# Notes
According to the assigment:
The required functionality is:
At a minimum:
A. Read MSA FASTA file and build a profile HMM (done, wrapping the BaseHMM/HMM classes)
B. Export learned init_probs, trans_probs, and emit_probs (done: output1.txt) 
C. Given new protein sequences (assume "testing" sequences):
* Compute likelihood via HMM.py's forward method (see note2 below)
* Compute the most likely path (Viterbi) and interpret match vs insertion vs deletion positions (again, see note1 below)
D. Provide a short script or notebook demostrating:
* Model construction from the training MSA
* Scoring of positive test sequences that contain the motif
* scoring of negative control sequences that lack the motif (or have disrupted motif)

Note1: File output for showing 'init_probs', 'emission_probs' and 'transition_probs' of the two training sequences is in "output1.txt".

Note2: As we decided to implement our Class with the dictionary format that seems to be preferred by the HMM classes, we created list of Dicts of Dicts as previously noted, however our implementation's "M0" (initialization state) emission probability is set to None - which does not work with as written HMM.py code (it fails in the second line as the M0 "None" is not subscriptable into any residues). 

While I could (and did) try to set the M0 emission states to our default, the error continues to be a problem when forward gets to the emission probabilities for the delete states.  The next idea I had was to decrease the size of the the "hidden_states" by all of the 'None' states.  Running this with the the first test sequence, forward still gets an error on the first line.   We just ran out of time.

# Successes
We have something to present.   I'm not sure it's pretty or complete or even useful, but it's there.  Creates a model given sequences from the FASTA files.  Unable to make this run with forward() or the other HMM functions, but I think we were on the edge of a break through.

# Struggles
Struggled with integrating all of the information given regarding this project (and the many different ways of saying similar things), with varying success.  Understanding mathematical equations when they are written in different ways with different variables, and in different contexts was particularly difficult with this one.  

Downtime after 7pm on Sunday night because the machine hosting the OOD server was down because it's domain certification was lost, and parts of the internet didn't trust the computer to do DNS, etc. Unfortunate timing.  Another unavailable OOD server/Explorer 12/10/25 AM ~ 1hour (proxy server errors, unable to log in)

# Personal Reflections
## Brooks
Group leader's reflection on the project

## Jacque
I'm tired.  Not much left to give regarding reflections.  This sounded like a basic 
implementation, but was not quite as straight forward as that.  Additional requirements 
for this week, including having to review a three week project x 2; didn't help the time 
requirement issue, and neither did cluster downtime.

# Generative AI Appendix
(jacque) Claude.ai input on Profile HMM contexts.
