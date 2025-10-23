# Introduction
In this project, we implemented a neighbor-joining algorithm to generate distance-based phylogenies for text sequences. The resulting trees are stored and returned as strings in newick format for easy rendering.

# Pseudocode

```
build_distance_matrix(sequences):
	""" Build a distance matrix using S-W scores, from a dictionary of named sequences """
	initialize n x n empty matrix as distance_matrix # n is number of seqs
	for i in seq_ids:
		for j in seq_ids:
			get the larger length between i and j
			dist_matrix at i,j = max_length - smith_waterman(seq_i, seq_j)

neighbor_joining(distance_matrix, newick_string = empty string):
	Recursive function to obtain a tree in newick format given a distance matrix.
  Each inner call carries out a single neighbor joining step, reducing the input matrix until we reach a top-level branch.

	if distance_matrix is 2x2:
		get top-level branch length from d_m
		update newick_string with final branch
		RETURN newick_string
	
	total_distances = sum of each taxon's distance in d_m	
	
	# Populate divergence matrix
	q_matrix = empty matrix
	for row in d_m:
		for col in d_m:
			if row == col: # self-comparisons
				d_m at row, col = 0	
			else:
				d_m at row,col - t_d[row] - t_d[col]
	
	find a minimum value in q_matrix
	get taxa1, taxa2 corresponding to this minimum

	# Calculate new limb lengths
	length1 = d_m at taxa1,taxa2 + t_d[taxa1] - t_d[taxa2]
	length2 = d_m at taxa1,taxa2 + t_d[taxa2]- t_d[taxa1]
	
	# Merge taxa into taxa column 1
	for row in d_m:
		row[taxa1] = row[taxa1] + row[taxa2] - d_m at taxa1,taxa2 /2
	drop row, col of taxa 2 from distance_matrix

	update newick_string with new branch and branch lengths
	
	# Recurse, using the updated distance_matrix and newick_string
	return neighbor_joining(distance_matrix, newick_string)


read_fasta(filename)
	""" Read sequences from FASTA file into a dictionary of id,seq pairs """
	out_dict = empty dict
	current_seq = empty string
	seq_id = None
	for line in file:
		if line starts with >:
			if we have a seq_id:
				save current_seq to out_dict[seq_id]
			empty current_seq
			extract a new seq_id from line
		else:
			append line to current_seq
	
	save final value of current_seq to out_dict[seq_id]
	return out_dict


# Main code
seq_dict = read_fasta(input .fasta)

dist_matrix = build_distance_matrix(seq_dict)

newick_tree = neighbor_joining(dist_matrix)
```

# Successes
Coding the neighbor-joining process went smoothly. We successfully produced distance matrices, calculated limb lengths based off of these matrices, and recursed this process. 

# Struggles
Formatting the leaves and their limb lengths so that ete3's Tree function could read and produce an unrooted Newick tree was our biggest struggle. Representing the limb lengths and their relation to the other nodes got complicated if the two taxa that were merging in the neighbor joining process were separate from the merged taxa created in a previous iteration. In the end, we could not find a method that universally handled the formatting, and instead implemented a couple if-statement conditions to handle these various cases. 

# Personal Reflections
## Group Leader (Brooks)
We completed the bulk of our project in two pair-programming sesions. Once we had iplemented a function that could do a single step of the neighbor-joining procedure, adapting it to operate recursively was actually surprizingly straightforward.

Even though Newick format itself is relatively easy to read and understand, generating these strings proved to the thorniest issue of the project. We spent a lot of time creating f-strings that would work perfectly for a particular test matrix and join-order, only to realize that they didn't generalize across other potential trees; it was simultaneously a fun puzzle to work through, and kinda infuriating. 

In retrospect, we may have been better served storing our tree in a custom object (Marcus did suggest an OOP approach....), rather than treating the newick string as our storage format. 

## Jason
Conceptually I thought this project was quite straightforward- the math for neighbor-joining was not too complex and combining branches as a form of making a common ancestor became easier to map out throughout the project. Coding the format for the Newick Tree was tough, especially when accounting for the different orders branches could combine in. In the end though, it was very rewarding fleshing out my understanding of phylogeny and how it works with bioinformatics.

# Generative AI Appendix
We did not use generative AI in creating this project.
