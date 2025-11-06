# Module 08: HMMs and the Viterbi Algorithm
- BINF6250; Fall 2025
- Authors: Brooks Groharing and Jacqueline Caldwell
- Due:  November 5, 2025

# Introduction
For this project, we created a custom class to store a Hidden Markov Model (HMM), and implemented a method in this class that uses the Viterbi algorithm to find the most likely sequence of hidden states from the model to explain a provided sequence of observations.

The Viterbi algorithm is a dynamic programming approach predicting hidden states, offering:
* Optimal path finding (finding the most probable sequence of hidden states)
* Dynamic programming (using tabulation to avoid redundant calculations)
* Log-space computation (to prevent numerical underflow)
* Traceback Mechanism (reconstructs the optimal state path after computation is complete)

## Further information (provided to us in the assignment description):
### Basic Algorithm Structure
* **Initialization**: Create two matrices with columns equal to the number of observations, and rows equal to the possible states from the model.
	* **Viterbi Matrix**: store the cumulative probability of the most probable sequence of states leading to the state/observation represented in a given cell. Initialize first column such that:

		$$p_{obs,  state} = p_{0}(state) * p(observation | state)$$

	* **Traceback Matrix**: stores pointers to the previous state/observation (cell) in the most probable path leading into each given cell. Initialize first column as -1 (or other value representing a "dead end"--it doesn't really matter, since we don't look back past column 1)

* **Propagation**: For each observation and possible state (or, cell) in the matrix, calculate the cumulative probability of each possible path leading into this observation/state as:

	$$p_{obs, state|path} = p_{obs-1,state-1} * p_{transition}(state_{current}, state_{prior}) * p_{emission|state} $$

	Store the largest p(path) from the previous column in Viterbi Matrix at the current cell, and a pointer to the the prior observation/state in the best path to the Traceback Matrix.

* **Termination:**

	* Identify the final state with the highest probability.

	* Starting from this cell, follow the pointers in Traceback Matrix to reconstruct the sequence of states in the optimal path.

## Computational Considerations:

Important factors to consider in implementation:
* Time complexity : O(NxK^2) where N is sequence length and K is number of states
* Space complexity: O(NxK) for storing the dynamic programming matrix
* Numerical Stability:  Using log probabilities to prevent underflow
* Edge Cases:  Handling zero probabilities with pseudocounts 


### For further discussion/information on Viterbi
See notes/virterbi_info.Rmd

# Class Outline/Pseudocode
## markov_models.py
```{Python}
class HMM: # A Hierarchical Markov Model object
  DATA:
  hidden_states: list containing HiddenState objects
  possible_emissions: list of values that every HiddenState might emit

  METHODS--
	init(intial probabilities, transition probabilities, emission probabilities):
		Object constructor; Parses provided dictionaries (see input data format below) to generate HiddenState objects and an emissions list.
	run_viterbi(list of observations):
		Construct viterbi and backpointer matrix (Initialize + Propagate)
      	Trace through backpointers to reconstruct most probable states at each observation index (Termination)
      	RETURN list of state names

class HiddenState:	# A single state that can be stored in an HMM
	DATA:
		Name: string representing state name
		Initial Prob: p(this state) at observation 0
		Emissions_Dict: 
		Transition_To (optional): a dict of ("state2", probability) pairs representing out-bound state transitions
	METHODS:
		init(name, init_prob, emission_probs):
			Object constructor; saves arguments to internal data.
			Initialize Transition_To as an empty dict, to allow for creation of "dead end states"
		set_transition(transition_dict):
			Load in a transition dictionary, and update Transition_To.
		emit():
			Randomly generate an emission from this state weighted by emission_probs 
			This is basic functionality for an HMM state; not strictly needed for Viterbi, but useful
```

## Project08.rmd
```
import markov_models

# Initialize a markov model with provided data
init_probs = {"I": 0.2, "G": 0.8}
trans_probs = {
    "I": {"I": 0.7, "G": 0.3},
    "G": {"I": 0.1, "G": 0.9}}
emit_probs = {
    "I": {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1},
    "G": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}}

hmm = new HMM(init_probs, trans_probs, emit_probs)

hmm.run_viterbi(list of observations)
```

## For our project Style Guide
See notes/style.Rmd


# Successes
Overall:
* Successfully designed and implemented an HMM object, including all data and methods from our initial project schematic.
* Implemented viterbi using a dynamic programming matrix, per the project description.
* Used a mix of pair programming, and asynchronous development + reviews of each others code to create a project we are both proud of.
  
Jacque:
* learned how to add a new branch to a prior repo fork on github.
* Wrangled a multi for-loop extravaganza into submission.

Brooks:
* became a lot more comfortable working with numpy arrays
* used vector operations to simplify the looping inside fill_viterbi(). Theoretically, this should be faster--but more importantly it *looks* much more like the viterbi equations, which really improves code readability.


# Struggles
Overall:
* Indexing into multiple arrays at the same time, during multiple loops, remembering what order the indexes need to be in for each structure.
* Trying to remain flexible with class structure/infrastructure when we're not quite sure what the next steps are going to be.
* LOG-PROBABILITIES: We implemented a flag to store/calculate our Viterbi probabilites in log10-space internally, to avoid underflow errors in long observation seqs. While our math *looks* right at every step we've examined, the end result is an incorrect path.
  * For now, log_values defaults to False, so our working flat-probability implementation is used. But we would really like to fix our log implementation and make it True by default. 

Jacque:
* continues to find it difficult to visualize classes/objects, and items/methods that
are private/hidden, but Brooks helped.  A lot.

Brooks:
* indexing. oh lord, indexing. So much of our debugging effort went towards figuring out why we weren't slices our arrays properly--and trying to verbally describe and pin down columns/rows/horizontal/vertical--and keep them straight in our heads--was a major pain. However, iterating through this struggle eventually resulted in some really solid and explicit variable names.

# Personal Reflections
## Group Leader (Brooks)
This project was in some ways a pain to implement, but it also presented a fairly interesting series of puzzles to solve. Bouncing ideas back and forth with Jacque was incredibly helpful. In theory I think either of us could have struggled through the code on our own, but by essentially peer reviewing each others' work we ended up with a stronger end product. 

We designed our classes with code readability and modularity in mind--my hope is that this will make expanding upon them in the next two projects easier. We had to resist the urge to indulge too much in premature optimization, but we should be in a good place if Marcus asks us (for example) to make our HMMs trainable instead of mandating hardcoded probabilities. I'm exciting to see what curveballs he throws us.

## Other member (Jacque)
The algorithm is not the problem; getting it implemented is the problem.  Arrays of arrays of arrays of arrays of arrays (okay lists of lists of dicts).  Indexing gets complicated.  Our focus on the module/problem, as on our last assignment together, is very different, but complimentary, allowing us to get the coding done, and make sure the details are smoothed over, without getting bogged down in the details of either.

# Generative AI Appendix and references
Discussions with GPT-5 to cement my (Jacque’s) understanding of Virterbi equations relationships to the data that we will be given, on-line through outlier.ai, October 30, 2025.

Discussions with Claude.ai again to cement my (Jacque’s) understanding of Virterbi equations and relationships, on-line through outlier.ai, October 30, 2025.

## Other sources:
https://www.youtube.com/watch?v=6JVqutwtzmo; Viterbi Algorithm; Keith Chugg (USC) ; posted March 9, 2017

https://www.youtube.com/watch?v=s9dU3sFeE40; Hidden Markov Models 11: the Viterbi algorithm, Donald Patterson *Westmont College”, posted April 7, 2020

