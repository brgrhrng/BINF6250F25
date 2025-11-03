# BINF6250F25
# Module 08 HMM: Viteri

#Setup
BINF6250; Fall 2025
Project 08.Rmd
Viteri Algorithm; a HMM with dynamic programming
Authors: Brooks Groharing and Jacqueline Caldwell
Due:  November 5, 2025

# Introduction
Our project involves creating a hidden Markov model using dynamic programming.  This hidden Markov Model is the Viterbi algorithm for finding the most likely sequence of hidden states given a sequence of observations.
The Viterbi algorithm is a dynamic programming approach to decoding hidden states, offering:
* Optimal path finding (finding the most probable sequence of hidden states)
* Dynamic programming (using tabulation to avoid redundant calculations)
* Log-space computation (to prevent numerical underflow)
* Traceback Mechanism (reconstructs the optimal state path after computation is complete)

# Information on algorithm given to us:
## basic algorithm structure:
Algorithm structure, involves three steps:
* Initialization
**Setup Prob matrix using initial probabilities and the first observation
***These were given to us
**	Initialize traceback matrix for path reconstruction.
* Recursion 
** For each position and possible state, calculate the maximum probability 
** Store both probabilities (just max, or states and max?) and traceback pointers
** Apply transition and emission probabilities at each step 
* Termination
** Identify the final state with the highest probability
** Traceback through the matrix to reconstruct the optimal path.

##Computational Considerations:

Important factors to consider in implementation:
* Time complexity : O(NxK^2) where N is sequence length and K is number of states
* Space complexity: O(NxK) for storing the dynamic programming matrix
* Numerical Stability:  Using log probabilities to prevent underflow
* Edge Cases:  Handling zero probabilities with pseudocounts 
Example Data Structures Given:
* Example observation sequence
```{}
obs = "GGCACTGAA"
```

## Example initial probabilities (probability of starting in each state)

```{}
init_probs = {
    "I": 0.2,
    "G": 0.8
}
```

## Example transition probabilities (probability of moving from one state to another)
```{}
trans_probs = {
    "I": {"I": 0.7, "G": 0.3},
    "G": {"I": 0.1, "G": 0.9}
}
```

## Example emission probabilities (probability of observing a symbol in a given state)

```{}
emit_probs = {
    "I": {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1},
    "G": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
}
```

## Other notes to consider:
* You will only be given these four data structures
* No other template code or coding-by-contract will be provided
* It may benefit you to utilize OOP as you will be developing out the whole HMM suite throughout the next 3 modules.
* Make no assumptions as to the number of hidden states you will be given.
* Make no assumptions as to the number of distinct observations you will be given
* Make no assumptions that the data structures will be modeling CpG islands (these are just examples) 

# Further Discussion/information on Viterbi

See virterbi_info.Rmd

# Style Guide

See style.Rmd

# Pseudocode
## Class
```{}
HMM.py
class HMM-
  DATA:
  hidden_states = list containing states
  possible_emissions (A T G C)

  METHODS--
    viterbi(list of observations):
      Constructs viterbi matrix
      traces back to reconstruct likely series of states
      returns (list of references/IDs of hidden states, probability)
    _build_matrix()
    _traceback()

    load_charstring()
    load_codons()
```
We are going to need functions/methods that do these things:

##Initialize viterbi table
```{}
Function “init_viterbi”(obs_states: string,
                        init_probs: [Dict] )
        returns state_prob_matrix(virtebri_matrix)

# Setup state_prob_matrix of Virtebri_table
# "I" = in CpG island
# "G" = in Genome
# obs is a string of characters whose position indicates a "state"; and whose 
#      value or character represents the emission table
   
    create the matrix that is 
        rows - number_of_hidden_states rows; 
        cols - length_of_obs_list of Virtebri_table class
        
    Set virtebri_matrix[0][0] to init_probs[“I”].values # prob of going from init to “I” state
    
    virtebri_matrix[0][1] to init_probs[“G”].values # prob of going from Init to G state
    
    returns (virtebri_matrix)
#end init_viterbi()
```

## "do/make/fill-in" the viterbi table

```{}
Function “make_viterbi”/“do_viterbi” (virtebri_matrix: array 2D array of Virtebri class,
obs string,
trans_probs [Dict],
emit_probs [Dict]
)-> virtebri_matrix (2D array of Viterbi class?)
    Start stepping through obs, one character at a time...
        1. Calculate the prob of the most likely path leading to this state
        2. Consider all possible prior states, 
            a. calcuate the prob of transitioning from each previous state to the current state, and 
            b. multipliews it by the stored probability of the best path to that prior state.
        3. a The (maximum/minimum (etc)) of these values is selected 
           b. and stored in the Viterbi table
           c. also stores a "backpointer" (ref to the prior state, the maximum probability) to reconstruct the path later.
           
           To start P(initial_state)*b_1(o_1)
           
returns virterbi_table
#end make_viterbi()
```

## Trace back the table "creating the ouptut string"; REVERSE IT! 
```
Function “Traceback viterbi”(virtebri_matrix
)-> opt_path: ?string

    read back through virtebri_matrix; translating into ouput string
    reverse it
    return it
    
#end traceback_viterbi()
```

## (optional) Visualize_Viterbi; create some sort of viz

```{}
Function “visualize_Viterbi optimal path”(opt_path ?string
) 
        for example ACTGAAATTTCCCGGG
                     ###  ####

#end visualize_Viterbi()
```

## Function that kicks everything off
```{}
Function “run_viterbi” given sequence, 
    init_probs [Dicts], trans_probs[Dicts], emit_probs[Dict]
) returns optimal_viterbi_path ?string
	
	Ok = Confirm_parameter_consistency(obs,init_probs,trans_probs,emit_probs)

	If (ok)
	     Vertibri_matrix = Init_viterbi (obs, init_probs)
         vertibri_matrix = do_vertibri(vertibri_matrix,obs,trans_probs,emit_probs)
	     Return(Traceback_viterbi(vertibri_matrix,obs))
    Else 
        print error message parameters, etc, run_viterbi function exited
        Return(“”)
 #end func run_viterbi

#Main()
opt_path = Run_viterbi(obs,init_probs,trans_probs,emit_probs) 
If not (opt_path == 0)

Vis_opt_path(opt_path)
#end Main()
```
Our initial Class structure(s)
```{}
class HiddenState:
    name = obs
    init_probs = init_prob
    emission_probs = emissions_dict
    set_transitions(self, transitions_dict)
       out_state_probs = transision_dict
    emit()
        returns random emission prob.

class HMM:
    emissions....
    self.states = [] (will be list of HiddenState objs)
    state_names = emit_probs.keys)
    __init__(init_probs,trans_probs,emit_probs)
    __traceback_viterbi__(traceback_pos,backptrs)
    __fill_viterbi_matrix__(obs)
    run_viterbi(obs)
    

```

# Successes
Jacque learned how to add a new branch to a prior repo fork on github.

# Struggles

i = ROWS = x position = hidden states
j = COLS = y position = time/emissions/obs

Description of the stumbling blocks the team experienced

# Personal Reflections
## Group Leader
Group leader's reflection on the project

## Other member
Other members' reflections on the project

# Generative AI Appendix and references
# Cite:
https://www.youtube.com/watch?v=6JVqutwtzmo; Viterbi Algorithm; Keith Chugg (USC) ; posted March 9, 2017
https://www.youtube.com/watch?v=s9dU3sFeE40; Hidden Markov Models 11: the Viterbi algorithm, Donald Patterson *Westmont College”, posted April 7, 2020
Discussions with GPT-5 to cement my (Jacque’s) understanding of Virterbi equations relationships to the data that we will be given, on-line through outlier.ai, October 30, 2025.
Discussions with Claude.ai again to cement my (Jacque’s) understanding of Virterbi equations and relationships, on-line through outlier.ai, October 30, 2025.
