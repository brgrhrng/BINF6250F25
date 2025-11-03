##########################################################################################
#   markov_models
#   A library of objects and functions for constructing and processing 
#   Hidden Markov Models (HMM)
#   Created by Jacque Caldwell and Brooks Groharing for BINF6250 at Northeastern University
###########################################################################################

import random
import numpy as np

class HiddenState:
  """
  An object representing one state in a hidden markov model.
  Internal values:
    name: a unique name representing the state
    init_prob: probability of state coming from the start node
    
    out_states: list of HiddenState names representing outbound transitions
    out_probs: list of probabilities of transitioning to each out_state
    
    emissions, emission_probs: list of possible emissions and their probabilities
    """
  def __init__(self, name: str, init_prob: float, emissions_dict: dict[str,float]):
    """
    Initialize an object representing a state.
    Args:
      name: unique string identifying the state
      init_prob: probability of state coming from the start node
      emissions_dict: dict of {emission_name, prob} pairs
        - the keys should consistent across HiddenStates in the parent HMM
        - example: {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1}
    """
    self.name = name
    self.init_prob = init_prob
    
    self.emission_probs = emissions_dict
    
    self.out_state_probs = {} # initial empty as a deadend
    
    
  def set_transitions(self, transitions_dict):
    """
    Update outgoing edges to match provided transitions_dict.
    """
    self.out_state_probs = transitions_dict
  
  
  def emit(self):
    """ Randomly select one emission, weighted by probabilities. Return its name."""
    emission = random.choices(self.emissions, weights=self.emission_probs, k=1)[0]
    return emission


class HMM:
  """
  An object representing a hidden markov model.
  Internal values:
    hidden_states: list of HiddenState objects.
    emissions: list of values that can be emitted by each hidden_state
  """
  def __init__(self, init_probs, trans_probs, emit_probs):
    """
    Initialize the HMM object. Currently this creates a state
    if and only if it has an emissions dict in emit_probs.
    Args:
      init_probs: dict of (state_name, probability) pairs
      trans_prob: nested dict where the key is a state name, and the value is
                  a dict of (state_name, probability) pairs
     emit_probs: nested dict where the key is a state name, and  value is
                  a dict of (emission_name, probability) pairs
    """
    self.emissions = list(list(emit_probs.items())[0][1].keys()) # emission keys from 1st state
    self.states = [] # container for HiddenState objects
    
    state_names = emit_probs.keys()
    
    for state_name in state_names:
      new_state = HiddenState(state_name, init_prob = init_probs[state_name], emissions_dict = emit_probs[state_name])
      
      if state_name in trans_probs.keys():
        new_state.set_transitions(trans_probs[state_name])
        
      self.states.append(new_state)
      
  
  def __traceback_viterbi__(self,traceback_pos,backptrs):
    ''' private function that traces back the backpointers 
        creating the hidden states list of our viterbi matrix.
    
    args: 
      traceback_pos: pos to start traceback of backptrs.
      backpointers: backpointers to create our final traceback of hidden states
      
    returns:  list of strings that show the hidden states
    '''
    i, j =traceback_pos
    state_names = []
    
    print(traceback_pos,backptrs)
    
    while i >= 0:
      state_names.append(self.states[j].name)
      new_pos = backptrs[j][i] #numpy refers to rows,col
      i = new_pos[0]
      j = new_pos[1]
  
    return(reversed(state_names))
    
    
  def run_viterbi(self, observations):
    """
    Run the viterbi algorithm to determine the most likely sequence of states
    that would produce a given set of observations under this model.
    
    args:
      observations: a list of observation values, or a string where each 
                    character is 1 observation.
    
    Returns: list of state names
    """
    
    if type(observations) == str:
      observations = [char for char in observations]
    
    #if not isinstance(observations, list):
    if not type(observations) is list:
      raise Exception("\'observations\' must be a list or string.")
      
    v_matrix, backptrs = self.__fill_viterbi_matrix__(observations)
    
    ypos = np.argmax(v_matrix[:,-1])
    
    xpos = v_matrix.shape[1]-1
    
    print(v_matrix.shape, xpos, ypos)
    
    traceback = self.__traceback_viterbi__((xpos,ypos), backptrs)
    
    # traceback will be ycoord to index the list of strings "IIIGGGGGGIII"

    print (v_matrix,backptrs)
    print (traceback)

  def __fill_viterbi_matrix__(self, observations):
    '''Creates a Viterbi matrix and backpointers
        1. Initialize the viterbi and traceback matrices
        2. populate the matrices one by one.
    
    args: 
      observations: list of strings (could be objects) that represent our states
      
      v_matrix:  size of hidden states and positions of observation states 
      
    Returns:
      v_matrix: matrix with floating point number representing probabilities.
      backpointers: to the preceeding grid cell at each possition
      
    '''
   
    #create the matrix that is 
    #    rows - number_of_hidden_states rows; 
    #    cols - length_of_obs_list of viterbi_matrix (v_matrix)
    num_rows = len(self.states)
    num_cols = len(observations)
    
    # initiallize a numpy array; then set the initial probabilities to it
    v_matrix = np.zeros((num_rows,num_cols))
    backpointers = [] # 2D matrix of tuples; same size as v_matrix
  
    first_emission = observations[0]
    for i,state in enumerate(self.states):
      first_emission_prob = state.emission_probs[first_emission]
      v_matrix[i,0] = state.init_prob*first_emission_prob
    # Now that we are initialized, we need to fill the remainder of the matrix
    print(f"v_matrix {v_matrix}")
    
    for obs_i in range(1,num_cols):   # go through obs string one at a time
                                  # index 0 was set above.
      obs_state = observations[obs_i]

      for state_i,hidden_state in enumerate(self.states):  # states found in 
        emission = obs_state 
        emission_prob = hidden_state.emission_probs[emission]
        val_prob=0     # temp value
        max_local_prob = 0 # reset local max

        for prior_i, prior_state in enumerate(self.states):
        #  if (v_matrix[state_i,prior_i] == 0):
        #    prior_state_prob = 1
        #  else:
          prior_state_prob = v_matrix[state_i,prior_i]    
            
          transit_prob = prior_state.out_state_probs[hidden_state.name] 
    
          val_prob = prior_state_prob*transit_prob*emission_prob # prior_prob*A(i,j)*B(j)
          if (max_local_prob) == 0 or (max_local_prob > val_prob):
            max_local_prob = val_prob
            max_local_index = prior_i # this is for possible use as back pointer
    # when we end the loop max_current_local and max_current_state/.
        v_matrix[state_i,obs_i] = max_local_prob
      
    
    print(f"v_matrix {v_matrix} backpointers {backpointers}")
    
    return(v_matrix,np.array(backpointers))
  

# Example data provided in project description
obs = "GGCACTGAA"
#obs = "ATGCGCGGCTTAGCGCGGATCGCGCTTAGCGCG"

init_probs = {
    "I": 0.2, "G": 0.8}
trans_probs = {
    "I": {"I": 0.7, "G": 0.3}, "G": {"I": 0.1, "G": 0.9}}
emit_probs = {
    "I": {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1},
    "G": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}}

test_HMM = HMM(init_probs, trans_probs, emit_probs)

print("--------------")
for i,state in enumerate(test_HMM.states):
  print(f"STATE {i}: \"{state.name}\"")
  print(f"Init_p: {state.init_prob}")
  print(f"emit probs:     {state.emission_probs}")
  print(f"out_probs: {state.out_state_probs}")
  print("\n")

test_HMM.run_viterbi(obs)

