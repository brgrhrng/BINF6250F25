##########################################################################################
#   markov_models
#   A library of objects and functions for constructing and processing 
#   Hidden Markov Models (HMM)
#   Created by Jacque Caldwell and Brooks Groharing for BINF6250 at Northeastern University
###########################################################################################

import random
import numpy as np
TESTING = False

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
    self.transition_to = {} # initialize state as a "dead end"
    
    
  def set_transitions(self, transitions_dict):
    """ Update outgoing edges to match provided transitions_dict. """
    self.transition_to = transitions_dict
  
  
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
      
      
  def run_viterbi(self, observations):
    """
    Predict the most likely sequence of states that would produce a given
    set of observations in this model, using the viterbi algorithm.
    Args:
      observations: a list of observation values, or a string where each 
                    character represents 1 observation.
    Returns: list of state names
    """
    if type(observations) == str: # convert str -> list(char)
      observations = [char for char in observations]
    
    if not type(observations) is list: # Verify type.
      raise Exception("\'observations\' must be a list or string.")
    
    v_matrix, backptrs = self.__fill_viterbi_matrix__(observations)
    
    obs_i = v_matrix.shape[1]-1
    state_i = np.argmax(v_matrix[:,-1])
    traceback = self.__traceback_viterbi__((obs_i,state_i), backptrs)
    return traceback

    
  def __fill_viterbi_matrix__(self, observations, log_values = True):
    '''Creates a Viterbi matrix and backpointers
        1. Initialize the viterbi and traceback matrices
        2. populate the matrices one by one.
    Args: 
      observations: a 1d list where each item represents one observation
      log_values (optional):  set to False to return flat p-values.
                              This will result in underflow errors!
    Returns:
      v_matrix: matrix with floating point number representing log-probabilities
      backpointers: to the preceeding grid cell at each possition
      
    '''
    # Initialize output matrices
    n_cols = len(observations) # Columns correspond to observations, in order
    n_rows = len(self.states) # Each row corresponds to a possible hidden state
    v_matrix = np.zeros((n_rows, n_cols))
    backpointers = np.empty((n_rows, n_cols))
    
    # Fill in the first column of our matrices
    # At observation 0, the probability of being in a particular state is:
    #   p = p_initial(state) * p(emitting observation 0 in this state)
    first_emission = observations[0]
    for state_i, state in enumerate(self.states):
      first_emission_prob = state.emission_probs[first_emission]
      if log_values:
        v_matrix[state_i,0] = np.log10(state.init_prob) + np.log10(first_emission_prob)
      else:
        v_matrix[state_i,0] = state.init_prob * first_emission_prob
      backpointers[state_i,0] = -1 # no prior column, so set pointer to -1
    
    if TESTING: print(f"v_matrix {v_matrix}")
    if TESTING: print(observations)
    
    # Now we can go column by column, filling in each cell in our matrices.
    # For each possible path into a cell:
    #   p_total = p(path into last cell) *
    #             p(transitioning from last cell state to current cell state) *
    #             p(current state emitting current observation)
    # We save p_total for the most probable path into v_matrix,
    # and the index of the prior cell in this path (within its column) to
    # backpointers.
    for obs_i, observation in enumerate(observations[1:], start=1): # skip col 0
      prior_path_probs = v_matrix[:,obs_i-1] # vector representing last column in v_matrix
      
      for state_i, current_state in enumerate(self.states):
        trans_here_probs = [prior_state.transition_to[current_state.name] for prior_state in self.states] # vector
        p_current_emission = current_state.emission_probs[observation] # scalar
        
        # Build a vector of probabilities for each possible path
        if log_values:
          total_path_probs = prior_path_probs
          total_path_probs += np.log10(trans_here_probs)
          total_path_probs += np.log10(p_current_emission)
        else:
          total_path_probs = prior_path_probs * trans_here_probs * p_current_emission
          
        # Save the best path.
        v_matrix[state_i, obs_i] = max(total_path_probs)
        backpointers[state_i, obs_i] = np.argmax(total_path_probs)
    
    if TESTING: print(f"v_matrix {v_matrix} backpointers {backpointers}")    
    return(v_matrix, backpointers)
  
  
  def __traceback_viterbi__(self, traceback_pos, backptrs):
    ''' private function that traces back the backpointers, 
        obtaining the most probable sequence of states up to traceback_pos
    Args: 
      traceback_pos: pos to start traceback of backptrs.
      backpointers: backpointers to create our final traceback of hidden states
    Returns:  list of strings that show the hidden states
    '''
    tb_obs_i, state_i = traceback_pos
    tb_obs_i, state_i = int(tb_obs_i), int(state_i)
    
    state_names = []
    for obs_i in range(tb_obs_i, 0, -1): # loop backwards through observations
      state_names.append(self.states[state_i].name) # Save current state name
      state_i = int(backptrs[state_i, obs_i]) # update index to pointer
    
    state_names = reversed(state_names)
    return state_names
  

TESTING = False

if TESTING:    # this entire block is commented out when TESTING= FALSE
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
    print(f"out_probs: {state.transition_to}")
    print("\n")

  test_HMM.run_viterbi(obs)
# End TESTING if statement.
