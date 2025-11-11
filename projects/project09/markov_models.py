##########################################################################################
#   markov_models
#   A library of objects and functions for constructing and processing 
#   Hidden Markov Models (HMM)
#   Created by Jacque Caldwell and Brooks Groharing for BINF6250 at Northeastern University
###########################################################################################

import numpy as np
import random

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
      new_state = HiddenState(state_name, init_prob = init_probs[state_name], 
                              emissions_dict = emit_probs[state_name])
      if state_name in trans_probs.keys():
        new_state.set_transitions(trans_probs[state_name])
      self.states.append(new_state)
      
  
  def run_forward(self, observations, return_matrix = False):
    """
    Calculate the probability of a sequence of observations given this
    hidden Markov Model, using the 'forward' algorithm.
    Args:
      observations: a list of observation values, or a string where each 
                    character represents 1 observation.
      return_matrix: if FALSE, return the overall probability of the sequence (default)
                     if TRUE, return (overall p, probability matrix)
    Returns: probability (float), or packed tuple (p, prob matrix)
    """
    if type(observations) == str: # convert str -> list(char)
      observations = [char for char in observations]
    
    if not type(observations) is list: # Verify type.
      raise Exception("\'observations\' must be a list or string.")
    
    # Initialize matrix, where each col is an obs, and each row is a possible state
    n_cols, n_rows = len(observations), len(self.states)
    p_matrix = np.zeros((n_rows, n_cols))
    
    # At observation 0, the probability of being in a particular state is:
    #   p = p_initial(state) * p(emitting observation 0 in this state)
    first_emission = observations[0]
    for state_i, state in enumerate(self.states):
      first_emission_prob = state.emission_probs[first_emission]
      p_matrix[state_i,0] = np.log(state.init_prob) + np.log(first_emission_prob)
      
    # For each possible path into a cell:
    #   p_total = p(path into last cell) *
    #             p(transitioning from last cell state to current cell state) *
    #             p(current state emitting current observation)
    for obs_i, observation in enumerate(observations[1:], start=1): # skip col 0
      prior_path_probs = p_matrix[:,obs_i-1] # vector at prior column
      
      for state_i, current_state in enumerate(self.states):
        trans_here_probs = [prior_state.transition_to[current_state.name] for prior_state in self.states] # vector
        p_current_emission = current_state.emission_probs[observation] # scalar
      
        # Build a vector of p_totals for each possible path into cell
        #   Since we are in log-space, we add values to AND them together
        total_path_probs = prior_path_probs.copy()
        total_path_probs += np.log(trans_here_probs)
        total_path_probs += np.log(p_current_emission)
        
        # The sum of these possible path probs is the total prob of this cell 
        p_matrix[state_i, obs_i] = sum_log_probs(total_path_probs)
    
    # Now that we have our matrix, sum the last column to get p(observations)
    overall_prob = sum_log_probs(p_matrix[:,-1])
    
    if return_matrix:
      return (overall_prob, p_matrix)
    else:
      return overall_prob
  
  
  def run_backward(self, observations, return_matrix=False):
    """
    Calculate the probability of a sequence of observations given this
    hidden Markov Model, using the 'backward' algorithm.
    Args:
      observations: a list of observation values, or a string where each 
                    character represents 1 observation.
      return_matrix: if TRUE, return internal probability matrix.
                     if FALSE, return the overall probability of the sequence (default)
    Returns: probability of the particular observation happening, or a matrix
    """
    if type(observations) == str: # convert str -> list(char)
      observations = [char for char in observations]
    
    if not type(observations) is list: # Verify type.
      raise Exception("\'observations\' must be a list or string.")

    # Initialize matrix where each col is an obs, and each row is a possible state
    n_cols, n_rows = len(observations), len(self.states)
    b_matrix = np.zeros((n_rows, n_cols+1)) # one extra col for the backward matrix
    
    # Since we start from the end, the probability of the last observation is 1.
    # So, we initialize the final column of b_matrix with 1s.
    for state_i, state in enumerate(self.states):
      b_matrix[state_i,n_cols] = np.log(1)
    
    # Now we can go column by column, filling in each cell in our matrices.
    # from the last observation moving to the left to the 
    # "inital state" in [state_i,0]
    # For each possible path into a cell (moving right to left):
    #   p_total = p(path into last cell) *
    #             p(transitioning from last cell state to current cell state) *
    #             p(current state emitting current observation)
    for obs_i in range(n_cols-1,-1,-1): # iterate backwards!
      observation = observations[obs_i]
      prior_path_probs = b_matrix[:,obs_i+1] # vector at previous column
      
      for state_i, current_state in enumerate(self.states):
        trans_here_probs = [prior_state.transition_to[current_state.name] for prior_state in self.states] # vector
        p_current_emission = current_state.emission_probs[observation] # scalar
      
        # Build a vector of probabilities for each possible path into cell
        total_path_probs = prior_path_probs.copy()
        total_path_probs += np.log(trans_here_probs)
        total_path_probs += np.log(p_current_emission)
        
        # OR the paths together to get overall prob of cell
        b_matrix[state_i, obs_i] = sum_log_probs(total_path_probs)
    
    # Now that we have our matrix, sum the last column to get p(observations)
    overall_prob = sum_log_probs(b_matrix[:,0]) 
    
    if return_matrix:
      return (overall_prob, b_matrix[:,:-1]) # drop extra column
    else:
      return overall_prob


  def run_forwardbackward(self, observations):
    '''
    part of the hidden Markov Model suite of functions
    Runs two methods of calculating probability of possible hidden 
    states then combines them to calcuate the posterior probabilities
    (this is considered an inference)
    
    Note as all of our functions return values in log space, we will
    also do the combination in log space (Ie using sums instead of multiplying)
    
    Args:
      observation: list of possible states to hand to our model
    
    returns:
      matrices with the forward, backward and posterior probs
    '''
    p_forward, f_matrix = self.run_forward(obs, return_matrix=True)
    p_backward, b_matrix = self.run_backward(obs, return_matrix=True)
    
    p_matrix = f_matrix + b_matrix - p_forward # no looping needed!
    
    return f_matrix, b_matrix, p_matrix


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
        v_matrix[state_i,0] = np.log(state.init_prob) + np.log(first_emission_prob)
      else:
        v_matrix[state_i,0] = state.init_prob * first_emission_prob
      backpointers[state_i,0] = -1 # no prior column, so set pointer to -1


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
          total_path_probs = prior_path_probs.copy()
          total_path_probs += np.log(trans_here_probs)
          total_path_probs += np.log(p_current_emission)
        else:
          total_path_probs = prior_path_probs * trans_here_probs * p_current_emission
        
        # Save the best path.
        v_matrix[state_i, obs_i] = max(total_path_probs).copy()
        backpointers[state_i, obs_i] = np.argmax(total_path_probs)
  
    if TESTING: print(f"backptrs: {backpointers}, v_matrix: {v_matrix}")
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
    
    if TESTING: print(f"traceback_pos, backptrs: {traceback_pos}, {backptrs}")
    state_names = []
    for obs_i in range(tb_obs_i, 0, -1): # loop backwards through observations
      state_names.append(self.states[state_i].name) # Save current state name
      state_i = int(backptrs[state_i, obs_i]) # update index to pointer
    
    return state_names[::-1] # return the state_names list in reverse

def sum_log_probs(list_of_logs):
  """Sum a list of floats stored in log-space without incurring underflow errors.
  
  This function solves the problems of having numbers in log space, and 
  wanting to add two together:
    a = log(10) = 2.302585
    b = log(12) = 2.484907
    
    a+b is log(10) + log(12) = log(10)*log(12) = log(120) = 4.787492
    
    we want to add log(a+b) or log(22) which is: 3.091042
    
  Args: list_of_logs  numpy list of items already in log space
  
  returns: sum of the items (again in log space)
  """
  total = list_of_logs[0]
  for prob in list_of_logs[1:]: 
    total = np.logaddexp(total, prob) # T
  
  return total

# Example data provided in project description
# Example observation sequence
obs = "ATGCAA"

# Example initial probabilities (probability of starting in each state: E := Exon, I := Intron)
init_probs = {
    "E": 0.6,
    "I": 0.4
}


# Example transition probabilities (probability of moving from one state to another)
trans_probs = {
    "E": {"E": 0.8, "I": 0.2},
    "I": {"E": 0.3, "I": 0.7}
}

# Example emission probabilities (probability of observing a symbol in a given state)
emit_probs = {
    "E": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3},
    "I": {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1}
}

if TESTING:
  test_HMM = HMM(init_probs, trans_probs, emit_probs)
  print("--------------")
  for i,state in enumerate(test_HMM.states):
    print(f"STATE {i}: \"{state.name}\"")
    print(f"Init_p: {state.init_prob}")
    print(f"emit probs:     {state.emission_probs}")
    print(f"out_probs: {state.transition_to}")
    print("\n")

  print("TEST")
  print(f"obs: {obs}")
  p_obs = test_HMM.run_forward(obs)
  print(f"forward prob: {p_obs} (log) {np.exp(p_obs)} (%)")
  print(f"for should be: -8.05724 (log) 0.0003169 (%)")

  p_obs2 = test_HMM.run_backward(obs)
  print(f"backward prob: {p_obs2} (log) {np.exp(p_obs2)} (%)")
  print(f"back should be: -7.887252 (log) 0.0003755 (%)")

  # reinitializse first
  test2_HMM = HMM(init_probs, trans_probs, emit_probs)

  f_mat, b_mat, post_mat = test2_HMM.run_forwardbackward(obs)
  # print(f"forward matrix: {f_mat}")
  # print(f"backward matrix: {b_mat}")
  print(f"posterior probs(log): {post_mat}")
  print(f"posterior probs(%): {np.exp(post_mat)}")

