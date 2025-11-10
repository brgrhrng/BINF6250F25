##########################################################################################
#   markov_models
#   A library of objects and functions for constructing and processing 
#   Hidden Markov Models (HMM)
#   Created by Jacque Caldwell and Brooks Groharing for BINF6250 at Northeastern University
###########################################################################################

import random
import numpy as np
TESTING = True

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
      
  
  def run_forward(self, observations, log_values = True):
    if type(observations) == str: # convert str -> list(char)
      observations = [char for char in observations]
    
    if not type(observations) is list: # Verify type.
      raise Exception("\'observations\' must be a list or string.")
    
    
    # Initialize probability matrix
    n_cols = len(observations) # Columns correspond to observations, in order
    n_rows = len(self.states) # Each row corresponds to a possible hidden state
    p_matrix = np.zeros((n_rows, n_cols))
    
    # Fill in the first column of our matrices
    # At observation 0, the probability of being in a particular state is:
    #   p = p_initial(state) * p(emitting observation 0 in this state)
    first_emission = observations[0]
    for state_i, state in enumerate(self.states):
      first_emission_prob = state.emission_probs[first_emission]
      if log_values:
        p_matrix[state_i,0] = np.log10(state.init_prob) + np.log10(first_emission_prob)
      else:
        p_matrix[state_i,0] = state.init_prob * first_emission_prob
      
    # For each possible path into a cell:
    #   p_total = p(path into last cell) *
    #             p(transitioning from last cell state to current cell state) *
    #             p(current state emitting current observation)
    # We save p_total for the most probable path into v_matrix,
    # and the index of the prior cell in this path (within its column) to
    # backpointers.
    for obs_i, observation in enumerate(observations[1:], start=1): # skip col 0
      prior_path_probs = p_matrix[:,obs_i-1] # vector representing last column in v_matrix
      
      for state_i, current_state in enumerate(self.states):
        trans_here_probs = [prior_state.transition_to[current_state.name] for prior_state in self.states] # vector
      
        p_current_emission = current_state.emission_probs[observation] # scalar
      
        # Build a vector of probabilities for each possible path,
        #   and save the sum of these paths
        if log_values:
          total_path_probs = prior_path_probs.copy()
          total_path_probs += np.log10(trans_here_probs)
          total_path_probs += np.log10(p_current_emission)
          
          p_matrix[state_i, obs_i] = sum_log_probs(total_path_probs)
        else:
          total_path_probs = prior_path_probs * trans_here_probs * p_current_emission
          p_matrix[state_i, obs_i] = sum(total_path_probs)
    
    # Now that we have our matrix, sum the last column to get overall probability
    if log_values:
      overall_prob = sum_log_probs(p_matrix[:,-1])
    else:
      overall_prob = sum(p_matrix[:,-1])
    
    return overall_prob
  
  
  def run_backward(self, observations, log_values=True):
    """
    Predict the most likely sequence of states that would produce a given
    set of observations in this hidden Markov Model, using the 
    'backward' algorithm.
    Args:
      observations: a list of observation values, or a string where each 
                    character represents 1 observation.
    Returns: probability of the particular observation happening
    """
    if type(observations) == str: # convert str -> list(char)
      observations = [char for char in observations]
    
    if not type(observations) is list: # Verify type.
      raise Exception("\'observations\' must be a list or string.")

    # Initialize output matrices
    n_cols = len(observations) # Columns correspond to observations
    n_rows = len(self.states) # Rows correspond to a poss hidden state
    b_matrix = np.zeros((n_rows, n_cols+1)) # one extra state for the backward matrix
    
    # Fill in the last column of our matrix
    # At observation[n_cols], the probability of being in a particular
    # state is 1, so we put this in b_matrix[state_i,n_cols]
 
    for state_i, state in enumerate(self.states):
      last_emission_prob = 1
      if log_values:
        b_matrix[state_i,n_cols] = np.log10(last_emission_prob)
      else:
        b_matrix[state_i,n_cols] = last_emission_prob
        
    # Now we can go column by column, filling in each cell in our matrices.
    # from the last observation moving to the left to the 
    # "inital state" in [state_i,0]
    # For each possible path into a cell:
    #   p_path_total = p(path into prior cell) *
    #             p(transitioning from prior cell to current cell state) *
    #             p(current state emitting current observation)
    # 
    # There will be "state" number of prob_paths, which will be added
    # together as they are "OR" probability states (so they are summed
    # together)
    #
    # We save p_total for the sum of the probable paths into b_matrix 
    
    # we will be going from n_cols to the left down to 0; note the last squares
    # have already been filed b_matrix[state_i,n_cols] above.
    
    for obs_i in range(n_cols-1,-1,-1):   # run it backwards
      observation = observations[obs_i] 
    
      prior_path_probs = b_matrix[:,obs_i+1] # vector representing last column in b_matrix
      
      for state_i, current_state in enumerate(self.states):
        trans_here_probs = [prior_state.transition_to[current_state.name] for prior_state in self.states] # vector
      
        p_current_emission = current_state.emission_probs[observation] # scalar
      
        # Build a vector of probabilities for each possible path
        if log_values:
          total_path_probs = prior_path_probs.copy()
          total_path_probs += np.log10(trans_here_probs)
          total_path_probs += np.log10(p_current_emission)
          b_matrix[state_i, obs_i] = sum_log_probs(total_path_probs)
 
        else:
          total_path_probs = prior_path_probs * trans_here_probs * p_current_emission
          b_matrix[state_i,obs_i] = sum(total_path_probs)
    
    if log_values:      
      overall_prob = sum_log_probs(b_matrix[:,0]) 
    else:
      overall_prob = sum(b_matrix[:,0])
      
    if TESTING: print(f"{overall_prob}")
    if TESTING: print(f"b_matrix: {b_matrix}")
    return(overall_prob)
  
  
  
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
          total_path_probs += np.log10(trans_here_probs)
          total_path_probs += np.log10(p_current_emission)
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
  """Sum a list of floats stored in log10-space without incurring underflow errors."""
  # base change formula: log_e(x) = log10(x) / log10(e)
  natural_logs = list_of_logs / np.log10(np.e) # convert log10-ln space
  total = natural_logs[0]
  for prob in natural_logs[1:]: 
    total = np.logaddexp(total, prob) # This func only exists for base e and 2
  total = total / np.log(10) # convert ln -> log10 space
  
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

test_HMM = HMM(init_probs, trans_probs, emit_probs)

if TESTING:
  print("--------------")
  for i,state in enumerate(test_HMM.states):
    print(f"STATE {i}: \"{state.name}\"")
    print(f"Init_p: {state.init_prob}")
    print(f"emit probs:     {state.emission_probs}")
    print(f"out_probs: {state.transition_to}")
    print("\n")

print("TEST")
p_obs = test_HMM.run_forward(obs)
print(f"forward prob: {p_obs} (log10) {10**p_obs} (%)")
print(f"for should be: -3.499 (log10) 0.0003169 (%)")
#print(sum_log_probs([-2.20202691,-2.9066793]))
#print(np.logaddexp(-2.20202691,-2.9066793))

p_obs2 = test_HMM.run_backward(obs)
print(f"backward prob: {p_obs2} (log10) {10**p_obs2} (%)")
print(f"back should be: -3.425 (log10) 0.0003755 (%)")



#prob1 = 1e-50
#prob2 = 2.5e-50
#print(sum_log_probs([np.log10(prob1),np.log10(prob2)]))
#print(np.logaddexp(np.log(prob1),np.log(prob2)))
