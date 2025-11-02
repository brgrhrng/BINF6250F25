##########################################################################################
#   markov_models
#   A library of objects and functions for constructing and processing 
#   Hidden Markov Models (HMM)
#   Created by Jacque Caldwell and Brooks Groharing for BINF6250 at Northeastern University
###########################################################################################

import random

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
  
  
  def run_viterbi(observations):
    """
    Run the viterbi algorithm to determine the most likely sequence of states
    that would produce a given set of observations under this model.
    
    args:
      observations: a list of observation values, or a string where each 
                    character is 1 observation.
    
    Returns: list of state names
    """
    if observations == str:
      observations = [char for char in observations]
    
    if not isinstance(observations, list):
      raise Exception("\'observations\' must be a list or string.")
    
    # 1. initialize the viterbi and traceback matrices
    # 2. populate them cell by cell
    # 3. traceback
    pass
  
  
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
    
    self.emissions = list(emissions_dict.keys())
    self.emission_probs = list(emissions_dict.values())
    
    self.out_states = [] # initialize as a "dead end"
    self.out_state_probs = []
    
    
  def set_transitions(self, transitions_dict):
    """
    Update outgoing edges to match provided transitions_dict.
    """
    self.out_states = list(transitions_dict.keys())
    self.out_state_probs = list(transitions_dict.values())
  
  
  def emit(self):
    """ Randomly select one emission, weighted by probabilities. Return its name."""
    emission = random.choices(self.emissions, weights=self.emission_probs, k=1)[0]
    return emission

