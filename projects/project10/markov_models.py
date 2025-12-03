##########################################################################################
#   markov_models
#   A library of objects and functions for constructing and processing 
#   Hidden Markov Models (HMM)
#   Created by Jacque Caldwell and Brooks Groharing for BINF6250 at Northeastern University
###########################################################################################

import numpy as np
import random
import string

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
    """ Update outgoing edges to match provided transitions_dict. 
    """
    self.transition_to = transitions_dict
  
  
  def emit(self):
    """ Randomly select one emission, weighted by probabilities. Return its name.
    """
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
  
  
    
  def __bw_get_emission_probs(self):
    '''  
    Helper function to pack our emission probs into an emission matrix.
    returns: 
        matrix of the emission_probability (2D matrix- NxM) in log space
    '''
    n = len(self.states) # hidden states
    m = len(self.emissions) # number of emission states
    
    emission_probs = np.ndarray((n, m))
    
    for i, state in enumerate(self.states):
      for k, emit in enumerate(self.emissions):
        emission_probs[i][k] = np.log(state.emission_probs[emit])
    return emission_probs


  def __bw_get_init_probs(self):
    ''' 
    Helper funtion to pack our inital probs into an init matrix
    returns:
        vector of the init_probs vector of size N of the state list; values 
            are in log space
    '''
    n = len(self.states)
    
    init_probs = np.ndarray((n,1))
    
    for i, state in enumerate(self.states):
      init_probs[i] = np.log(state.init_prob)
    
    return init_probs

   
  def __bw_get_trans_to_probs(self):
    ''' Helper function to pack our transition probs into a matri
    returns:
        matrix: that size(state_list) x size(state_list)
    '''
    n = len(self.states)
    
    trans_probs = np.ndarray((n,n))
    
    for row_i,row_state in enumerate(self.states):
      for col_j,col_state in enumerate(self.states):
        trans_probs[row_i,col_j] = np.log(row_state.transition_to[col_state.name])
        
    return trans_probs

  
  def __bw_get_log_lhood(self, obs, return_matrix=True):
    ''' function to do the 'forward' and 'backward' function calls 
          and return the average log_likelihood.  If return_matrix is set 
          then the function also returns alpha_m and beta_m matrices
    Arg:
    obs: sequence of emision states representing hidden states
    return_matrix: optional parameter that says whether or not we have to return
                  the matrixes from 'forward' or 'backward', otherwise, the
                  return is only the log_likelihood
    Return:
      log_likelihood log space probability that is the average 
            of 'forward' and 'backward'
      alpha_m (also known as forward_matrix) calculated prob of observation
              happening in the order it is in observed sequence 
              (calculalated in 'run_forward') 
      beta_m (also known as backward_matrix) in state Si at time t, what 
              is prob of seeing the remaining states from t+1 to m in the
              observation sequence (calculated in 'run_backward')
    '''
    T = len(obs) # observed sequence
    N = len(self.states) # number of hidden states in model
    
   # set up the log likelihood, and alpha & beta matrices
    f_log_lhood, alpha_m = self.run_forward(obs,True)
    b_log_lhood, beta_m = self.run_backward(obs,True) 
    log_lhood_p = np.logaddexp(f_log_lhood,b_log_lhood) - 2 # avg in log space
    
    if (return_matrix):
      return(log_lhood_p,alpha_m,beta_m)
    else:
      return(log_lhood_p)


  def run_baum_welch(self, observations, max_loop_count, epsilon):
    """
      Given a list of observations; and possible initial model (is stored in 
      hmm 'self'); Baum Welch it calculates a maximized local model based 
      on the list of observations.   The model is represented by three 
      matrices init_probs, trans_to_probs and emission_probs.  These matrices
      are returned from this function.
    Args:
      observations: a list of observation sequences with observered 
                    emission states that represent a set of hidden states.
      max_loop_count (integer):  this is the maximum loop count for the
                    baum_welch algorithm
      epsilon: (float in probabilty space): the allowable difference 
                    when comparing likelihoods to see if they are equivalent
                    translated into log state in this function
      self:         Our hmm model that represents 
                    the starting states of a model.  It contains:
                        init_probs aka pi
                        trans_to   aka A_matrix or A_m
                        emissions  aka B_matrix or B_m 
                    In our model theses are stored as Dicts (or lists of Dicts)
      
    Returns:
      Maximized hmm model returned as probability space matrices: 
                init_probs,trans_to_probs,emit_probs
    """
    # variables that end in _m are matrices
    #                end in _v is a 1D matrix (init_probs)
    #                end in _p is a log state probability (single float)
    #
    # variables labled "current_" are the current baseline matrices/varables
    # variables labeld "seq_" or "new_" are the updated versions from training. 
    #
    # Note: in log_space -np.inf is considered 'zero'
    #
    # Basic plan is:
    #
    # Initialize
    # while loop to iterate many, many times.
    #   for loop to train over sequences
    #     using Estep & Mstep calculate new model for seq in seq_
    #     sum_seq_lhood, and seq_model matrices
    #   scale back sum_seq and seq_model matrices (in new_)
    #   check for convergence
    #     break loop with current model if converged
    #   if not converged - reset for next while loop iteration
    # return(current model)
    #
    
    N, M = len(self.states), len(self.emissions)
    
    # Initialize current model parameters
    current_init_v = self.__bw_get_init_probs() # current_model
    current_trans_to_m = self.__bw_get_trans_to_probs()
    current_emissions_m = self.__bw_get_emission_probs()
    current_log_lhood = self.__bw_get_log_lhood(observations[0],return_matrix=False)
    
    # Initialize updated model parameters
    new_seq_init_v = np.ndarray((N,1))
    new_seq_trans_to_m = np.ndarray((N,N))
    new_seq_emissions_m = np.ndarray((N,M))
    
    
    # Iterate over training steps until we converge, or max loop count reached
    log_epsilon_p = np.log(epsilon) # convert prob to log_prob
    converged = False
    loop_count = 0
    
    while not converged:
      if loop_count == max_loop_count:
        break
      
      update_count = 0 # initialize count of how many times we update our seq_model
      
      new_log_lhood = -np.inf
      new_init_v = np.full((1,N), -np.inf)
      new_trans_to_m = np.full((N,N), -np.inf)
      new_emissions_m = np.full((N,M), -np.inf)
      
      # Train on the sequences
      for i, obs in enumerate(observations): # for each of our sequences; train...
        # Calculate seq model for this sequence
        seq_log_lhood, gamma_m, xi_m = self.bw_EStep(obs, current_init_v, current_trans_to_m, current_emissions_m)
        
        seq_init_v, seq_trans_to_m, seq_emissions_m = self.bw_MStep(obs, gamma_m, xi_m)
        
        # Summarize liklihood as well as sequence models
        if np.isinf(seq_log_lhood) == False: # have we updated?
          update_count += 1 # count the number of times this was updated
          new_log_lhood = np.logaddexp(new_log_lhood,seq_log_lhood)
          new_init_v = np.logaddexp(new_init_v,seq_init_v)
          new_trans_to_m = np.logaddexp(new_trans_to_m, seq_trans_to_m)
          new_emissions_m = np.logaddexp(new_emissions_m, seq_emissions_m)
      
      if update_count:
        # Scale back by the number of times we increased our 'new model'
        new_log_lhood -= np.log(update_count)
        new_init_v -= np.log(update_count)
        new_trans_to_m -= np.log(update_count)
        new_emissions_m -= np.log(update_count)
        
        # Check for convergence -- will run through each of the observations before exiting
        if new_log_lhood == current_log_lhood:
          print(f"Converged {loop_count} iterations to local maximum, log_lhood (equal): {np.exp(new_log_lhood)}")
          converged = True
        elif abs(np.exp(new_log_lhood) - np.exp(current_log_lhood)) <= log_epsilon_p:
          print(f"Converged {loop_count} iterations to local maximum, log_lhood (within epsilon): {np.exp(new_log_lhood)}")
          converged = True
        else: # reset for next 'while' loop run
          current_init_v = new_init_v
          current_trans_to_m = new_trans_to_m
          current_emissions_m = new_emissions_m
      
      loop_count += 1
    
    # Update state objects with new init/trans/emit probs
    state_names = [state.name for state in self.states]
    for state_i, state in enumerate(self.states):
      trans_probs = current_trans_to_m[state_i]
      emission_probs = current_emissions_m[state_i]
      
      state.init_prob = current_init_v[0][state_i]
      state.transition_to = dict(zip(state_names,trans_probs))
      state.emission_probs = dict(zip(self.emissions, emission_probs))
    
    return np.exp(current_init_v), np.exp(current_trans_to_m), np.exp(current_emissions_m)
  
	
  def bw_EStep(self, obs, pi_m, A_m, B_m): # aka 'Expectation setp'
    ''' Expectation step of Baum-Welch -- needs a single observation,
         Also takes the model that we need to calculate the 'expectation step' on.
         
         obs: sequence of T observed emissions.
         pi_m : 1xN matrix in log_space ; this init probs for our model
         A_m : NxN matrix in log space ; transition matrix for our model
         B_m: NxM matrix in log space; emission matrix for our model
    Returns
      log_lhood (from average of forward&backward run)
      gamma_m: NxT matrix in log_space of the probability of a 
              particular state (Si) to another state (Sj) in our obs sequence
      xi_m: NxT matrix in log_space of the probability that I was in state i 
              at time t AND then transitioned to state j at time t+1, given 
              all I have observed. 
              Xi tells you which state transitions happened in our obs seq.
    '''
    
    T = len(obs) # length of the obs sequence
    N = len(self.states) # number of hidden states
    
    avg_log_lhood, alpha_m, beta_m = self.__bw_get_log_lhood(obs,return_matrix=True) #runs forward & backward!
    
    # gamma_m is (alpha_m + beta_m) - avg_log_lhood (scalar)
    # math is in log space!
    gamma_m = np.ndarray((N,T))
    
    gamma_m = alpha_m + beta_m - avg_log_lhood
    
    # xi_m is the prob of being in state Si at time t and state Sj at time t+1 
    # We need three dimensional array
    # xi[t-1,i,j] = P(q_t = i, q_t+1 = j given obs & 'model')
    # xi_num (alpha_m(i,t) + beta_m(j,t+1)) + A_m[i][j] + B_m(O_t+1,j)
    xi_m = np.ndarray((T-1, N, N))
    
    # Compute denom,numerator and then xi_m for all state pairs
    for t in range(T-1):
     
      denominator = 0# Denom: P(O | model) = sum over all states @time t
      for i in range(N):
        for j in range(N):
          emit=obs[t+1] # snag the emission state character
          B_j_obs_t_plus1 = np.log(self.states[j].emission_probs[emit])
          if denominator:
            denominator +=alpha_m[i,t]+A_m[i, j] + B_j_obs_t_plus1 + beta_m[j,t+1]
          else:
            denominator = alpha_m[i,t]+A_m[i,j] + B_j_obs_t_plus1 + beta_m[j,t+1]
          
      for i in range(N):
        for j in range(N):
          emit=obs[t+1]
          B2_j_obs_t_plus1 = self.states[j].emission_probs[emit]
          numerator = alpha_m[i,t] + A_m[i,j] + B2_j_obs_t_plus1 + beta_m[j,t+1]
          
          xi_m[t, i, j] = numerator - denominator # more log math!
                
    return(avg_log_lhood,gamma_m,xi_m)


  def bw_MStep(self, obs, gamma_m, xi_m):
    ''' Creates new model (init, transition and emission matrices)
        given an obs seq and calculated probabilities in gamma_m and xi_m
        Normalization will also be done ; all in log space
    Args:
      gamma_m # P of going from Si to Sj 
      xi_m    # P of being in Si at time t and Sj at time t+1
    Returns
      init_hat # model matrix init_probs Step 1
      A_hat_m  # model matrix trans_to_probs Step 2
      B_hat_m  # model matrix emission_probs Step 3
    '''  
    # We need init_probs, A_m and B_m in order to create the next model
    T = len(obs)
    N = len(self.states)
    M = len(self.emissions)
    
    # 1 Re-estimate pi (init_probs)
    init_hat = np.ndarray((N,T))
    init_hat = gamma_m[:,0] - np.logaddexp.reduce(gamma_m[:,0]) # gamma_m[0] is init states (but normalize it!)
    
    # 2. Re-estimate A (transition matrix)
    # A[i,j] = sum_t(xi[t,i,j]) / sum_t(gamma[t,i])
    A_hat = np.ndarray((N,N))
 
    for i in range(N): # for each state - calculate the transitions to other states
        A_denom = np.logaddexp.reduce(gamma_m[:-1,i])#sum over t=0 to T-2
        for j in range(N): # calculate for each 'other' state
            A_num = np.logaddexp.reduce(xi_m[:, i, j])#sum over all time (i,j)
            A_hat[i,j] = A_num - A_denom
        A_hat[i,:] -= np.logaddexp.reduce(A_hat[i,:]) # confirm it's normalized
        
    # 3. Re-estimate B (emission matrix)
    # B[j,k] = sum_t(gamma[t,j] * I(o_t = k)) / sum_t(gamma[t,j])
    B_hat = np.ndarray((N,M))
    
    for j in range(N):
        B_denom = np.logaddexp.reduce(gamma_m[:, j])  # sum over all time steps
        for k, emit in enumerate(self.emissions):
            # Sum gamma[t,j] for all t where observation is 'emit'
            B_num = self.bm_logsum_emission_probs(obs,gamma_m,j,emit)
            B_hat[j, k] = B_num - B_denom
        B_hat[j,:] -= np.logaddexp.reduce(B_hat[j,:]) # confirm it's normalized
        
    return init_hat, A_hat, B_hat
    #end M-Step
    
    
  def bm_logsum_emission_probs(self,obs,gamma,index,emit):
    '''Creates sum of all of the gamma probabilities of type emit
    Args:
        obs:  list of char; current sequence
        gamma: prob of going from Si to Sj given obs and model
        index: index into gamma that the sum is needed for.
        emit: character that we want to sum up the probabilities for
    Returns 
      in log space the summ of all of prob of the 'emit' in current obs seq
    '''
    T = len(obs)
    running_sum = 0
    
    for t in range(T-1):
      if (obs[t] == emit):
        running_sum = np.logaddexp(running_sum,gamma[index,t])
    return(running_sum)
      
      
  def run_forward(self, observations, return_matrix = False):
    """
    Calculate the probability of a sequence of observations given this
    hidden Markov Model, using the 'forward' algorithm.
    Args:
      observations: a list of observation values, or a string where each 
                    character represents 1 observation.
      return_matrix: if FALSE, return the overall probability of the sequence
                      (default) if TRUE, return (overall p, probability matrix)
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
    b_matrix = np.zeros((n_rows, n_cols)) # initialize B matrix with zeros
    
    # Since we start from the end, the probability of the last observation is 1.
    # So, we initialize the final column (t-1) of b_matrix with 1s.
    # for state_i, state in enumerate(self.states):
    #  b_matrix[state_i,n_cols] = np.log(1) ; but as np.log(1) is 0; no need!
    
    # Now we can go column by column, filling in each cell in our matrices (we filled it from t-2 down to)
    # from the last observation moving to the left to the 
    # "inital state" in [state_i,0]
    # For each possible path into a cell (moving right to left):
    #   p_total = p(path into last cell) *
    #             p(transitioning from last cell state to current cell state) *
    #             p(current state emitting current observation)
    # changed to log states -- so log(p_total) = prior + log(p(transit_to)) + log(p(emission(state)))
    for obs_i in range(n_cols-2,-1,-1): # iterate backwards!
      observation = observations[obs_i+1] #  we will be looking at the +1 observation for emission and transition_to
      prior_path_probs = b_matrix[:,obs_i+1] # vector at previous column
      
      for state_i, current_state in enumerate(self.states):
        trans_here_probs = [current_state.transition_to[prior_state.name] for prior_state in self.states] # vector
        p_current_emission = current_state.emission_probs[observation] # scalar -- obs is already t+1!!!!!
        
        # Build a vector of probabilities for each possible path into cell
        total_path_probs = prior_path_probs.copy()
        total_path_probs += np.log(trans_here_probs)
        total_path_probs += np.log(p_current_emission)
        
        # OR the paths together to get overall prob of cell
        b_matrix[state_i, obs_i] = np.logaddexp.reduce(total_path_probs)
    
    # Now that we have our matrix, sum the last column to get p(observations)
    overall_prob = np.logaddexp.reduce(b_matrix[:,0]) 
    
    if return_matrix:
      return (overall_prob, b_matrix)
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
      observations: list of possible states to hand to our model
    
    returns:
      matrices with the forward, backward and posterior probs
    '''
    p_forward, f_matrix = self.run_forward(observations, return_matrix=True)
    p_backward, b_matrix = self.run_backward(observations, return_matrix=True)
    
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
    for obs_i in range(tb_obs_i, -1, -1): # loop backwards through observations to 0!
      state_names.append(self.states[state_i].name) # Save current state name
      state_i = int(backptrs[state_i, obs_i]) # update index to pointer
    
    return state_names[::-1] # return the state_names list in reverse


def sum_log_probs(list_of_logs):
  """Sum a list of floats stored in log-space without incurring underflow errors.
  
  This function solves the problems of having tiny numbers stored in log space, 
  and wanting to add them together:
    ln_a = ln(10e-100) = -227.955924206
    ln_b = ln(12e-100) = -227.77360265
    
  To calculate ln(a+b) from the stored ln(a) and ln(b), mathematically we would 
  want to first convert them back out of log-space:
    a+b = e**ln(10) + e**ln(12)
        = 10e-100 + 12e-100
        = 22e-100
    ln(a+b) = ln(22e-100) = -227.167466846
  
  However, this inner conversion risks causing underflow errors.
  Instead, we use numpy's logaddexp to stably add the log'd values together.
    
  Args: list_of_logs  numpy list of items already in log space
  Returns: sum of the items (again in log space)
  """
  total = list_of_logs[0]
  for prob in list_of_logs[1:]: 
    total = np.logaddexp(total, prob) # T
  
  return total


def create_alphabet(observations):
  ''' Creates a basic alphabet given a list of observations
      will create a unique list of emission names in the returned list
  Args:
      observations:  list of sequences that contain lists of characters 
                      that represent emission states for our hmm model; 
                      our current assumption is that all of the states are 
                      represented by single character emission state.
  Return:
      alphabet: a unique set of emission states based on the observations
  '''
  list_max = len(observations)
  total_list = ""
    
  for i, obs in enumerate(observations):
    if isinstance(obs,str):
      total_list += obs
    else:
      print("currently we cannot create alphabets for observations that are not strings")
      return("")
  return(sorted(set(total_list)))


def create_simple_equal_probs(num_states):
  ''' Creates a simple probabilities given a number statess (doesn't matter if it is hidden or emission states)
  
    As we haven't coverted everything to log probs yet, this is just a simple 1/num_states that will create a 
    probability for each 'state'.
    Ex: if we are given 2 states then we will return a numpy list with ( 0.5, 0.5 )
    Args:
      num_states integer, the number of states to calculate simple initial probabilities for 
  '''
  # set default_model init to have 2D vector 1*N 
  
  state_probs = list()
  if (num_states):
    equal_state_probs = 1/num_states
    for states in range(num_states):
      state_probs.append(equal_state_probs)
    return(state_probs)
  else:
    print("create_simple_equal_probs(): sent invalid number of states {num_states}")
    return("")
  
  
def create_hidden_states(emission_states,num_trans_states):
  ''' 
    Creates a string list of the hidden state "names"; we know that the number of satates is num_trans_states; 
    and we are given the names of the emission_states; so we can pick a couple of characters to represent the
    hidden states
    
    emission_states: string of characters known emission states (the alphabet if you will)
                      These will be EXCLUDED from the list we pick from
    num_trans_states: the number of hidden states that we need to pick
    
    returns:
    hidden_states: string of characters representing the hidden states in this model.
  '''
  all_possible  = list(string.ascii_uppercase)
  new_poss_states = list()
  for char in all_possible:
    if char in emission_states:
      continue # we don't want to duplicate the names if we can avoid it
    else:
      new_poss_states += char
  
  return new_poss_states[0:num_trans_states]


def create_simple_default_model(observations,num_trans_states):
  """
  Note:
  While most of our functions are done all in log space, this one will be done in probability space,
  as the base functions for the user of this suite, would be inputing the data in prob space not log space.
  
  Given a list of observatations, creates 'model' of symbolic and state transitions as needed for initializing
  our HMM class.
  
  Args: 
    observations: list of strings that is a list of sequences representing the observed states that are 
                  seen in our sequences
    num_trans_states:  integer - the number of hidden states that are preseent in our model system
  
  returns:
    init: list of dict that is the initialization transition frequencies.
    trans_to: list of dict of dicts of our hidden states transition probabilities(this is an num_trans_states x num_trans_states dictionary)
    emissions: list of dict of dict of our state emission probabilities.(this is a num_trans_states x num_emission_states dictionary)
  """
  
  init = {}
  trans_to = {}
  emissions = {}
  sub_trans_to = {}
  sub_emit = {}
  
  obs_state_names = create_alphabet(observations)
  num_emission_states = len(obs_state_names)
  emission_probs = create_simple_equal_probs(num_emission_states)
  
  hidden_state_names = list(create_hidden_states(obs_state_names,num_trans_states))
  hidden_state_probs = create_simple_equal_probs(num_trans_states)
  
  init_state_names = hidden_state_names
  
  for key, value in zip(hidden_state_names,hidden_state_probs):
    init[key] = value
  
  for top_key, top_hidden_state in enumerate(hidden_state_names):
    for sub_dict_key, sub_hidden_probs in zip(hidden_state_names,hidden_state_probs):
      sub_trans_to[sub_dict_key] = sub_hidden_probs
    for sub_emit_key, sub_emit_probs in zip(obs_state_names,emission_probs):
      sub_emit[sub_emit_key] = sub_emit_probs
    trans_to[top_hidden_state] = sub_trans_to
    emissions[top_hidden_state]= sub_emit
 
  return(init, trans_to, emissions)


if TESTING:
  # Example data provided in project description
  # Example observation sequences (multiple sequences for training)
  observations = ["GGCACTGAA", "ATGCAATGC", "AATGCCTGA"]

# Example initial probabilities (probability of starting in each state)
  init_probs = {
    "H": 0.5,  # H = High GC content state
    "L": 0.5   # L = Low GC content state
  }

# Example transition probabilities (probability of moving from one state to another)
  trans_probs = {
    "H": {"H": 0.6, "L": 0.4},
    "L": {"H": 0.3, "L": 0.7} 
  }

# Example emission probabilities (probability of observing a symbol in a given state)
  emit_probs = {
    "H": {"A": 0.2, "C": 0.3, "G": 0.3, "T": 0.2},
    "L": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
  }
  
  test_HMM = HMM(init_probs, trans_probs, emit_probs)
  
  print("--------------")
  for i,state in enumerate(test_HMM.states):
    print(f"STATE {i}: \"{state.name}\"")
    print(f"Init_p: {state.init_prob}")
    print(f"emit probs:     {state.emission_probs}")
    print(f"out_probs: {state.transition_to}")
    print("\n")

  print("TEST")
  
  our_loop_max = 1000
  our_epsilon = 0.0000001
  new_init, new_trans, new_emit = test_HMM.run_baum_welch(observations,our_loop_max,our_epsilon)
  
  print(f"Baum Welch Recalculated")
  print(f"Init: {new_init}")
  print(f"Transitions: {new_trans}")
  print(f"Emissions: {new_emit}")
  
  print("\nUPDATED STATES:")
  for i,state in enumerate(test_HMM.states):
    print(f"STATE {i}: \"{state.name}\"")
    print(f"Init_p: {state.init_prob}")
    print(f"emit probs:     {state.emission_probs}")
    print(f"out_probs: {state.transition_to}")
    print("\n")
  print("--------------")
  '''
  '# And with our own simple starting point of transition states and emission states"
  number_of_hidden_states = 2
  
  init_probs2, trans_probs2, emit_probs2 = create_simple_default_model(observations,number_of_hidden_states)
  test2_HMM = HMM(init_probs2, trans_probs2, emit_probs2)
  
  for i,state in enumerate(test2_HMM.states):
    print(f"STATE {i}: \"{state.name}\"")
    print(f"Init_p: {state.init_prob}")
    print(f"emit probs:     {state.emission_probs}")
    print(f"out_probs: {state.transition_to}")
    print("\n")
  
  our_loop_max = 10
  our_epsilon = 0.00001
  new_init, new_trans, new_emit = test2_HMM.run_baum_welch(observations,our_loop_max,our_epsilon)
  
  print(f"Baum Welch Recalculated")
  print(f"Init: {new_init}")
  print(f"Transitions: {new_trans}")
  print(f"Emissions: {new_emit}")

  '''
  
