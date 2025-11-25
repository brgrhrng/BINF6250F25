##########################################################################################
#   markov_models
#   A library of objects and functions for constructing and processing 
#   Hidden Markov Models (HMM)
#   Created by Jacque Caldwell and Brooks Groharing for BINF6250 at Northeastern University
###########################################################################################

import numpy as np
import random

# Global Constants

EQ = "EQ" # equal
GT = "GT" # greater than
LT = "LT" # less than

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
      
  
  def bw_setup_baum_welch(self, observations):
    """
    Place holder -- currently we will be sending in our initial setup in through
    the same method we used for all of our other HMM functions (with init, trans_to and emissions)
    """
    print("Function setup_baum_welch is not setup yet")
  
  def bm_initialize_new_hmm(self):
    ''' Given a self hmm model; initializes a new model with the same settings
    Args:
        self
    Returns
        new_model ; initialized to all of the same variables as the current model as a 
                  starting the function 
    '''
    print("bm_initialization_new_hmm: not implemented yet, place holder")
    print(" will need to create a new HMM and set all it's values to the ones in self")
    new_model = self
    return new_model
    
  def bw_create_alphabet(observations):
    ''' Creates a basic alphabet given a list of observations -- will create a unique list
        of emission names in the returned list
    Args:
        observations:  list of strings/characters that represent emission states for our 
                        new hmm class; our current assumption is that all of the states
                        are represented by single character emission state.
    Return:
        alphabet: a unique set of emission states based on the observations
    '''
    list_max = len(observations)
    total_list = ""
    
    for i, obs in enumerate(observations):
      if is_instance(obs,str):
        total_list += obs
      else:
        print("currently we cannot create alphabets for observations that are not strings")
        return("")
    return(sort(set(total_list)))

  def bw_get_emission_probs(self):
    '''  Helper function that takes an hmm model
    Args:
        self: the model itself
        state_list:  list of current hidden states to help set up the matrix to return
    returns: 
        matrix of the emission_probability (2D matrix rows - N (hidden states) x col- M (emissions))
    '''
    n = len(self.states)
    m = len(self.emissions)
    
    emission_probs = np.zeros((n, m))
    
    for i, state in enumerate(self.states):
      for k, emit in enumerate(self.emissions):
        emission_probs[i][k] = np.log(state.emission_probs[emit])
    return emission_probs

  def bw_get_init_probs(self):
    ''' helper funtion that takes an hmm and returns a vector with the initialization probs in it.
    returns:
        vector that is the size of the state list
    '''
    n = len(self.states)
    
    init_probs = np.zeros((n,1))
    
    for i, state in enumerate(self.states):
      init_probs[i] = np.log(state.init_prob)
    
    return init_probs
   
  def bw_get_trans_to_probs(self):
    ''' helper funtion that model, and returns a vector with the transition state probabilities
    Args:
    returns:
        matrix: that size(state_list) x size(state_list)
    '''
    n = len(self.states)
    
    trans_probs = np.zeros((n,n))
    
    for row_i,row_state in enumerate(self.states):
      for col_j,col_state in enumerate(self.states):
        trans_probs[row_i,col_j] = np.log(row_state.transition_to[col_state.name])
        
    return trans_probs
  
  def bw_get_log_lhood(self, obs, return_matrix=False):
    ''' function to do the function calls and return the log_likelihood
  
    Arg:
    obs: sequence of emision states representing hidden states
    return_matrix: optional parameter that says whether or not we have to return 
                  the matrixes from 'forward' or 'backward', otherwise, the return is
                  only the log_likelihood
    Return:
      log_likelihood log space probability that is the average of 'forward' and 'backward'
      alpha_m (also known as forward_matrix) calculated probability of observation happening in the order it is (calculalated forward) 
      beta_m (also known as backward_matrix) calculated probability of observation happending in the order it is (calculated backward)
    '''
   # set up the log likelihood, and alpha & beta matrices if needed.
    f_log_lhood, alpha_m = self.run_forward(obs,return_matrix)
    b_log_lhood, beta_m = self.run_backward(obs,return_matrix)
    log_lhood_p = np.logaddexp(f_log_lhood,b_log_lhood) - 2 # average in log space
    
    if (return_matrix):
      return(log_lhood_p,alpha_m,beta_m)
    else:
      return(log_lhood_p)

  def run_baum_welch(self, observations, max_loop_count, epsilon):
    """
      Given a list of observations; and possible initial model; it calculates a 
      maximized local model based on the list of observations.
    Args:
      observations: a list of observation values with observered 'emissions' that represent
                    set of hidden states.
      max_loop_count:  int this is the maximum loop count for the baum_welch algorithm
      epsilon:  float - the delta between that is allowable between likelihoods -- will be translated into log state in this function
      self:         This is an initialization of values that represent the starting states of 
                    a model.  It contains:
                        init_probs aka pi
                        trans_to   aka A_matrix or A_m
                        emissions  aka B_matrix or B_m 
                      These are in our model as Dictiaries.
                        
      
    Returns:
      Maximized model returned as init_probs,trans_to_probs,emit_probs
      
                    
    """
  # variables that end in _m are matrices
  #                end in _v is a 1D matrix (init_probs, slices of matrices)
  #                end in _p is a log state probability (single float)
  #
  # variables labled "current_" are the current baseline matrices/varables
  # variables labeld "new_" are the new state baseline matrices/varables 
  #
  # in our loop we will need to keep track of:
  #    current_log_lhood_p.  this is the average of the forward/backward function likelihood
  #    new_log_lhood_p 
  #  
  #    may need to keep track of -- how many times have we run E-Step/M-Step
  #                              -- scale of how much it's changed?????
  #
    if TESTING: print(f"Starting Baum Welch")
    current_init_v = self.bw_get_init_probs()
    current_trans_to_m = self.bw_get_trans_to_probs()
    current_emissions_m = self.bw_get_emission_probs()
    current_log_lhood, alpha_m, beta_m = self.bw_get_log_lhood(observations[0],return_matrix=True)
    if TESTING: print(f"init: {current_init_v}")
    if TESTING: print(f"trans: {current_trans_to_m}")
    if TESTING: print(f"emit: {current_emissions_m}")
    if TESTING: print(f"alpha: {alpha_m}")
    if TESTING: print(f"beta: {beta_m}")
    
    new_init_v = current_init_v # initially set up new_model to be the same as current
    new_trans_to_m = current_trans_to_m
    new_emissions_m = current_emissions_m
    
    # other setup for loops/counts here
    loop_count = while_loop_count = 0
    changed_model = 0
    added_log_probs = 0
  
    while (loop_count <= max_loop_count): # note convergence is tested below
      if TESTING: print(f"In while loop, loop_count={loop_count}, current_likelihood: {current_log_lhood}")
      for i, obs in enumerate(observations[0]):
        
        new_log_lhood, gamma_m, xi_m = self.bw_EStep(obs, new_init_v, new_trans_to_m, new_emissions_m)
        new_init_v, new_trans_to_m, new_emissions_m = self.bw_MStep(obs, gamma_m, xi_m)
        
        print(f"gamma_m: {gamma_m}")    
            #Now check for convergence!
        compared = self.bw_compare_likelihood(current_log_lhood,new_log_lhood, epsilon)
        match self.bw_compare_likelihood(current_log_lhood,new_log_lhood, epsilon):
          case x if x==GT: # better than old model
            current_init_v = new_init_v
            current_trans_to_m = new_trans_to_m
            current_emissions = new_emissions_m
            current_log_lhood = new_log_lhood
            # ? update hmm????
            # current_log_lhood will be updated the next time we go through the loop
            loop_count = 0 # reset for new_model checking
            changed_model += 1
            added_log_probs += new_log_lhood
          case x if x==LT:
            # keep current_model
            loop_count += 1
          case x if x==EQ:
            # keep current_model
            break # would exit the for loop we just go on to the next observation in the list
        # end of for loop
        
        if (changed_model > 0): # if we changed the model in the for loop
            
           # We need to scale/normalize here with added_log_probs and changed_model
            changed_model = 0 # reset because we've adjusted for it
            added_log_probs = 0 # reset
                
    #end of while loop; we found our local maximum
    
    return(current_init_v,current_trans_to_m,current_emissions) 
  
	
  def bw_EStep(self, obs, pi_m, A_m, B_m): # aka 'Expectation setp'
    ''' Expectation step of Baum-Welch -- needs a single observation,
         Also takes the model that we need to calculate the 'expectation step' on.
         
         obs: sequence of observed emissions.
         pi_m :  matrix in log_space ; this initialization probs for our model
         A_m : matrix in log space ; this is the transition matrix for our model
         B_m: matrix in log space; this is the emission matrix for our model
    Returns
      log_lhood (from average of forward&backward run)
      gamma_m:   matrix in log_space of the probability of a particular state to another state
      xi_m: matrix in log_space of the expectation probabilty of being in a particular state
    '''
    avg_log_lhood, alpha_m, beta_m = self.bw_get_log_lhood(obs,return_matrix=True) #run forward & backward!
  
    # for our example:
    # pi or init is N state probabilities
    # A_m is 2x2 N x N
    # B_m is 2x4 N x M
    # alpha_m should be N x len(obs) T
    # beta_m should be N x len(obs) T
    
    # gamma_m is easy (alpha_m + beta_m) - avg_log_lhood (scalar)
    # this is the probability from transisition from Si -> Sj 
    #    where Si and Sj are states in observation sequence at pos i and j
    # gamma_m = alphpa_m*beta_m/avg_log_lhood or in log space:
    # gamma_m = alpha_m + beta_m - avg_log_lhood  which is N x T matrix
    
    gamma_m = alpha_m + beta_m - avg_log_lhood
    
    # xi_m is the probability of being in state Si at time t and state Sj at time t+1 
    # will need three dimensional array
    # xi[i,j,t] = P(q_t = i, q_t+1 = j given obs & 'model')
    # xi_num (alpha_m(i,t) + beta_m(j,t+1)) + A_m[i][j] + B_m(O_t+1,j)
    
    T = len(obs)
    N = len(self.states)
    
    xi_m = np.zeros((T-1, N, N))
    
    for t in range(T-1):
        # Denominator: P(O | model) = sum over all states at time t
        denominator = 0.0
        for i in range(N):
            for j in range(N):
                emit=obs[t+1]
                B_j_obs_t_plus1 = self.states[j].emission_probs[emit]
                denominator += alpha_m[t, i] + A_m[i, j] + B_t1_obs_k + beta_m[t+1,j]
        
        # Compute xi for all state pairs
        for i in range(N):
            for j in range(N):
                B2_j_obs_t_plus1 = self.states[j].emission_probs[obs[t+1]]
                numerator = alpha_m[t,i] + A_m[i,j] + B2_j_obs_t_plus1 + beta[t+1, j]
                xi_m[t, i, j] = numerator - denominator # more log math!
	
    return(avg_log_lhood,gamma_m,xi_m)

  def bw_MStep(self, obs, gamma_m, xi_m):
    ''' 
    Args:
      gamma_m # P of going from Si to Sj 
      xi_m # P of being in Si at time t and Sj at time t+1
    Returns
      init_hat
      A_hat_m
      B_hat_m
    '''  
    # We need init_probs, A_m and B_m in order to create the next model
    N = len(obs)
    M = len(self.emissions)
    
    init_hat = gamma_m[0, :]
    
    # 2. Re-estimate A (transition matrix)
    # A[i,j] = sum_t(xi[t,i,j]) / sum_t(gamma[t,i])
    A_hat = np.zeros((N, N))
    A_denom = 0
 
    for i in range(N):
        A_denom = np.logaddexp.reduce(gamma_m[:-1, i])#sum over t=0 to T-2
        for j in range(N):
            A_num = np.logaddexp.reduce(xi_m[:, i, j])#sum over all time
            A_hat[i, j] = A_num - A_denom
    
    # 3. Re-estimate B (emission matrix)
    # B[j,k] = sum_t(gamma[t,j] * I(o_t = k)) / sum_t(gamma[t,j])
    B_hat = np.zeros((N, M))
    
    for j in range(N):
        B_denom = np.logaddexp.reduce(gamma_m[:, j])  # sum over all time steps
        for k, emit in enumerate(self.emissions):
            # Sum gamma[t,j] for all t where observation is 'emit
            B_num = self.bm_logsum_emission_probs(obs,gamma_m,j,emit)
            B_hat[j, k] = B_num - B_denom
    
    return init_hat, A_hat, B_hat
    #end M-Step
    
  def bm_logsum_emission_probs(self,obs,gamma,index,emit):
    '''creates sum of all of the gamma probabilities of type emit
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
    
    for t in range(T):
      if (obs[t] == emit):
        running_sum = np.logaddexp(running_sum,gamma[t,index])
    return(running_sum)
  

  def some_default_model(num_states,num_emissions):
    '''
    NOT COMPLETED!!!!
    the idea was given a number of states and a number of emissions 
    to calculate a baseline state probs and emission probs,
    as well as init_probs
    
    returns:
      default_model type lambda
    '''
  
  # set default_model init to have 2D vector 1*N 
    equal_state_log_probs = log( 1/num_states )
#  init_probs = np.fill(...length=max_states,fill=equal_state_log_probs)
  
#  trans_to = np.fill( ..length=max_states, width=max_states, fill = equal_state_log_probs))
  
    equal_emission_log_probs = log ( 1/num_emissions )
    emissions = np.fill( length= max_states, width=max_emissions, fill=equal_emission_log_probs)
  
    default_model = set_model(init_probs,trans_to,emissions)
  
    return(default_model)

  def bw_compare_likelihood(self,current,new,epsilon):
    '''
    Returns:
      comparitor = EQ (1) or GT (2) or LT (0) # defined as global constants
    '''
    diff = new - current 
    if (diff > epsilon):
      return GT
    elif (diff < epsilon):
      return LT
    return EQ
    
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
    b_matrix = np.zeros((n_rows, n_cols)) # initialize B matrix with zeros
    
    # Since we start from the end, the probability of the last observation is 1.
    # So, we initialize the final column of b_matrix with 1s.
    # for state_i, state in enumerate(self.states):
    #  b_matrix[state_i,n_cols] = np.log(1) ; but as np.log(1) is 0; no need!
    
    # Now we can go column by column, filling in each cell in our matrices.
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


TESTING = True
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
  
  our_loop_max = 100
  our_epsilon = 0.01
  new_init, new_trans, new_emit = test_HMM.run_baum_welch(observations,our_loop_max,our_epsilon)
  
  print(f"Baum Welch Recalculated")
  print(f"Init: {new_init}")
  print(f"Transitions: {new_trans}")
  print(f"Emissions: {new_emit}")
  
  
