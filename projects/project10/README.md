# Module 10: Baum-Welch Algorithm
- BINF6250; Fall 2025
- Authors: Brooks Groharing and Jacqueline Caldwell
- Due:  December 03, 2025

# Introduction
Description of the project

# Pseudocode
Put pseudocode in this box:

```{}
Pseudocode:

Todo:
  what do we send forward * backward
  what is the math for gamma (fowardbackward)
  implement fastforward
  how can I access matrices in our class?
  are the results from forward and backward in log space (I think they are?)
  
  do I need a dtype lambda (model dictionary) or not?


EQ = 1
GT = 2
LT = 0

# N is our number of hidden states
# M is our number of emissions
# T is our number of observations (time T)

#values in the dictionaries are in log_space (natural log space)

lambda dictionary with:   # not sure I need this as we already have hmm.
  init: vector # (N*1)
  trans_to: matrix # (N*M)
  emission: matrix # (N*M)
  
# Notes:
# these are all associated with a single observation 'name'
# A matrix is trans_to matrix
# B matrix is emission matrix
# alpha_matrix is output from 'forward' (likelihood also comes from here)
# beta_matrix is output from 'backward'
# gamma_matrix is combined alpha & beta
# xi is combined alpha, A, B and beta

Initialize()
  Args:
	  Observations[]
	  starting_model of type lambda
	Returns
	  starting_model of type lambda 
	
	Obtain observation(s) (strings, containing lists of letters/strings with emissions)
	Create alphabet for emissions (length of this alphabet = M)
	Create hidden states (length # of states = N)
	If starting_model == ""
	  starting_model = some_default_mod()
	return starting_model

Setup_for_loop()
  Args
	  Obs - list of observed states 
	  Model - type lambda ; can't be hmm.
  Returns:
    hmm class
    
    hmm = hmm.class # Create/initialize our hmm class with name, etc.
    
    confirm sum(himm[init]matrix) = 1
    confirm trans_to appropriate cols/rows sum to 1
    confirm emission appropriate cols/rows sum to 1
    scale if needed!
    
    put the matrices into the hmm
    
    return(hmm)
    
BaumWelch()
  '''
	Args:
    observations[]
    max_loop_count
    epsilon
    starting_model optional lambda type default=""
	    
	Returns
    current_model from last hmm run through loop 
  '''
  
  #Note:  during this in some places I will refer to hmm and in others current_matrix
  # this signifies that I have not decided how I want to do this yet... 
  # new_model can just be a "dictionary" with the current vector,matrix,matrix triplet
  # and associated new_log_likelihood as if it is not better than current, we ditch it.
  # it may be that we just need "one more than N" hmm -- that one could be used as the 
  # "new_model" and used for comparisons, resetting values each time we go through the FOR
  # loop.
    
    starting_model = Initialize(observations[,],starting_model)

    curr_HMM = Setup(observations[0],starting_model) # hmm now contains the current_model
    current_log_likelihood = fast_forward(obs,starting_model)
    loop_count = 0
    
    for i, obs in enumerate(observations[]) (starting at 0 going to len(observations[]))
    
      while (loop_count <= max_loop_count) # note convergence is tested below
        
        current_log_likelihood, gamma_matrix, xi_matrix = EStep(obs,curr_hmm) # on curr_HMM
        
        new_model = Mstep(obs,gamma_matrix,xi_matrix)
        new_log_likelihood = fast_forward(obs,new_model)
        
                #Now check for convergence!
        match compare_likelihood(current_log_likelihood, new_log_likelihood,epsilon) 
        case GT: # better than old model
          current_model = new_model # update hmm here!
          loop_count = 0 # reset for new_model checking
        case LT:
          # keep current_model
          loop_count =+ loop_count
        case EQ:
          # keep current_model
          break # exit while loop to go on to the next observation in the list
        #end of while loop; we found our local maximum, now we go on to the next observation

      if we haven't gone through all the observations then
        #Initialze for next run through the model with the next observation, create a new hmm  
        
        # we could save the old hmm in a list here if we wanted....
        # OR we could just reuse the old HMM with the new model by re-initializing?
        # OR we could just create a new instance of hmm and forget the old one existed
        HMM = Setup(obs[i+1],current_model) # for the next obs; start with current_model
        loop_count = 0 # reset loop count to zero
        
      # end while loop
    #end for loop
    
    return(hmm.however_we're_going_to_output_lambda.current_model)  
	
EStep(obs, hmm) # aka 'forwardbackward' pass
  Args: 
    obs
    current_hmm # we need init_probs,trans_to,emissions
  Returns
    log_likelihood (from forward run)
    gamma_matrix
    xi_matrix
    
  alpha_matrix, log_likelihood=forward(obs,?model) #self is set up with current_model
  
  beta_matrix = backward(obs,?model)
  
  gamma_matrix = alpha_matrix + beta_matrix
    
  get A_matrix # hmm."self.trans_to" NEED_IN log_matrix form
  get B_matrix # hmm."self.emissions" NEED IN log_matrix form
    
  xi_matrix = alpha_matrix + A_matrix + B_matrix + beta_matrix
    
  #	Normalize the matrices we're sending back
  gamma_norm = gamma_matrix - current_log_likelihood
	xi_norm = xi_matrix - current_log_likelihood
	
	return(log_likelihood,gamma_norm,xi_norm)

MStep()
  Args
    obs
    gamma_matrix
    xi_matrix
    
  Returns
    hat_init
    hat_A_matrix
    hat_B_matrix
    
  # We need init_probs, A_matrix and B_matrix in order to create the next model
  
  hat_init = gamma(0,i) # the row/column of the inital matrix!
 
  A_denom = logsum_shenanigans_summation(gamma_matrix(1 to T-1,"i")) ???? check the equations
  A_numerator = logsum_shenanigans_summation(xi_matrix(1 to T-1,"[i,j]") ??? check equations
  hat_A_matrix = A_numerator - A_denom

  B_denom = gamma("1 to T","j") # check the equations
  B_numerator = logsum_shenanigans_summation(gamma("1 to T","j")) # check the equations
  hat_B_matrix B_numerator - B_denom
  
  new_model = set_model(hat_init, hat_A_matrix, hat_B_matrix)
  return(new_model)
#end M-Step

set_model(pi,A,B)
  ''' Creates a model dictionary lambda from pi,A, and B
  Args:
    pi: vector of init probabililites
    A: matrix of trans_to probabilities
    B: matrixs of emission probabilities
  returns:
    model type lambda
  '''
  model['init'] = pi
  model['trans_to'] = A
  model['emissions'] = B
  return(model)
#end set_model
  
fast_forward(obs,a_model:lambda)
  '''Calculates the log_likelihood of a particular model and returns it.
  Args:
    obs: list of emissions of length T
    a_model of type lambda with init_probs, trans_to probs and emission_probs in matrix form
  Returns:
    total_log_likelihood float
  '''
  run 'forward' without the matrix, we only need the probability here.
  
logsum_shenanigans_summation(matrix,index1,index2)
  '''
  Args:
    given a matrix (alpha,beta,gamma,xi) matrix of (N*N) of _LOG SPACE_ probabilities
    
  # NOT SURE WHAT TO DO HERE YET need to know math for M-step first.
  

some_default_model(num_states,num_emissions)
  ''' creates a dictionary of type lambda
  returns:
    default_model type lambda
  '''
  # set default_model init to have 2D vector 1*N 
  equal_state_log_probs = log( 1/num_states )
  init_probs = np.fill(...length=max_states,fill=equal_state_log_probs)
  
  trans_to = np.fill( ..length=max_states, width=max_states, fill = equal_state_log_probs))
  
  equal_emission_log_probs = log ( 1/num_emissions )
  emissions = np.fill( length= max_states, width= max_emissions, fill=equal_emission_log_probs)
  
  default_model = set_model(init_probs,trans_to,emissions)
  
  return(default_model)

compare_likelihood(current,new,epsilon)
  Args:
    current
    new
    epsilon
  Returns:
    comparitor = EQ (1) or GT (2) or LT (0) # defined as global constants

  diff = new - current
  if (diff > epsilon)
    return(GT)
  elif (diff < epsilon)
    return(LT)
  return(EQ)

```

# Successes
Description of the team's learning points

# Struggles
Description of the stumbling blocks the team experienced

# Personal Reflections
## Group Leader (Brooks)
Group leader's reflection on the project

## Other member (Jacque)
Other members' reflections on the project

# Generative AI Appendix
As per the syllabus
