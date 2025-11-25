# Module 10: Baum-Welch Algorithm
- BINF6250; Fall 2025
- Authors: Brooks Groharing and Jacqueline Caldwell
- Due:  December 03, 2025

# Introduction
Description of the project

# Current todo list:
    X Need to find out why matrices are not being returned properly ?copy
    
    * Compare function isn't working properly?

    * !!! we need a way of breaking out of both the for loop and the while loop (other than max_loop)
    
    * will need to add the probabilities after we have (part of the training)
            looked at all of observations in our training list
    
    * Math issues:
        X make sure that we are only using:
            beta[t-1], gamma[t-1], xi[t-1]
    
        ** Scaling?  (see if (changed_model > 0) in while/for loop in run_baum_welch
            a. keeping it in log space helps, but is not all we need to do?
            b. Consider:
                i. may need to keep track of how many times the models is changed?
                ii. may need to keep track of how many times the likelihood is changed
                iii. may need to adjust by one or both of these factors
                    in loop in run_baum_welch at "changed_model > 0"
    
    * how do we update the HMM model when current is updated>
    * related to above - is the model being updated at all?
    
    * We need a way of initializing the model properly -- should 
        <done-ish> create a function that creates alphabet,etc. and then sends the
        <done-ish> we could just initilize with baseline probabilities (based on the number of emissions in the alphabet - divide by one, etc, etc.)
        * We are going to have to have the hidden states set up as well.
   
    * low priority some_default_model is not really implemented
   
    * Testing!
        
        
# Pseudocode
Put pseudocode in this box:

```{}
# Global variables/constants
EQ = 1
GT = 2
LT = 0

# Notes on the math/algorithm:

# N is our number of states (indexed by i,j (going from S_i to S_j))
# M is our number of emissions (indexed by j and obs_k; prob of having obs_k emission at state S_j, where K is the list of our alphabet!)
# T is our number of observations (time T)

# values in the dictionaries are in log_space (natural log space)

# we take as given that our model is the following three things:
#  init: vector # (N*1)
#  trans_to: matrix # (N*N) ; assoc state_names
#  emission: matrix # (N*M) ; assoc emit_names
 
# a model can represent more than one observation/training

# To help with math translation:
#
# pi = init_probs
# A matrix = trans_to probs
# B matrix = emission probs
# alpha_m = f_matrix from 'forward' (for_likelihood also comes from here) N*T
# beta_m = b_matrix from 'backward' (only t-1 values are valid) N*T-1 (back_likelihood))
# gamma_m = expectation prob of Si -> Sj N*T-1
# xi_m = expectation prob of Si N*T-1

Initialize()
  Args:
	  Observations[]
	  starting_model
	Returns
	  starting_model 
	
	* Obtain observation(s) 
	* Create_alphabet for emissions 
	* setup matrices, current_model

Create_alphabet(observations[])
    creates alphabet given a list of observations
    returns alphabet of emissions
    
    * for obs in observations[]
        all_states += obs
    * return sort(set(all_states))
    
Create_emission_probs(alphabet,state_list)
    creates emisission list
    for state in state_list
    create list of emission probs

create_init_probs()
    creates init_probs for a model (may just set them)
    for state in state_list
        create init prob for the state

create_trans_to_probs(state_list)
    creates trans_to probs for a model (may just set them)
    for state in state list
        create list of state probs
    
run_baum_welch(observations,max_loop_count,epsilon)
	Args:
	starting_model (which has already been initialized by HMM?)
    observations[]
    max_loop_count
    epsilon

	Returns
    current_model from last hmm run through loop 
    #Note: 
    # current_HMM - will be our base model in hmm
    #.  current_log_likelihood
    # new_model - will be the one that we are comparing to it.
        new_log_likelihood
    #
    # we will be getting multiple observations to train our model on.
  
    setup(current_model,current_log_likelihood)
    set new_log_likelihood, new_model

    # other setup for loops/counts here
    while_loop_count = 0
    changed_model = 0
    added_log_probs = 0
    
    while (loop_count <= max_loop_count) # note convergence is tested below
        
        for i, obs in enumerate(observations[])
        
            current_log_lhood, gamma_m, xi_m = EStep(obs,curr_hmm) # on curr_HMM
            new_model = Mstep(obs, gamma_m, xi_m)
            
            new_log_lhood, new_gamma_m, new_xi_m = EStep(obs,new_model)
            
            #Now check for convergence!
            match compare_likelihood(current_log_lhood, new_log_lhood, epsilon) 
                case GT: # better than old model
                    current_model = new_model 
                    # ? update hmm????
                    # current_log_lhood will be updated the next time we go through the loop
                    loop_count = 0 # reset for new_model checking
                    changed_model += 1
                    added_log_probs += new_log_lhood
                case LT:
                    # keep current_model
                    loop_count =+ loop_count
                case EQ:
                    # keep current_model
                    # exit go on to the next observation in the list
            
        # end of for loop
        
        if (changed_model > 0)
            We need to scale/normalize here with added_log_probs and changed_model
            changed_model = 0 # reset because we've adjusted for it
            added_log_probs = 0 # reset
                
    #end of while loop; we found our local maximum
    
    return(hmm.however_we're_going_to_output_current_model)  
	
EStep(obs, hmm) # aka 'Expectation setp'
    obs
    current_hmm # we need init_probs,trans_to,emissions
  Returns
    log_lhood (from average of forward&backward run)
    gamma_m
    xi_m
    
  alpha_m, f_log_lhood = run_forward(obs) #self is set up with current_model
  
  beta_m, b_log_lhood = run_backward(obs) # self is set up with this.
  
  log_lhood = np.addsumlog(f_log_lhood,b_log_lhood) - 2 # log math averaging
  
  gamma_matrix = alpha_m + beta_m - log_lhood
    
  get A_m # hmm."self.trans_to" NEED_IN log_matrix form
  get B_m # hmm."self.emissions" NEED IN log_matrix form
    
  xi_m = (alpha_m + A_m + B_m + beta_m) - 
              (summation of blah blah)
    
  xi_norm_m = xi_matrix - log_lhood # is this normalizing xi????
	
  return(log_likelihood,gamma_norm,xi_norm_m)

MStep(obs,gamma_m,xi_m)
  Returns
    new_model
    
  # We need init_probs, A_m and B_m in order to create the next model
  
  hat_init = gamma(0,i) # the row/column of the inital matrix!
 
  A_denom = logsum_shenanigans(gamma_m(1 to T-1,"i"))
  A_numerator = logsum_shenanigans(xi_matrix(1 to T-1,"[i,j]")
  hat_A_m = A_numerator - A_denom

  B_denom = gamma_m("1 to T","j") 
  B_numerator = logsum_shenanigans(gamma_m("1 to T","j")) # check the equations
  hat_B_m = B_numerator - B_denom
  
  new_model = set_model(hat_init, hat_A_m, hat_B_m)
  return(new_model)
#end M-Step

set_model(pi,A,B)
  ''' sets a model to have new init_probs, trans_to, emission_probs from
    pi, A_m, and B_m
  Args:
    pi: vector of init probabililites
    A: matrix N*N trans_to probabilities
    B: matrixs N*M emission probabilities
  returns:
    ???
  '''
  model['init'] = pi
  model['trans_to'] = A
  model['emissions'] = B
  return(model)
#end set_model
  
logsum_shenanigans(matrix,indexes1,indexes2)
  '''
  Args:
    given a matrix (could be alpha,beta,gamma,xi) 
        N = len(states_list)
        T = len(obs)
        alpha_m: N * T
        beta_m: N * T-1
        gamma_m: N * T-1
        xi_m N * T-1 
  returns 
    addsumexp of the two values from that matrix
  
  matrix.dims
  check that indexes1, indexes2 is appropriate for the matrix dims.
  

some_default_model(num_states,num_emissions)
     creates a dictionary of type lambda
  returns:
    default_model type lambda
 
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
(Cited)
Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257–286.

Yang, X. (2024). Hidden Markov model based network security posture prediction model. Applied Mathematics and Nonlinear Sciences, 9(1), 1–17.
