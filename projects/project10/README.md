# Module 10: Baum-Welch Algorithm
- BINF6250; Fall 2025
- Authors: Brooks Groharing and Jacqueline Caldwell
- Due:  December 03, 2025

# Introduction
Implement the Baum-Welch portion of our hidden Markov model suite of functions. Given multiple
sequences and a starting model (with sample/simplified init, transition and emission probs) calculates a
localized maximized model solution.

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
    creates emission list
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
  
  hat_init = gamma(0,i) # the row/column of the initial matrix!
 
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
    pi: vector of init probabilities
    A: matrix N*N trans_to probabilities
    B: matrix N*M emission probabilities
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
1. The idea of a particular item converging is not difficult once you know what thing
is converging, and how your algorithm is changing it.
2. As long as you stay consistent, doing this algorithm completely in log_math was not
as difficult as we thought it might become.
3. (Jacque) did go back to the backward algorithm and fixed her issue of wanting to 
have the backward matrix longer by one element.  Mea culpa to all.  I have verified 
with different implementations online that they are done placing the probability space
values in the t-1 position, and then starting with the t-2 position and filling down 
until you get to the zeroth position.
4. (Brooks) I successfully  added code to update our states' probabilities "in place"
when baum-welch is run. This was something we struggled with conceptually, and was
therefore one of the last features to be added.

# Struggles
1. (Jacque) Struggled to understand the math behind the algorithm, more specifically,
during the E-Step, the second part of that function.   The problem was understanding
the part that calculates the probability of a particular state happening at a particular
time, or in other words, the probability of being in state i at time t and being 
in state j at time t+1.  Implementing the algorithm wasn't that hard, but wrapping 
my head around the three/four (i,j,t and k) variables at the same time was.   
3. (Jacque) continued to struggle with object-oriented programming.  I may have spent 
more time swearing at the fact that I couldn't debug something because Python considered 
things in our class 'private'.  grrrrrr.   Unfortunately, designing software when you
don't know most of the constraints at the beginning are very difficult.  I'm sure I 
missed some opportunities to make this more object-oriented.
4. (Jacque) had difficulty wanting to break up parts of the main baum_welch loop as I 
wasn't sure of the efficiency of constantly handing large matrices to multiple small 
procedures.  If I had been using our hmm class object more effectively I expect that
our Baum-Welch loop would be much smaller.  I found myself constantly assigning and 
reassigning our model (which consists of at least three things (which were basically 
already in our hmm model).  This is just me struggling with O-O programming, I expect.
6. (Jacque) I continue to struggle with the idea of not knowing if my answer is 
'correct' or not.  
7. (Jacque) pseudocode did not work out the way I had planned.
8. (Brooks) Our class constructor still mandates that we hand it hardcoded dictionaries of
   probabilities. It's good to have that option, but ideally we shouldn't *have to* if we're
   going to generate new values with B-W anyway.
   
   I wanted to simplify HMM.__init__() to intialize a "blank" HMM from just a list of state
   names and an emissions alphabet, and then create an optional function outside the class called
   `HMM_from_probs()` containing the logic to create, populate, and return an HMM based on hardcoded
   probability dicts. I experimented with this a bit on my local branch, but did not finish the
   implementation in time to push it before the primary deadline.

   I'm annoyed that I didn't finish this, since it's something we anticipated and planned for way back
   in project 08 (it's part of why we chose an object-oriented approach in the first place!) I'm still
   hoping to go back and add it, time permitting.

# Personal Reflections
## Group Leader (Brooks)
This was certainly the most difficult part of the part of the larger HMM project to plan and implement. Overall I consider Project 10 a success, since our class has the baseline B-W functionality in place; that being said, I would like to devote a little more time to finishing some incomplete "quality of life" features, polishing our code and documentation up to the level of our project 8 and 9 functions, etc.

I really enjoyed designing and implementing our HMM module. I'm happy with what we've accomplished, and
feel okay calling it "done" (even as I see ways we could polish, improve, and extend the project).

## Other member (Jacque)
After I was done working on the project, I am planning to continue researching online code and
papers on the backward algorithm, because the idea of having a 1 in a sum of 
probabilities in an array bothers me, especially because we are using it in a summation
later on, though I have a feeling I'm not going to get a satisfactory answer.  
           
Additionally, I'm looking at how the pfam and other online sequence databases are using 
profile hmms to describe sequences/proteins for comparison and storage.

# Generative AI Appendix

# Citations
(Use of HMMs/Curiosity)
Eddy S (1998). Profile hidden Markov models.  Bioinformatics Review, 14(9), 755-763.

Krogh A, et al (1994). Hidden Markov models in computational biology.  Applications
to protein modeling.  J Mol Biol., 235(5), 1501-31.

(Reviewed by JC and used for understanding of Xi/E-Step implementation)
Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257–286.

Yang, X. (2024). Hidden Markov model based network security posture prediction model. Applied Mathematics and Nonlinear Sciences, 9(1), 1–17.
