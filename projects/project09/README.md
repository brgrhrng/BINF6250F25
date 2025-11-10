# Module 09: Forward, Backward, and Forward-Backward Algorithms
- BINF6250; Fall 2025
- Authors: Brooks Groharing and Jacqueline Caldwell
- Due:  November 12, 2025

#Introduction

This week we will be continuing to implement Hidden Markov
Models by implementing the 'forward', 'backward' and '
forwardbackward' algorithms.

The 'forward' algorithm is similar to our Viterbi algorithm,
in that it calculates the probability of going down a particular
path of hidden states while loging the probabilites of getting to
each observed state and it's emission probability.  Once we find 
the probability for each state in each path, we then add those
together to get our final probability (Viterbi takes the max of
these pathways at this point).  The algorithm continues from
there to calculate the possibility of the complete observation 
given our hidden Markov Model.

The 'backward' algorithm, operates as the 'forward' algorithm
does, summing the probabilities as we go along, however
instead of starting with the left side of the observation and 
moving to the right, we start at the end on the right side, and 
move to the left until the matrix is complete, and we have the 
calculation for the probability of the complete observation 
given our hidden Markov Model.

The 'forwardbackward' algorthm, uses both the forward and the
backward models in combination to calculate calculate the
probability of a position in our sequence being assigned a
particular hidden state (used for posterior decoding).

# IMPLEMENTATION
```{} # pseduocode here
Need to:
	Implement Forward:  
	    * Sum vector for each v_matrix square instead of ‘max’
        * ? Normalize each column to sum to 1.
        * return v_matrix.
        * Formula for Forward:
          F_t is the forward matrix (looking at prior)
          a_ij is the transition from i to j
          b_j (obs state) is the emision at this state in obs
    F_t(i) == summation 1 to N (a_ij) bj(Ot_1) F_t-1(j)

	Implement Backward:  
    	* run _fill_viterbi backwards; 
        * Init for matrix[:,length(obs)] = 1 (our matrix will 
            to multiply into the equations as the prior states.
        * ? Normalize by making column to sum to 1.
        * Termination ? do we need init states here (NO)
        * formula for backward: 
            B_t is our backward matrix
            a is our transition matrix and 
            b is our emission matrix.
	B_t(i) == summation N-1 to 0 (a_ij) bj(Ot_1) B_t+1(j)
	                                           
	Implement Forward/Backward:  
	
	Take forward and backward matrices:
	P(x(k)) given y(1), y(2), y3,…,y(t) for some k<t;
	Probability distribution of hidden states for a point in time k, relative to time t.
	a = transition probs 
	b = emission probs 
	f0 = initialization probs gives you:

	F = forward probs  (v_matrix col); 
			C(0) contains initialization probs * emission probs
	B = backward probs (v_matrix col; reverse order? (no need))

Formula for forwardbackward probs:
      FB = forwardbackward probs = F*B or
         = log10(F) + log10(B)

For implmentation of forwardbackward.      
You just run forward, run backward, and then put them together with the function FB.

Starting with our implementation from or HMM/Viterbi, including our Class structure:
class HiddenState:
    name: a unique name representing the state
    init_prob: probability of state coming from the start node
    transition_to: dict of probabilities of transitioning to each out_state
    emission_probs: dict of possible emissions and their probabilities
  def __init__(self, name: str, init_prob: float, emissions_dict: dict[str,float]):
  def set_transitions(self, transitions_dict):
  def emit(self):
 
class HMM:
	states = List of HiddenState objects
	emisions = list of list of dicts
	state_names = list of possible state names.
	New_state = initial set up for HiddenState object
def __init__(self, init_probs, trans_probs, emit_probs)     
def run_viterbi(self, observations) 
def __fill_viterbi_matrix__(self, observations, log_values = True):             
def __traceback_viterbi__(self, traceback_pos, backptrs):
```
## Sample Input
```
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
```
## Expected output from given input
```{}
Doing the math matrices by hand
  v_matrix
      A         T          G         C        A        A    
      0         1          2         3        4        5    
0 E 0.6*0.3    0.8*0.3   0.8*0.2   0.8*0.2  0.8*0.3  0.8*0.3 
               0.3*0.3   0.3*0.2   0.3*0.2  0.3*0.3. 0.3*0.3
                      
for->0.18      0.0594    0.13068   0.002875 0.0009487 0.0003131
bac->0.0003415 0.001875  0.005271  0.01597  0.0726    0.33

1 I 0.4*0.1    0.2*0.1   0.2*0.4   0.2*0.4  0.2*0.1  0.2*0.1 
     0.04      0.7*0.1   0.7*0.4   0.7*0.4  0.7*0.1  0.7*0.1

for->0.04      0.0036    0.001296  0.0004666 4.199e-05 3.779e-06
bac->3.401e-05 3.779e-04 0.001050  0.002916  0.0081    0.09 

total for ->  0.0003131 + 3.779e-06 = 3.168e-04
      bac ->  3.401e-05 + 0.0003415 = 3.755e-04
      
```

## Other issues

I proposed the following issue to Claude.ai 11/10/2025

Prob(going backwards)*Prob(emission) + Prob(going forwards)*Prob(emissions); I have been doing the probabilites in log10 so that underflow isn't an issue with the floating point math, but the + at the end is going to get in the way of this...

Per Claude.ai:
Identified log-space addition problem and formulated numerically stable solution.

You're hitting the classic 'log-space addition' problem! You
can't just add log probabilities directly because log(a + b) ≠ 
log(a) + log(b).

The solution is the log-sum-exp trick:
If you have two log probabilities log_p1 and log_p2, compute 
their sum in log space like this:

log(p1 + p2) = max(log_p1, log_p2) + 
log10(1 + 10^(min(log_p1, log_p2) - max(log_p1, log_p2)))

Why this works:

Factor out the larger value to prevent overflow: log(a + b) = 
log(a(1 + b/a)) = log(a) + log(1 + b/a)

Since you're using log10, raise to base 10 instead of e

In practice:
python
def log_sum(log_a, log_b):
    if log_a > log_b:
        return log_a + math.log10(1 + 10**(log_b - log_a))
    else:
        return log_b + math.log10(1 + 10**(log_a - log_b))

This keeps everything in log space and avoids underflow because 
the exponent (min - max) is always ≤ 0, so you're computing 
10^(negative number) which stays in a safe range.

Is this for a Hidden Markov Model or similar algorithm?
RetryClaude can make mistakes. Please double-check responses. 
Sonnet 4.5

# Successes

# Struggles

# Personal Reflections
## Group Leader
Group leader's reflection on the project.

## Other member
Other group member's reflection on the project.

# Generative AI Appendix and references
# Cite:
Discussions with Claude.ai (Sonnet 4.5): to clear up understanding of summing log products, on-line, November 10, 2025.




