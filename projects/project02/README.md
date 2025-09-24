# Introduction

* Authors:  Brooks Groharing, Jacqueline Caldwell
* Class: BINF 6250; Fall 2025
* Project: Project02
* Due Date: 9/24/2025

## Description
The goal of this project is to implement Markov model technology in stages, first from a 1 level Markov Model, then a N level Markov that is able to take in a book size length of text, and finally a book/text generator that uses the built Markov model to create the text or a book.   

The markov model itself is created as a dictionary of dictionaries, and was modified from inital form to have the probability of particular text following a string of text, which is what allows us to create text from our markov_model.  In addition to the probabilities, we are using np.random.choice to select (if there are options) the next section of text for our book.

# Pseudocode
Put pseudocode in this box:

``` {python}
build_markov_model(markov_model, string, order=n):
    """Build or add to a Nth order Markov model given a string of text"""
    
    extract list of words from text
    
    get first n words from list
    initialize markov_model[START][first_n] to 1
    
    append dummy END word to text
    
    for each n-length string of words:
        get next word
        increment markov_model[string][next_word]
        
    return markov_model



get_next_word(current word, markov_model):
    """Randomly generate a valid next state given a markov model and a current state"""
    get valid output states from current word
    sum counts from out_states
    
    probs = new list
    for output state:
        probs = count from state / total
        
    return(next word from states, weighted by probs)


generate_random_text(markov_model):
    """Generate a string of text using a markov model"""
    
    initialize empty string as sentence
    
    initialize state to "START"
    next_word = get_next_word(state, markov_model)
    
    until next_word is END:
        state = get_next_word(state, markov_model)
        add next_word to sentence
        
        drop oldest word from STATE
        append next_word to STATE
```

# Code
Please see the file project02.Rmd in this directory.

# Successes




# Struggles

## Brooks

## Jacque
* Struggles with Explorer/OnDemand/RStudio; uncertain if disk is filling up or there is a protection problem, or a system 
* Read file code never returns a null line for the file one_fish_two_fish.txt.  When using the code that was given to us for reading sonnets.txt ; I was having problems because the markov model was never being built, turns out that was because line never returned "" from the file.  I do not know why this is, but it is repeatable.  Changed code to instead end the loop when the open/read file loop as my prior group did for project01, this seems to work.
* Comfort with git is definately improving but getting a smooth workflow going continues to be a challenge.
* GRRRR ondemand and RStudio issues/Explorer issues.

# Personal Reflections
## Group Leader
Group leader's reflection on the project

## Other member (Jacque)
This project was more straightforward than it first appeared.  Given what I was understanding at class, I thought we would have to be implementing something with multi-dimentional matrices (N level would be an N level matrix).  Coding became much easier once I figured out that was not the case. 

I did not always feel that I knew what my next step to do in this project might be. I found myself often waiting for the next step in order to get something done.  I think I'm struggling for feeling out of control over the entire project.  I'm used to being responsible for anything that is happending on a project, and ultimately responsible for what anyone else does.   Harder to do when schedules are different, and people's working styles are different.  

Debugging is a pain when your system won't stay up.

Rstudio does not give helpful syntax errors as it's row counters are completely wrong, and "relative row counters" are even more wrong.  Even then the syntax errors are not even about the real problem, they are usually about another downstream text issue.  

# Generative AI Appendix

We did not make use of any generative AI tools in this project.