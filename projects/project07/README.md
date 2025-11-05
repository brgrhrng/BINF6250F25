# Project 07: Burrows-Wheeler Transform
## Introduction
The goal of this project was to use properties of the Burrows-Wheeler transform to enable highly efficient string matching, as in major bioinformatics aligners. I took an object-oriented approach, implementing a custom FM-Index class that computes and stores the BWT of input strings efficiently using suffix-arrays, with built-in methods to align a substring to the input string with last-to-first matching.

## Pseudocode
```
class FMIndex:
  """Object storing a BWT-compressed string, with methods for efficient alignment."""
  Data:
    bwt: final column from sorted BWT matrix of input string
    suffix_array: suffix array of input string. This could instead by calculated on the fly
    _lesser_chars: internal dict enabling fast alignment (see calc below)
    _occurrences_map: internal dict enabling fast alignment (see calc below)
    
  Methods:
    init(string):
      """Initialize object, storing the burrows-wheeler transform of string and some auxiliary data."""
      if string does not end with an END_CHAR, append it # END_CHAR = $
      store bwt_fast(string) in self.bwt
      store suffix_array(string) in self.suffix_array
      store _count_lesser_chars(string) in self._lesser_chars
      store _build_occurrences_map(string) in self._occurrences_map

    locate(pattern):
      """Get list containing indexes of every occurrence of string 'pattern' in encoded string."""
      set lower search bound to 0
      set upper search bound to length of string
      loop through chars in pattern string, backwards
        if lower bound > upper bound:
          RETURN empty list # no valid matches
        recalculate lower, upper using _update_search_window(current_char, upper, lower)

        RETURN list of suffix_indexes between lower and upper bounds
    
    count(pattern):
        get list of occurrences from locate(pattern)
        RETURN length of list
      
    _count_lesser_chars_(string): # PRIVATE
      """Generate a dictionary where keys are chars in string, and values are counts of
      chars which are lexicographically < char in string"""
      get unique characters from string as alphabet
      sort alphabet lexicographically

      initialize empty dict count_dict
      initialize cumulative_count as 0
      for char in alphabet: # ascending lexicographically!
        count_dict at char = cumulative_count
        increase cumulative_count by count of char in string
      RETURN count_dict

    _build_occurences_map_(string): # PRIVATE
      """Creates an occurence dict, where keys are unique chars in ref_string 
      alphabet, and values are lists such that:
        occur_dict[char][i] = count of char in ref_string[0:i]"""

      initialize empty dict occurrence_map # to contain (alpha,list) pairs
      loop through char in string:
        for every list already initialized inside occurence_map:
          append a copy of last value in list to its end
        if occurence_map[char] does not have a list, intialize it with list[0] = 0
        increment the last value in the list at occurence_map[char]

    _update_search_window(current_char, upper, lower): # PRIVATE
        """Adjust the span of our search window inside locate()"""
        if lower is 0
          set new lower to _lesser_char_counts[current_char]
        else
          set new lower to lesser_char_count[current_char] + _bwt_occurence_map[current_char][lower-1]
        set new lower to lesser_char_count[current_char] + self._bwt_occurence_map[current_char][upper-1]
        RETURN new lower, new upper


func suffix_array(string):
  """Calculate suffix-array for a given string."""
  suffixes = list of "character:end of string" strings for character in string
  assign an index to each suffix
  lexicographically sort suffixes, suffix_indexes together
  RETURN sorted indices

func bwt_fast(string):
  """Efficiently calculate burrows-wheeler transform of string using its suffix index""""
  save suffix_array(string) as suffix_indexes
  for each suffix, get the character in string preceding it and add to an out_string
  RETURN out_string # This is the Burrow-wheelers transform!
```

## Successes
I was successful in implementing the intended functionality of FMIndex. In working on this, I developed an understand of how the BW-T properties and suffix arrays enable efficient alignment--this was unintuitive, but interesting.

More broadly, I am glad I finally took an object-oriented approach to an assignment in this course. While I certainly could have done all of this using separate functions, conceptualizing FMIndex as essentially a data type helped me to understand and break down the problem; I also think this is one of the cleanest scripts I've written for the class.

## Struggles
The primary challenge lay in wrapping my head around how/why a compression algorithm is being used for sequence alignment. The actual steps in the implementation were actually easier than some other projects, but making it make sense required really digging into the logic underlying each "basic" step.

I sometimes struggled conceptually with what values should and should not be cached inside the FMIndex, versus computed on the fly. I'm still not sure if I should have stored the suffix array or not--the project description mentions that you could store just a subset of suffix indices, but I am not sure exactly how to implement this in practice. I'm tempted to look into existing FMIndex implementations when I have time, just to see how they handled it.

## Personal Reflections
Due to some extenuating circumstances, I requested to work alone this week. I am overall happy with the amount I got done, and the quality of the code I submitted. Still, having missed out on collaboration this week, I'm definitely looking forward to receiving feedback during the code review.

While my alignment/counting methods are functional, I ran out of time to implement one feature I intended. Namely, my FMIndex stores the full-length burrows-wheeler transforms, instead of compressing it via run-length encoding. This is annoying: first, because a tendency towards repeated characters is one of the notable properties of the BW-T, making it ideal for compression; and second, because it means my implemenation isn't quite optimized for the normal bioinformatics use-case (aligning to a long sequence, like a reference genome). I really want to go back and add this at some point... especially since it looks like a nice opportunity to write a `@property` function, which I haven't gotten to practice much.

# Generative AI Appendix
I did not use generative AI in this project.
