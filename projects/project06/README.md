# Introduction
Description of the project

# Pseudocode
Put pseudocode in this box:

```
Some pseudocode here
```

# Successes
Coding the neighbor-joining process went smoothly. We successfully produced distance matrices, calculated limb lengths based off of these matrices, and recursed this process. 

# Struggles
Formatting the leaves and their limb lengths so that ete3's Tree function could read and produce an unrooted Newick tree was our biggest struggle. Representing the limb lengths and their relation to the other nodes got complicated if the two taxa that were merging in the neighbor joining process were separate from the merged taxa created in a previous iteration. In the end, we could not find a method that universally handled the formatting, and instead implemented a couple if-statement conditions to handle these various cases. 

# Personal Reflections
## Group Leader
Group leader's reflection on the project

## Jason
Conceptually I thought this project was quite straightforward- the math for neighbor-joining was not too complex and combining branches as a form of making a common ancestor became easier to map out throughout the project. Coding the format for the Newick Tree was tough, especially when accounting for the different orders branches could combine in. In the end though, it was very rewarding fleshing out my understanding of phylogeny and how it works with bioinformatics.

# Generative AI Appendix
As per the syllabus
