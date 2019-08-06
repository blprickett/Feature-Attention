# Probabilistic Feature Attention

The script "PFA_Learner.py" will run a MaxEnt model with Probabilistic Feature Attention (PFA). To read more about PFA, see [this handout](https://people.umass.edu/bprickett/Downloads/UNC%20Colloquium%20Handout%20-%20Prickett%202019.pdf) or [this manuscript](https://people.umass.edu/bprickett/Downloads/PFA-Manuscript-Prickett2019.pdf). The python script requires the following packages:

* numpy
* re
* matplotlib
* sys
* itertools
* random
* sympy

It takes five inline arguments:

* The # of epochs (i.e. passes through the data) to train the model for
* The learning rate
* A pattern label (see below for the ones I used for the manuscript above)
* Attention probability (i.e. how likely are you to attend to each variable?)
* The # of repetitions to run the model for

And requires three kinds of input file:

* A "Features" file that specifies all possible segments and the featural representations. Name of the file should be "\[pattern label]\_Features.txt". 
