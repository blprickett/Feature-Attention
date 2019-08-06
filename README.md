# Probabilistic Feature Attention

The script "PFA_Learner.py" will run a MaxEnt model with Probabilistic Feature Attention (PFA), using Python 3. To read more about PFA, see [this handout](https://people.umass.edu/bprickett/Downloads/UNC%20Colloquium%20Handout%20-%20Prickett%202019.pdf) or [this manuscript](https://people.umass.edu/bprickett/Downloads/PFA-Manuscript-Prickett2019.pdf). To install Python 3, see [this website](https://www.python.org/downloads/).

### Dependencies

* numpy
* re
* matplotlib
* sys
* itertools
* random
* sympy

### Commandline Arguments

* The # of epochs (i.e. passes through the data) to train the model for
* The learning rate
* A pattern label (see below for the ones I used for the manuscript above)
* Attention probability (i.e. how likely are you to attend to each feature?)
* The # of repetitions to run the model for

So, for example, if you wanted to run the model on a pattern called "Ident_Bias", for 5 repetitions of 100 epochs, with a learning rate of .05 and an attention probability of .25, you would run the following command:

```shell
python PFA_Learner.py 100 .05 Ident_Bias .25 5
```

### Input Files

Examples of each of these input files are included in this repo.

#### Feature File

* Comma delimited file, specifying all possible segments and their featural representations (loosely following the format used by Hayes and Wilson 2008). 

* First column gives the segments (first row of this column is blank).

* First row gives the features (first column of this row is blank)

* All other cells show whether that row's segment is "+" or "-" for that column's feature (only binary feature schemas are allowed).

* Name of the file should follow the format "\[pattern label]\_Features.csv". 

* A "TD" file that specifies the training data.
* An "AmSegs" file that specifies which segments are ambiguous with which when each feature is ignored.
