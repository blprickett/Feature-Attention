# Probabilistic Feature Attention

The script "PFA_Learner.py" will run a MaxEnt model with Probabilistic Feature Attention (PFA), using Python 3. To read more about PFA, see [this handout](https://people.umass.edu/bprickett/Downloads/UNC%20Colloquium%20Handout%20-%20Prickett%202019.pdf) or [this manuscript](https://people.umass.edu/bprickett/Downloads/PFA-Manuscript-Prickett2019.pdf). [Click here](https://www.python.org/downloads/) for instructions on installing Python 3.

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

Examples of the input files that I've used in my research so far are included in this repo. To run the script, all input files must be in the same directory as "PFA_Learner.py" and follow the naming conventions laid out below.

#### Feature File

* Comma delimited file, specifying all possible segments and their featural representations (loosely following the format used by Hayes and Wilson 2008). 

* First column gives the segments (first row of this column is blank).

* First row gives the features (first column of this row is blank)

* All other cells show whether that row's segment is "+" or "-" for that column's feature (only binary feature schemas are allowed).

* Name of the file should follow the format "\[pattern label]\_Features.csv". 

#### Training Data File

* Comma delimited file, showing the attested words in each language. The model can only handle phonotactic learning at this point, so no information about underlying representations should be given.

* Each row is either a new language or a set of (optional) withheld data to test the model on. 

* For languages, give the name of the language in the first column of the row. For withheld data, the first column should say "Nonce".

* Every subsequent column represents a training datum in that language. The words you specify are treated as tokens, not types. So if you want to test the effects of frequency, you need to repeat words that are meant to be more frequent.

* Name of the file should follow the format "\[pattern label]\_TD.csv". 

#### Ambiguous Segment File

* Comma delimited file, giving symbols that each segment should be mapped to, given a particular set of ignored features. For example, if voicing is ignored, \[t\] and \[d\] might both be represented with the symbol "D".

* Creation of this file could be automated (using the information in the feature file), but I found that it was easier to just to specify these by hand.

* The first column specifies which features are ignored (first row of this column is blank). If more than one feature is ignored, seperate them with the "+" symbol. Every combination of the features from the feature file must be represented here.

* The first row specifies which segment is mapping to the ambiguous symbols below it (first column of this row is blank). This should be the same as the first row in your features file.

* All other cells show what symbol represents the column's segment when the row's set of features are being ignored. The script is case-sensitive so you can treat upper and lowercase letters as seperate symbols.

* Name of the file should follow the format "\[pattern label]\_AmSegs.csv". 

### References
Hayes, B., & Wilson, C. (2008). A maximum entropy model of phonotactics and phonotactic learning. *Linguistic inquiry, 39(3)*, 379-440.
