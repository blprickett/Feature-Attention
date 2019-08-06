import numpy as np
import re
from matplotlib.pyplot import plot, ylabel, show, legend, xlabel, title
from sys import exit, argv
from itertools import product, combinations, combinations_with_replacement
from random import uniform, shuffle
from sympy.utilities.iterables import multiset_permutations

######USER SETTINGS#############################
EPOCHS = int(argv[1]) #How many updates will happen (I used 200)
ETA = float(argv[2]) #Learning rate (I used .05)
PATTERN =  argv[3] #Label for the pattern its learning--this should match your training data files
#The pattern names I used were:
#"Ident_Bias"           Bias for identity-based phonotactics (Gallagher 2013)
#"Ident_Generalization  Generalization of identity (Berent 2013; Gallagher 2013)
#"IntraDim_Bias"        Bias for intradimensional patterns (Moreton 2012)
#"Sim_Generalization"   Similarity-based generalization (Cristia et al 2013)
ATTENTION_PROB = float(argv[4]) #Probability that each feature will be attended to (1=vanilla MaxEnt)
REPS = int(argv[5]) #Repitions (I used 15--this can take a while though)
WD= "" #If you want the script to use a different working directory, you can hard-code that here
################################################     

######FUNCTIONS#################################      
def get_predicted_probs (weights, viols):
    '''Takes a 1d vector of weights and a 2d vector of violation profiles as
       input. Also requires a global variable 'v' defining violation profiles
       for every possible word.
       
       Calculates probabilities given the weights and violations.
       
       Returns a 1d vector of predicted probabilities.
      '''
    
    #We use different violation matrices here so that the probs we output
    #match up with whatever ambiguity the model is currently dealing with,
    #but also so that the probability distribution we're defining is over 
    #all strings, including unambiguous ones.
    this_harmony = viols.dot(weights) #Uses local variable "viols"
    all_harmonies = v.dot(weights) #Always uses global variable v

    #Typical MaxEnt stuff:
    this_eharmony = np.exp(this_harmony) #e^H
    all_eharmonies = np.exp(all_harmonies)
    Z = sum(all_eharmonies)
    this_prob = this_eharmony/Z
    
    return this_prob
    
                                                                                                                                                                 
def grad_descent_update (weights, viols, td_probs, attention, this_datum, eta=.05):
    '''Takes a 1d vector of weights, a 2d vector of violations, a 1d vector of
       training data probabilities, an attention probaility parameter, 
       a datum from the training data, and a learning rate (default=.05)
       as input. Requires the global variable 'features' that maps valued 
       features to the segments they define. Also requires the global variable
       'AMBIGUITY_WORD_DICT' that maps a list of ignored features and data 
       indeces to the indeces of ambiguous data.
    
       Peforms standard online gradient descent if attention==1. Otherwise, it
       performs gradient descent with Probabilistic Feature Attention. 
       
       Output is the vector of new weights.'''
         
    #Create a new matrix of violation profiles, based on which features 
    #are *not* attended to this forward pass (can only handle binary features):
    if ATTENTION_PROB == 1.0:
        ambig_viols = viols
    else:
        ignored_features = ""
        for feature in sorted(features.keys()):
            if feature[0] == "-":
                continue
            if uniform(0,1)>=attention: 
                ignored_features += feature[1:]
        ambig_word = AMBIGUITY_WORD_DICT[this_datum][ignored_features]
        ambig_viols = v[ambig_word]
              
    #Forward pass
    le_probs = get_predicted_probs (weights, ambig_viols)
    
    #Backward pass:
    TD = ambig_viols.T.dot(td_probs) #Violations present in the training data
    LE = ambig_viols.T.dot(le_probs) #Violations expected by the learner
    gradients = (TD - LE)
    updates = gradients * eta
    new_weights = weights + updates

    return new_weights
################################################               

######PROCESS INPUT FILES########################
#Feature file (can only handle binary features):
feature_file = open(WD+PATTERN+"_Features.csv", "r")
feature_list = feature_file.readline().rstrip().split(",")[1:]
features = { #Dictionary mapping valued features to their segments.
                valued_feature:[] for valued_feature in \
                ["+"+feat for feat in feature_list]\
               +["-"+feat for feat in feature_list]
            }
SIGMA = [] #List of possible segments
for row in feature_file.readlines():
    columns = row.rstrip().split(",")
    this_segment = columns[0]
    for feature_index, value in enumerate(columns[1:]):
        features[value+feature_list[feature_index]].append(this_segment)
        if this_segment not in SIGMA:
            SIGMA.append(this_segment)

#Training data file:
training_file = open(WD+PATTERN+"_TD.csv", "r")
languages = {}
max_word_length = 0
nonce_words = []
for language in training_file.readlines():
    columns = language.rstrip().split(",")
    lang_name = columns[0]
    if lang_name == "Nonce":
        nonce_words = columns[1:]
        continue
    lang_data = columns[1:]
    languages[lang_name] = lang_data
    for word in lang_data:
        if len(word) > max_word_length:
            max_word_length = len(word)
nonce_words = [nw for nw in nonce_words if nw != ""]
               
#Ambiguous segment file:
if ATTENTION_PROB < 1.0:
    ambig_seg_file = open(WD+PATTERN+"_AmSegs.csv")
    unambig2ambig = {}
    ambig2unambig = {}
    firstRow = True
    for row in ambig_seg_file.readlines(): 
        #Process row:       
        columns = row.rstrip().split(",")
        if firstRow:
            firstRow = False
            unambig_segs = columns[1:]
        ambFeats = "".join(sorted(columns[0].split("+")))
        
        #Get unambig->ambig mappings:
        unambig2ambig[ambFeats] = {}
        for seg_index, unambig_seg in enumerate(unambig_segs):
            ambig_seg = columns[seg_index+1]
            if ambig_seg not in SIGMA:
                SIGMA.append(ambig_seg)
            unambig2ambig[ambFeats][unambig_seg] = ambig_seg
            
        #Get ambig->unambig mappings:
        for col in range(1,len(columns)):
            try:
                ambig2unambig[columns[col]].append(unambig_segs[col-1])   
            except:
                ambig2unambig[columns[col]] = [unambig_segs[col-1]]                 
                          
#Find all possible words (this is the full training data set):
if max_word_length == 1:
    SIGMA_STAR = SIGMA
else:    
    SIGMA_STAR = sorted(["".join(ngram) for ngram in \
                        product(SIGMA, repeat=max_word_length)])
word_lookup = lambda word : SIGMA_STAR.index(word) 

#Finally, we make a dictionary that maps all the possible words to their 
#ambiguous counterparts, based on what features are being ignored:
if ATTENTION_PROB < 1.0:
    AMBIGUITY_WORD_DICT = {word_index:{} for word_index in range(len(SIGMA_STAR))}       
    for ambFeats in unambig2ambig.keys():
        for word in SIGMA_STAR:
            newWord = ""
            for seg in word:
                try:
                    newSeg = unambig2ambig[ambFeats][seg]
                except:
                    newSeg = seg
                newWord += newSeg
            AMBIGUITY_WORD_DICT[word_lookup(word)][ambFeats] = word_lookup(newWord)

    #Add in ambiguous segments to the feature vectors: 
    for feat in features.keys():
        additions = []
        for orig_seg in features[feat]:
            for amb_seg in ambig2unambig.keys():
                if orig_seg in ambig2unambig[amb_seg]:
                    additions.append(amb_seg)
        features[feat] += list(set(additions)) 
################################################  
            
#####CONSTRAINTS################################
CON_names = [] 

##Unigram constraints:
#Constraints that refer to more than one feature:
for feat_num in range(2,int(round((len(features.keys())/2)+1))):
    these_combos = list(combinations(features.keys(), r=feat_num))
    for bundle in these_combos:
        bundle_string = "".join(bundle)
        bad_bundle = False #bad bundles have the same feature twice
        for feature in [feat[1:] for feat in features.keys() if feat[0] == "-"]:
            if re.search(feature+".*"+feature, bundle_string):
                bad_bundle = True
        if bad_bundle:
            continue
            
        #Sort the features in the bundle and save it:
        new_bundle = sorted(list(bundle))
        CON_names.append(new_bundle)
        
#Constraints that refer to only 1 feature:
single_feature = features.keys() 
CON_names += [[f] for f in single_feature]
    
##N-gram constraints, where N > 1:
for gram_size in range(2, max_word_length+1):
    #Find all grams of size n:
    ngrams = list(combinations_with_replacement(CON_names, gram_size))
    
    #Find all permutations of these grams:
    for gram in ngrams:
        CON_names += list(multiset_permutations(gram))

##Convert each constraint bundle to a regex:
CON_regexes = ["."] #Bias constraint, per GMECCS and Configural Cue Model
regexes_soFar = CON_regexes[:]
new_names = ["Bias Constraint"]
special_chars = ["@", "$", "^", "*", "(", ")", "+", "-", "%", "&",
                    "|", ".", "?", "!", "=", "[", "]", "{", "}", "#"]
for c in CON_names:
    if isinstance(c[0],str): #Is this a unigram constraint?
        seg_lists = [features[f] for f in c]
        seg_intersection = list(set(seg_lists[0]).intersection(*seg_lists))
    
        #No need to keep constraints that don't refer to any segments:
        if len(seg_intersection) == 0:
            continue
        
        #Escape any special characters:
        for index, seg in enumerate(seg_intersection):
            if seg in special_chars:
                seg_intersection[index] = "\\"+seg
        
        #No need for redundant constraints, either:        
        this_regex = "("+"|".join(seg_intersection)+")"
        if this_regex in regexes_soFar:
            continue
        
        #Add regex to the master list:
        CON_regexes.append(this_regex)
        regexes_soFar.append(this_regex)
        new_names.append(c)
    else: #Is this an N>1-gram constraint?
        empty_bundle = False
        this_regex = ""
        for gram in c:
            seg_lists = [features[f] for f in gram]
            seg_intersection = list(set(seg_lists[0]).intersection(*seg_lists))
            
            #No need to keep constraints with empty bundles:
            if len(seg_intersection) == 0:
                empty_bundle = True
                break
                
            #If it's not empty, convert the bundle to a regex:                   
            this_regex += "("+"|".join(seg_intersection)+")"
        
        #Skip to the next constraint if this one had an empty bundle:
        if empty_bundle:
                continue
        
        #Also skip redundant constraints:
        if this_regex in regexes_soFar:
            continue
        
        #Add regex to the master list:
        CON_regexes.append(this_regex)
        regexes_soFar.append(this_regex)
        new_names.append(c)
CON_names = new_names                  
                                   
##Constraint weights (initialized at zero)                
w = np.array([0.0 for c in CON_names])

##Violation profiles:                    
print ("Finding violation profiles...")
try:
    v_file = open(WD+PATTERN+"_Violations (attention="+str(ATTENTION_PROB)+").csv", "r")
    print ("...from file.")
    v = []
    for datum in v_file.readlines():
        v.append([float(d) for d in datum.split(",")])
    v = np.array(v)
    v_file.close()
except:   
    print ("...from scratch. Might take a while (only have to do it once per attention prob, though).") 
    v = np.array([[-1.0 * len(re.findall(c, word)) for c in CON_regexes]\
                for word in SIGMA_STAR]) 
    np.savetxt(WD+PATTERN+"_Violations (attention="+str(ATTENTION_PROB)+").csv", v, delimiter=",", newline="\n")
################################################ 

######SIMULATIONS###############################
last_w = []
output_file = open(WD+PATTERN+"_output (attention="+str(ATTENTION_PROB)+").csv", "w")  
output_file.write("Language,Rep,Epoch,Word,TD_Prob,LE_Prob\n")
for lang in languages.keys():
    #Set up learning data probabilities for each language:
    grammatical_words = languages[lang]
    grammatical_word_indeces = [word_lookup(word)\
                                    for word in grammatical_words if word!=""]

    test_word_indeces = list(set(grammatical_word_indeces \
                       +[word_lookup(word) for word in nonce_words]))
    N = float(len(grammatical_words))
    p = np.array([0.0 for word in SIGMA_STAR])
    p[grammatical_word_indeces] = 1.0/N
    shuffled_probs = list(enumerate(p))
        
    for rep in range(REPS):
        for epoch in range(EPOCHS): 
            shuffle(shuffled_probs)
            for iter_index, iter_prob in shuffled_probs:
                w = grad_descent_update(
                                            w, #Current weights
                                            v[iter_index], #Datum violations
                                            iter_prob, #Datum TD probability
                                            ATTENTION_PROB, #p(attend to feat)
                                            iter_index,#Datum index
                                            ETA #Learning rate
                                        )
                
            if epoch % 10 == 0:
				#Sometimes my version of python crashes on this print statement...
				#...fixed this with a try-except:
                try:
                    print ("Epoch: " + str(epoch), ", Pattern: " + str(lang), ", Rep: " + str(rep))
                except:
                    print ("Epoch: " + str(epoch), ", Pattern: " + str(lang), ", Rep: " + str(rep))
                    
            #Get our model's current estimation of the unambiguous data:
            current_probs = get_predicted_probs(w, v)
            
            #Record the info that we care about, depending on TEST_FOR's value
            for datum_index in test_word_indeces:
                output_row = []
                output_row.append(lang)
                output_row.append(str(rep))
                output_row.append(str(epoch))
                output_row.append(SIGMA_STAR[datum_index])
                output_row.append(str(p[datum_index]))
                output_row.append(str(current_probs[datum_index]))
                output_file.write(",".join(output_row)+"\n")
        
        #Reset constraint weights:
        last_w = np.copy(w)
        w *= 0.0       
################################################

#####Close files################################
output_file.close() 
training_file.close()
feature_file.close()
try:
    ambig_seg_file.close()
except:
    pass