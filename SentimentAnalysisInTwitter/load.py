import os
import numpy as np
import pickle
import random
import operator
import time
from Crypto.Util import Counter
from collections import Counter
from sklearn.cross_validation import train_test_split
from comtypes.npsupport import numpy

'''
Note: No obligation to use this code, though you may if you like.  Skeleton code is just a hint for people who are not familiar with text processing in python. 
It is not necessary to follow. 
'''
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale
    
    
def scalar_mult(d1, scale):
    temp = Counter({})
    if(scale == 0):
        return temp
    for f, v in d1.items():
        temp[f] = v * scale
    return temp

def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings. 
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on', 
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(None, symbols).strip(), lines)
    words = filter(None, words)
    return words
	
###############################################
######## YOUR CODE STARTS FROM HERE. ##########
###############################################

def shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = "D:\Academics\Courses\MachineLearning\hw3-sentiment\data\pos"
    neg_path = "D:\Academics\Courses\MachineLearning\hw3-sentiment\data\\neg"
	
    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)
	
    all_review = pos_review + neg_review
    #random.shuffle(all_review)
    return all_review
	
    
#Converts reviews to sparse representation.
def convert_to_dict_representation(reviews):
    all_reviews = []
    for review in reviews:
        all_reviews.append(Counter(review))
    return all_reviews
   
def convert_to_dict_representation_features(reviews):
    all_reviews=[]
    for review in reviews:
        dict = Counter({})
        dict[review[0]] = dict.get(review[0], 1)
        for i in range(1,len(review)):
            dict[review[i]] = dict.get(review[i], 1)
            dict[str(review[i]) + str(review[i-1])] = dict.get(str(review[i]) + str(review[i-1]), 1)
        all_reviews.append(dict)
    return all_reviews
    
#Normal pegasos algorithm.
def pegasos(X_train, alpha, lambda_reg, num_epochs):
    num_instances = len(X_train)
    theta = {}
    step_size = alpha
    t = 2;
    for i in range(0, num_epochs):
        for j in range(0, num_instances):
            step_size = 1.0/(t * lambda_reg)
            y = 1
            if(X_train[j].get(-1)>0):
                y = -1
            X = X_train[j]
            if(y * dotProduct(theta, X) < 1):
                theta = scalar_mult(theta, 1- step_size * lambda_reg)
                increment(theta,1, scalar_mult(X, step_size * y))
            else:
                theta = scalar_mult(theta,1- step_size * lambda_reg)
            t= t+1
    return theta

#Faster pegasos algorithm.
def faster_pegasos(X_train, X_test, alpha, lambda_reg, num_epochs):
    num_instances = len(X_train)
    wtheta = {}
    step_size = alpha
    t = 2;
    s = 1
    loss = numpy.zeros(num_instances*num_epochs)
    for i in range(0, num_epochs):
        for j in range(0, num_instances):
            step_size = 1.0/(t * lambda_reg)
            y = 1
            if(X_train[j].get(-1)>0):
                y = -1
            X = X_train[j] #Need to check this. Included output term as well.
            s = (1-step_size*lambda_reg)*s
            if(y * dotProduct(wtheta, X) < 1):
                increment(wtheta,1, scalar_mult(X, step_size * y/s))
            t=t+1
    theta = scalar_mult(wtheta, s)
    return theta    


#Calculates the percentage of correct predictions.
def calculate_percent_error(theta, X_test):
    num_instances = len(X_test)
    correct_prediction = 0.0
    for i in range(0, num_instances):
        y = 1
        if(X_test[i].get(-1)>0):
            y = -1
        if(y * dotProduct(theta, X_test[i]) > 0):
            correct_prediction = correct_prediction + 1
    return (1 - correct_prediction*1.0/num_instances)*100
    
#This is used to get product of weights and corresponding counter values of different words.
def feature_contribution(theta, sample_input):
    contribution = Counter({})
    y = 1
    if(sample_input.get(-1)>0):
        y = -1
    for f,v in sample_input.items():
        contribution[f] = theta.get(f, 0) * v
    contribution = sorted(contribution.items(), key=operator.itemgetter(1))
    return contribution,y

'''
Now you have read all the files into list 'review' and it has been shuffled.
Save your shuffled result by pickle.
*Pickle is a useful module to serialize a python object structure. 
*Check it out. https://wiki.python.org/moin/UsingPickle
'''
 
def main():
    np.random.seed(10)
    
    print('Reading data and converting it to sparse representation.')
    all_rev = shuffle_data()
    all_reviews = convert_to_dict_representation(all_rev)
    
    print('Splitting data to training and test set')
    X_train, X_test = train_test_split(all_reviews, test_size=500, random_state=10)
    
    print('Both pegasos and modified pegasos will be run and note that their corresponding values are equal.Just running 2 epochs')
    print('Running normal pegasos algorithm.')
    starttime = time.time()
    t1 = pegasos(X_train, 0.1, 1, 2)
    print("Time taken for Pegasos:")
    print(time.time() - starttime)
    
    
    print('Running modified pegasos algorithm')
    starttime = time.time()
    t2 = faster_pegasos(X_train, X_test, 0.1, 1, 2)
    print("Time taken for Modified Pegasos:")
    print(time.time() - starttime)
    
    
    #pickle.dump(t2, open("D:\Academics\Courses\MachineLearning\hw3-sentiment\\trained.p", "wb"))
    #t3 = pickle.load(open( "D:\Academics\Courses\MachineLearning\hw3-sentiment\\trained.p", "rb"))
    
    print('Trying different lamda and their errors')
    for j in range(-3,3):
        Lambda = 10**j;
        theta = faster_pegasos(X_train, X_test, 0.1, Lambda, 10)
        print('Percentage error with Lambda ' + str(Lambda) + '=' + str(calculate_percent_error(theta, X_test))) 
    
    
    print("Trying new feature representation.")
    all_reviews_new_features = convert_to_dict_representation_features(all_rev)
    X_train, X_test = train_test_split(all_reviews_new_features, test_size=500, random_state=10)
    print('Trying different lamda for new features added representation and their errors')
    for j in range(-3,3):
        Lambda = 10**j;
        theta = faster_pegasos(X_train, X_test, 0.1, Lambda, 10)
        print('Percentage error with Lambda ' + str(Lambda) + '=' + str(calculate_percent_error(theta, X_test))) 
    
    
if __name__ == "__main__":
    main()
