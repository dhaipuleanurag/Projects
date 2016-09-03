from collections import Counter
from statsmodels.sandbox.distributions.quantize import prob_bv_rectangle
import pickle
import numpy

t_sum = -1
def word_total_count(word_dict):
    global t_sum
    if(t_sum == -1):
        for key in word_dict.keys():
            t_sum = t_sum + sum(word_dict[key].values())
    return t_sum

def read_data(file):
    '''
    Read each file into a list of strings. 
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on', 
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    arc_dict = Counter({})
    word_dict = Counter({})
    f = open(file)
    lines = f.read().split('\n')
    previous = 'START_TAG'
    for line in lines:
        if(line == ''):
            previous = 'START_TAG'
            continue
        line_split = line.split('\t') 
        word_tag_dict = word_dict.get(line_split[1], Counter({}))
        word_tag_dict[line_split[0]] = word_tag_dict.get(line_split[0], 0) +1
        word_dict[line_split[1]] = word_tag_dict
        
        arc_tag_dict = arc_dict.get(previous, Counter({}))
        arc_tag_dict[line_split[1]] = arc_tag_dict.get(line_split[1], 0) +1
        arc_dict[previous] = arc_tag_dict
        
        previous = line_split[1]
    return arc_dict,word_dict
    
def get_transition_probability(previous_tag, current_tag, arc_dict, word_dict): 
    dict_previous_tagged = arc_dict.get(previous_tag, None)
    least_prob = 1.0/word_total_count(word_dict)
    if(dict_previous_tagged != None):
        return (dict_previous_tagged.get(current_tag,least_prob))*1.0/sum(dict_previous_tagged.values())
    else:
        return least_prob

def get_word_probability(word, tag, word_dict):   
    sum = 0
    for alltag in word_dict.keys():
        sum = sum + word_dict.get(alltag).get(word, 0)
    dict_tagged = word_dict.get(tag,None) 
    if(sum == 0):
        return 1.0/len(word_dict.keys())
    if(dict_tagged != None):
        return dict_tagged.get(word,0) * 1.0 /sum
    else:
        return 0.0

def get_viterbi_table(sentence, representation, arc_dict, word_dict):
    viterbi_table = []
    for tag in representation:
        if(tag == 'START_TAG'):
            viterbi_table.append([[1, None]])
        else:
            viterbi_table.append([[0, None]])
    
    for i in range(1, len(sentence) + 1):
        for j in range(0, len(representation)):
            max_value = [0, None]
            for index, previous_tag in enumerate(representation):
                prob = get_transition_probability(previous_tag, representation[j], arc_dict, word_dict) * viterbi_table[index][i-1][0]
                if(prob > max_value[0]):
                    max_value = [prob, previous_tag]
            viterbi_table[j].append([max_value[0]* get_word_probability(sentence[i-1], representation[j], word_dict),max_value[1]])
    return viterbi_table

def get_index_in_representation(tag, representation, sentence):
    for i in range(0, len(representation)):
        if(representation[i] == tag):
            return i
    print(sentence)
    print('There is some error')
    
def trace_back(column_number, previous_tag, viterbi_table, representation, sentence):
    if(column_number == 0):
        return []
    index = get_index_in_representation(previous_tag, representation, sentence)
    trace_path = trace_back(column_number-1, viterbi_table[index][column_number][1], viterbi_table, representation, sentence)
    trace_path.insert(len(trace_path), previous_tag)
    return trace_path
    
def get_best_tagged_sentence(sentence, viterbi_table, representation):
    
    #Think about what should happen if no best tag found.
    max_value = [0, None]
    columns = len(sentence) + 1
    rows = len(representation)
    for i in range(0, rows):
        if(viterbi_table[i][columns-1][0] > max_value[0]):
            max_value = [i , viterbi_table[i][columns-1][1]]
    
    tag_for_last = representation[max_value[0]]
    
    trace_path = trace_back(columns-2, max_value[1], viterbi_table, representation, sentence)
    trace_path.append(tag_for_last)
    return trace_path
    
def add_to_result_file(r, sentence, tagged):
    for i in range(0, len(sentence)):
        line = sentence[i] + "\t" + tagged[i] + "\n"
        r.write(line)
    r.write("\n")
    
def main():
    #print(score("D:\Academics\Courses\NLP\hw4_programming\WSJ_POS_CORPUS_FOR_STUDENTS\WSJ_24.pos", "D:\Academics\Courses\NLP\hw4_programming\WSJ_POS_CORPUS_FOR_STUDENTS\\result1.pos"))
    training = "D:\Academics\Courses\NLP\hw4_programming\WSJ_POS_CORPUS_FOR_STUDENTS\WSJ_02-21.pos"
    #arc_dict, word_dict = read_data(training)
    #sentence = ['Let' ,'us','take', 'a','break','and', 'talk', '.']
    #sentence = ['But', 'what', 'about', 'those', 'of', 'us', 'whose', 'views', 'are', 'not', 'predetermined', 'by', 'formula', 'or', 'ideology', '?']
        
    #pickle.dump(arc_dict, open("D:\Academics\Courses\NLP\\trained_arc.p", "wb"))
    #pickle.dump(word_dict, open("D:\Academics\Courses\NLP\\trained_word.p", "wb"))
    arc_dict = pickle.load(open( "D:\Academics\Courses\NLP\\trained_arc.p", "rb"))
    word_dict = pickle.load(open( "D:\Academics\Courses\NLP\\trained_word.p", "rb"))
    
    representation = arc_dict.keys()
    #viterbi_table = get_viterbi_table(sentence, representation, arc_dict, word_dict)
    #tagged = get_best_tagged_sentence(sentence, viterbi_table, representation)
    
    test = "D:\Academics\Courses\NLP\hw4_programming\WSJ_POS_CORPUS_FOR_STUDENTS\WSJ_24.words"
    result_file = "D:\Academics\Courses\NLP\hw4_programming\WSJ_POS_CORPUS_FOR_STUDENTS\\result1.pos" 
    f = open(test)
    re = open(result_file,'w')
    lines = f.read().split('\n')
    sentence = []
    for line in lines:
        if(line == ''):
            if(sentence == []):
                continue
            viterbi_table = get_viterbi_table(sentence, representation, arc_dict, word_dict)
            tagged = get_best_tagged_sentence(sentence, viterbi_table, representation)
            add_to_result_file(re, sentence, tagged)
            sentence = []
        else:
            sentence.append(line)
    f.close()
    re.close()
    
    print(score("D:\Academics\Courses\NLP\hw4_programming\WSJ_POS_CORPUS_FOR_STUDENTS\WSJ_24.pos", "D:\Academics\Courses\NLP\hw4_programming\WSJ_POS_CORPUS_FOR_STUDENTS\\result1.pos"))
    
def score (keyFileName, responseFileName):
    keyFile = open(keyFileName, 'r')
    key = keyFile.readlines()
    responseFile = open(responseFileName, 'r')
    response = responseFile.readlines()
    if len(key) != len(response):
        print(len(key))
        print(len(response))
        print "length mismatch between key and submitted file"
        exit()
    correct = 0
    incorrect = 0
    for i in range(len(key)):
        key[i] = key[i].rstrip('\n')
        response[i] = response[i].rstrip('\n')
        if key[i] == "":
            if response[i] == "":
                continue
            else:
                print "sentence break expected at line " + str(i)
                exit()
        keyFields = key[i].split('\t')
        if len(keyFields) != 2:
            print "format error in key at line " + str(i) + ":" + key[i]
            exit()
        keyToken = keyFields[0]
        keyPos = keyFields[1]
        responseFields = response[i].split('\t')
        if len(responseFields) != 2:
            print "format error at line " + str(i)
            exit()
        responseToken = responseFields[0]
        responsePos = responseFields[1]
        if responseToken != keyToken:
            print "token mismatch at line " + str(i)
            exit()
        if responsePos == keyPos:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    print str(correct) + " out of " + str(correct + incorrect) + " tags correct"
    accuracy = 100.0 * correct / (correct + incorrect)
    print "  accuracy: %f" % accuracy
            
if __name__ == "__main__":
    main()
    