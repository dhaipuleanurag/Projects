import os
import numpy as np
import math
import gensim
from nltk.corpus import stopwords


def LCS(f_sentence, s_sentence):
    len_f = len(f_sentence)
    len_s = len(s_sentence)
    res = np.zeros((len_f+1, len_s+1))
    for i in range(1,len_f+1):
        for j in range(1, len_s+1):
            if f_sentence[i-1] == s_sentence[j-1]:
                res[i][j] = res[i-1][j-1] + 1
            else:
                res[i][j] = max(res[i-1][j], res[i][j-1])
    return (res[len_f][len_s] * res[len_f][len_s])/(len_f * len_s)

def MCLCS1(f_sentence, s_sentence):
    len_f = len(f_sentence)
    len_s = len(s_sentence)
    if len_f < len_s:
        small_sent = f_sentence
        long_sent = s_sentence
    else:
        small_sent = s_sentence
        long_sent = f_sentence


    while len(small_sent) > 0:
        if small_sent in long_sent:
            return len(small_sent)
        else:
            small_sent = small_sent[:-1]
    return len(small_sent)

# def MCLCS1_wrong(f_sentence, s_sentence):
#     min_len = min(len(f_sentence), len(s_sentence))
#     mclcs = 0
#     for i in range(min_len):
#         if f_sentence[i] == s_sentence[i]:
#             mclcs = mclcs + 1
#         else:
#             return mclcs
#     return min_len


def MCLCSN(f_sentence, s_sentence):
    len_f = len(f_sentence)
    len_s = len(s_sentence)
    min_len = min(len_f, len_s)
    if len_f < len_s:
        small_sent = f_sentence
        long_sent = s_sentence
    else:
        small_sent = s_sentence
        long_sent = f_sentence
    mclcsn = 0
    for len_gram in range(1, min_len+1):
        for i in range(min_len-len_gram+1):
            gram = small_sent[i:i+len_gram]
            mclcsn = max(mclcsn, MCLCS1(gram, long_sent))
    return mclcsn


def get_word_similarity_model(wsj_corpus_path):
    all_sentences = []
    sentence = []
    sentences = open(wsj_corpus_path)
    for line in sentences:
        if line.split('\t')[0] == '\n':
            all_sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line.split('\t')[0].lower())
    model = gensim.models.Word2Vec(all_sentences, min_count=5)
    return model

def remove_stop_words(sentence, stop_words):
    sentence = clean_up_and_lower(sentence)
    sentence = ' '.join([word for word in sentence.split() if word not in stop_words])
    return sentence

def clean_up_and_lower(sentence):
    to_be_cleaned_list = ',./()\'""'
    new_sentence = ''
    for word in sentence.split(' '):
        for i in range(len(to_be_cleaned_list)):
            word = word.replace(to_be_cleaned_list[i], '')
        new_sentence = new_sentence + word.lower() + ' '
    return new_sentence

def remove_same_words(sentence1, sentence2):
    sentence1_trim = ' '.join([word for word in sentence1.split() if word not in sentence2])
    sentence2_trim = ' '.join([word for word in sentence2.split() if word not in sentence1])
    return sentence1_trim, sentence2_trim

def buildm1(sentence1, sentence2, w1, w2, w3):
    sen1_words = sentence1.split(' ')
    sen1_len = len(sen1_words)
    sen2_words = sentence2.split(' ')
    sen2_len = len(sen2_words)
    res = np.zeros((sen1_len, sen2_len))
    for i in range(sen1_len):
        for j in range(sen2_len):
            res[i][j] = w1*LCS(sen1_words[i], sen2_words[j]) + w2*MCLCS1(sen1_words[i], sen2_words[j]) + w3*MCLCSN(sen1_words[i], sen2_words[j])
    return res

def buildm2(sentence1, sentence2, model):
    sen1_words = sentence1.split(' ')
    sen1_len = len(sen1_words)
    sen2_words = sentence2.split(' ')
    sen2_len = len(sen2_words)
    res = np.zeros((sen1_len, sen2_len))
    for i in range(sen1_len):
        for j in range(sen2_len):
            sim_value = 0
            try:
                sim_value = model.similarity(sen1_words[i].lower(), sen2_words[j].lower())
            except KeyError:
                sim_value = 0.5
            if sim_value > 0.7:
                res[i][j] = 1
            else:
                res[i][j] = 0
                #res[i][j] = sim_value/0.7
    return res

def get_max_value_row_column(arr):
    (row_len,column_len) = arr.shape
    max = 0
    max_row = 0
    max_column = 0
    for i in range(row_len):
        for j in range(column_len):
            if(arr[i][j] > max):
                max = arr[i][j]
                max_row = i
                max_column = j
    return max, max_row, max_column


def get_max_valued_list(arr):
    max_values = []
    while(True):
        max_val, max_row, max_column = get_max_value_row_column(arr)
        if max == 0:
            return max_values
        max_values.append(max_val)
        try:
            arr = np.delete(arr, max_row, 0)
            arr = np.delete(arr, max_column, 1)
        except:
            return max_values
def similarity_score(max_valued_list, delta, m, n):
    score = (sum(max_valued_list) + delta) * (m + n)
    score = score / (2 * m * n)
    return score

def main():
    # lcs = LCS('SPANKING', 'AMPUTATION')
    # mclcs = MCLCS1('apansdas', 'asdasdaapansasdasda')
    # mclcsn = MCLCSN('asdasabcdefghijklmnopq', 'abcdefgh')
    #model = get_word_similarity_model('D:\Academics\Courses\NLP\hw4_programming\WSJ_POS_CORPUS_FOR_STUDENTS\WSJ_02-21.pos')
    #model.save('D:\Academics\Courses\NLP\project\model.l')
    model = gensim.models.Word2Vec.load('D:\Academics\Courses\NLP\project\model.l')


    stop_words = stopwords.words('english')
    #sentence1 = 'Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.'
    #sentence2 = 'Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.'
    msr_corpus_training = 'C:\MSRParaphraseCorpus\msr_paraphrase_train.txt'
    lines = open(msr_corpus_training)
    count = 0
    total = 0
    results = np.zeros(300)
    cut = 0.7
    so = 0
    while cut < 1.3:
        count = 0
        total = 0
        lines = open(msr_corpus_training)
        for line in lines:
            line_split = line.split('\t')
            sentence1 = line_split[3]
            sentence2 = line_split[4]
            try:
                y = int(line_split[0])
            except:
                continue
            #sentence1 = 'That compared with $35.18 million, or 24 cents per share, in the year-ago period.'
            #sentence2 = 'Earnings were affected by a non-recurring $8 million tax benefit in the year-ago period.'
            sentence1 = remove_stop_words(sentence1, stop_words)
            sentence2 = remove_stop_words(sentence2, stop_words)
            m = len(sentence1.split(' '))
            n = len(sentence2.split(' '))
            sentence1, sentence2 = remove_same_words(sentence1, sentence2)
            m_delta = len(sentence1.split(' '))
            delta = 1.5 * (m - m_delta)
            joint_matrix = 0.5 * buildm1(sentence1, sentence2, 0.3, 0.4, 0.4) + 0.5 * buildm2(sentence1, sentence2, model)
            max_valued_list = get_max_valued_list(joint_matrix)
            pred = 0
            if similarity_score(max_valued_list, delta, m, n) > cut:
                pred = 1
            if pred == y:
                count = count + 1
            total = total + 1

        #close(msr_corpus_training)
        results[so] = count

        so = so + 1
        cut = cut + 0.01
    x = 5
if __name__ == "__main__":
    main()