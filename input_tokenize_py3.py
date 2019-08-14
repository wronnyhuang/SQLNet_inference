from nltk.tokenize.stanford import StanfordTokenizer
import os
os.environ['CLASSPATH'] = '/root/datasets/stanford-postagger-2018-10-16/'
with open('input_question.txt', 'r') as f:
    raw_q = f.read()
stanford = StanfordTokenizer()
print(stanford.tokenize(raw_q))
