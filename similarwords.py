import random

from gensim.models import Word2Vec

f = open('wikisent2.txt', 'r', encoding='utf8')
text = f.read()
t_list = text.split('\n')


corpus = []

for cumle in t_list:
	corpus.append(cumle.split())
random.shuffle(corpus)
corpus = corpus[:int(len(corpus)/5)]

print("x")
model = Word2Vec(corpus, window=7, min_count=5, sg=1)
print("y")
# model.wv.most_similar('youtube')

model.save('word2vec.model')
print('Done')