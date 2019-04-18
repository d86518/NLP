#檢查有沒有install成功
#import sys, os
#site_path = ''
#for path in sys.path:
#    if 'site-packages' in path.split('/')[-1]:
#        print(path)
#        site_path = path
## search to see if gensim in installed packages
#if len(site_path) > 0:
#    if not 'gensim' in os.listdir(site_path):
#        print('package not found')
#    else:
#        print('gensim installed')


import logging

#先做 pip install gensim
#如無法則嘗試 pip install --upgrade gensim
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import sys, os
corpus = api.load('text8')  # download the corpus and return it opened as an iterable => 大約需要10min(31MB)

#用預設參數直接訓練 => 約 60萬 words、5個epoch滿久的
model = Word2Vec(corpus)  
# train a model from the corpus
model.most_similar("car")

print("King AND Queen，LIKE Man AND...")
res = model.most_similar(['king','queen'],['man'],topn = 10)
for item in res:
    print(item[0]+","+str(item[1]))

print("Calculate Cosine Similarity")
res = model.similarity('king','prince')
print(res)




#先載好text8可用此
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)
#sentences = word2vec.LineSentence("text8")
#model = word2vec.Word2Vec(sentences,size=300)
#model.save("word2vec.model")

#設定model參數
#gensim.models.word2vec.Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count =5, max_vocab_size =None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs =0, negative=5, cbow_meancbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)

#res = model.most_similar('love',topn = 10)
#for item in res:
#    print(item[0]+ "," + str(item[1]))
    