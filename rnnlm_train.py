# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


# 設定超參數
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN的隱藏狀態向量的元素數
time_size = 5  # 展開RNN的大小
lr = 0.1
max_epoch = 100

# 載入學習資料
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000  # 縮小測試用的資料集corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]  # 輸入
ts = corpus[1:]  # 輸出（訓練標籤）

# 產生模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()

