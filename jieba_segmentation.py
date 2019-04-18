#coding = utf-8
import jieba

jieba.set_dictionary('dict.txt.big')
jieba.load_userdict("userdict.txt")
s = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽"
print("Input:",s)
sresult = jieba.cut(s, cut_all=False)
print("Full Mode:")
print(' '.join(sresult))

for word in sresult:
    print(word.word, word.flag)