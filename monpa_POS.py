# -*- coding: utf-8 -*-
import sys 
import requests
from bs4 import BeautifulSoup
#pos 詞性標註

def get_monpa_result(sentence):
    res = requests.get('http://monpa.iis.sinica.edu.tw:9000/'+sentence)
    soup = BeautifulSoup(res.text,'html.parser')
    result = ''
    for item in soup.select('mark'):
        result += item.text + ' ' + item.attrs['data-entity'] + ' '
    return result


s = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽"
print(get_monpa_result(s))
