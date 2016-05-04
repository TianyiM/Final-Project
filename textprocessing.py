
#coding=utf-8

#Chinese word segmentation, postagger, sentence cutting and stopwords filtering function.


import xlrd
import jieba
import jieba.posseg
jieba.load_userdict('~') #Load user dictionary to increse segmentation accuracy




def get_excel_data(filepath, sheetnum, colnum, para):
    table = xlrd.open_workbook(filepath)
    sheet = table.sheets()[sheetnum-1]
    data = sheet.col_values(colnum-1)
    rnum = sheet.nrows
    if para == 'data':
        return data
    elif para == 'rnum':
        return rnum




def get_txt_data(filepath, para):
    if para == 'lines':
        txt1 = open(filepath, 'r')
        txt_tmp1 = txt_file1.readlines()
        txt_tmp2 = ''.join(txt_tmp1)
        txt_data1 = txt_tmp2.decode('utf8').split('\n')
        txt1.close()
        return txt1
    elif para == 'line':
        txt2 = open(filepath, 'r')
        txt_tmp = txt_file2.readline()
        txt_data2 = txt_tmp.decode('utf8')
        txt2.close()
        return txt2




def segmentation(sentence, para):
    if para == 'str':
        seg = jieba.cut(sentence)
        seg_result = ' '.join(seg_list)
        return seg_result
    elif para == 'list':
        seg2 = jieba.cut(sentence)
        seg_result2 = []
        for w in segt2:
            seg_result2.append(w)
        return seg_result2




def postagger(sentence, para):
    if para == 'list':
        pos_data1 = jieba.posseg.cut(sentence)
        pos_list = []
        for w in pos_data1:
             pos_list.append((w.word, w.flag)) 
        return pos_list
    elif para == 'str':
        pos_data2 = jieba.posseg.cut(sentence)
        pos_list2 = []
        for w2 in pos_data2:
            pos_list2.extend([w2.word.encode('utf8'), w2.flag])
        pos_str = ' '.join(pos_list2)
        return pos_str



def cut_sentences(words):
    start = 0
    i = 0 
    sents = []
    punt_list = ',.!?:;~，。！？：；～ '.decode('utf8') # Sentence cutting punctuations
    for word in words:
        if word in punt_list and token not in punt_list:
            sents.append(words[start:i+1])
            start = i+1
            i += 1
        else:
            i += 1
            token = list(words[start:i+2]).pop()
    if start < len(words):
        sents.append(words[start:])
    return sents


 
def seg_fil_excel(filepath, sheetnum, colnum):    
    review_data = []
    for cell in get_excel_data(filepath, sheetnum, colnum, 'data')[0:get_excel_data(filepath, sheetnum, colnum, 'rnum')]:
        review_data.append(segmentation(cell, 'list')) # Seg every reivew
        stopwords = get_txt_data('lines')
    seg_fil_result = []
    for review in review_data:
        fil = [word for word in review if word not in stopwords and word != ' ']
        seg_fil_result.append(fil)
        fil = []

    return seg_fil_result



def seg_fil_senti_excel(filepath, sheetnum, colnum):

    review_data = []
    for cell in get_excel_data(filepath, sheetnum, colnum, 'data')[0:get_excel_data(filepath, sheetnum, colnum, 'rnum')]:
        review_data.append(segmentation(cell, 'list')) 

    sentiment_stopwords = get_txt_data('lines')

    seg_fil_senti_result = []
    for review in review_data:
        fil = [word for word in review if word not in sentiment_stopwords and word != ' ']
    return seg_fil_senti_result