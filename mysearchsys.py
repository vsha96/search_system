import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from pymystem3 import Mystem

import re

import os

import pickle

# сбор текстов из файлов fact_<i>.txt
# возвращает объединенный текст
def collect_text():
    reg_file = r'fact_\d.txt'

    file_list = []

    for file_name in sorted(os.listdir('text')):
        if re.match(reg_file, file_name):
            file_list.append('text/'+file_name)

    if file_list == []:
        print("there are no text files")

    all_text = ""

    for file_name in file_list:
        with open(file_name) as file:
            print("open", file_name)
            all_text = all_text + file.read()
    print("done!")

    return all_text

# функция разбиения текста на предложения
# возвращает список предложений
def sentence_list(text):
    proc_text = []

    for el in text.split('\n'):
        if el:
            sent_list = sent_tokenize(el, language="russian")
            for s in sent_list:
                proc_text.append(s)
    return proc_text

# функция лемматизации
# возвращает список токенов
reg_filter = r'[а-яА-Я]|[a-zA-Z]|\d'
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")
english_stopwords = stopwords.words("english")
def process_text(text):
    tokens = mystem.lemmatize(text)
    tokens = [token.lower() 
              for token in tokens 
              if token not in russian_stopwords 
              and token not in english_stopwords
              and token != " "
              and re.match(reg_filter, token) ]
    return tokens

# функция сборки термов
# возвращает кортеж: словарь термов и список предложений (списков) термов
def make_terms(proc_text):
    terms = {}
    term_text = []

    for sentence in proc_text:
        new_terms = process_text(sentence)
        term_text.append(new_terms)

        for t in new_terms:
            if t not in terms:
                terms[t] = {'df': None, 'idf': None}
                
    for t in terms:
        terms[t]['df'] = 0
        for doc in term_text:
            if t in doc:
                terms[t]['df'] += 1
    
    number_of_docs = len(proc_text)
    for t in terms:
        terms[t]['idf'] = np.log10(number_of_docs / terms[t]['df'])
    
    return terms, term_text

# функция нормализации вектора
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec/norm

# функция представления вектора в пространстве термов:
def tf_vec(doc, terms):
    words = list(terms.keys())
    doc_vec = np.zeros(len(terms.keys()))

    for t in doc:
        if t in words:
            i = words.index(t)
            doc_vec[i] += 1
        else:
            print("WARN: query word {" + t + "} is not in the collection")
        
    return doc_vec

# Функция взвешенного вектора документа
# выдает два ответа, соответственно двум разным способам учета tf
def weight_tf_idf_vec(doc, terms):
    words = terms.keys()
    w_vec = np.zeros(len(terms.keys()))
    
    doc_vec_tf1 = tf_vec(doc, terms)
    doc_vec_tf2 = tf_vec(doc, terms)
    #print("doc = ", doc, "\ndoc_vec after tf_vec():\n", doc_vec_tf1, "\n")
    
    i = 0
    for word in words:
        doc_vec_tf1[i] *= terms[word]['idf']
        doc_vec_tf2[i] = np.log(1+doc_vec_tf2[i]) * terms[word]['idf']
        i += 1
    
    return (doc_vec_tf1, doc_vec_tf2)

# Функция подсчета близости документов:
def proximity(vec1, vec2):
    cos = np.dot(normalize(vec1), normalize(vec2))
    return cos

# Функция сборки коллекции (сохраняет объект pickle):
def make_collection():
    print("building collection...")
    all_text = collect_text()
    proc_text = sentence_list(all_text)
    terms, term_text = make_terms(proc_text)
    proc_collection = list(zip(proc_text, [weight_tf_idf_vec(sent, terms) for sent in term_text]))
    if 'obj' not in os.listdir():
        os.mkdir('obj')
    with open('obj/core_collection.pkl','wb') as f:
        pickle.dump(proc_collection, f, pickle.HIGHEST_PROTOCOL)
    with open('obj/terms.pkl','wb') as f:
        pickle.dump(terms, f, pickle.HIGHEST_PROTOCOL)

# Функция поиска по коллекции
# выдает документы со значениями по мере их релевантности
def search(query):
    obj_name = 'core_collection.pkl'
    if 'obj' not in os.listdir() or obj_name not in os.listdir('obj'):
        make_collection()

    with open('obj/'+obj_name,'rb') as f:
        proc_collection = pickle.load(f)
    with open('obj/terms.pkl','rb') as f:
        terms = pickle.load(f)

    vec_q = normalize(tf_vec(process_text(query), terms))

    rel_docs1 = []
    rel_docs2 = []
    
    for i in range(len(proc_collection)):
        vec_d1 = proc_collection[i][1][0] # вектор по подсчету tf = count
        vec_d2 = proc_collection[i][1][1] # вектор по подсчету tf = log(1+count)
        
        rel_docs1.append((proc_collection[i][0], proximity(vec_q, vec_d1)))
        rel_docs2.append((proc_collection[i][0], proximity(vec_q, vec_d2)))
                         
    rel_docs1.sort(key=lambda x:x[1], reverse=True)
    rel_docs2.sort(key=lambda x:x[1], reverse=True)
        
    return rel_docs1, rel_docs2
