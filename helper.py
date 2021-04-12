import pandas as pd 
import numpy as np
import xml.etree.ElementTree as ET
import io
from constants import max_aspect_len, max_context_len, trainFiles,testFiles,all_data,embedding_file_name, embedding_dim
import pickle 
import ast
from constants import *

def load_word_embeddings():
    word_id = get_word_id()

    embedMatrix = np.random.uniform(-0.01, 0.01, [len(word_id), embedding_dim])
    with open(embedding_file_name, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            word = content[0]
            embedding = content[1:]
            if word in word_id:
                embedMatrix[word_id[content[0]]] = np.array(list(map(float, embedding)))
    embedMatrix[0, :] = 0
    return embedMatrix

def get_word_id():
    df = pd.read_csv(all_data)
    all_tokens = df.tokens.to_list()
    all_tokens.extend(df.aspect.to_list())
    word_id={"$":0}
    for sent in all_tokens:
        for token in sent:
            if ' ' not in token and '\n' not in token and token not in word_id:
                word_id[token]=len(word_id)
    return word_id

def get_final_data(type):
    word_id = get_word_id()
    df = pd.read_csv(f'{type}.csv')
    finalaspects, finalcontexts, finallabels, finalaspect_lens, finalcontext_lens = list(), list(), list(), list(), list()
    allaspects = df.aspect.to_list()
    allcontexts = df.tokens.to_list()
    labels = df.label.to_list()
    for ind,row in df.iterrows():
        context = ast.literal_eval(row['tokens'])
        aspects = ast.literal_eval(row['aspect'])
        label = int(row['label'])
        outcontext = []
        outaspect = []

        for token in context:
            if token in word_id:
                outcontext.append(word_id[token])

        for token in aspects:
            if token in word_id:
                outaspect.append(word_id[token])
        finalaspects.append(outaspect + [0] * (max_aspect_len - len(outaspect)))
        finalcontexts.append(outcontext + [0] * (max_context_len - len(outcontext)))

        finallabels.append(label)
        finalaspect_lens.append(len(aspects))
        finalcontext_lens.append(len(context) - 1)
                

    aspects = np.asarray(finalaspects)
    contexts = np.asarray(finalcontexts)
    labels = np.asarray(finallabels)
    aspect_lens = np.asarray(finalaspect_lens)
    context_lens = np.asarray(finalcontext_lens)
    np.savez(f'dataset_{type}', aspects=aspects, contexts=contexts, labels=labels, aspect_lens=aspect_lens, context_lens=context_lens)
    return f'dataset_{type}'
    

def xml_to_csv():
    xmls = {'train_laptop' : ET.XML(open(trainFiles['Laptops'],'r').read()),
    'train_restaurant' : ET.XML(open(trainFiles['Restaurants'],'r').read()),

    'test_laptop' : ET.XML(open(testFiles['Laptops'],'r').read()),
    'test_restaurant' : ET.XML(open(testFiles['Restaurants'],'r').read())}

    # print(train_laptop[0][1][2].attrib)
    df = pd.DataFrame(columns=['text','aspect']) #aspect => {'term':(category,polarity)}
    # exit()
    for typee in ['train','test']:
        for topic in ['laptop','restaurant']:
            xml = xmls[f"{typee}_{topic}"]
            for sentence in xml:
                aspect = {}
                try:
                    text = sentence[0].text
                except:
                    # text is not present
                    continue
                i = 0
                while(True):
                    try:
                        aspect[str(sentence[1][i].attrib['term'])]=str(sentence[1][i].attrib['polarity'])
                        i+=1
                    except:
                        # print(sentence[1][i])
                        break
                df = df.append({'text':text,'aspect':aspect,'topic':topic,'type':typee}, ignore_index= True)

    print(df.shape)
    df.to_csv('data.csv')




# load_word_embeddings()