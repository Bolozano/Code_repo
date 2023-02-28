import os
from multiprocessing import Pool, Lock,Manager
import torch
import spacy
import progressbar
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
import os

df1=pd.read_json(os.path.join('/GPFS/data/yutongmeng/VCR/vcg_annots/visualcomet','train_annots.json'))

def extract_entity(nlp,seqs):
    all_ents=[]
    for seq in seqs:
        doc = nlp(seq)
        input_ents=[]
        for i in range(len(doc)):
            token=doc[i]
            if token.pos_ =='VERB':
                input_ents.append(token.lemma_)
            if token.pos_=='NOUN':
                if i==0:
                    input_ents.append(token.lemma_)
                    continue
                if i>0 and  doc[i-1].pos_!='NOUN':
                    input_ents.append(token.lemma_)
                    continue
                elif token.pos_=='NOUN' and doc[i-1].pos_=='NOUN':
                    input_ents[-1]+='_'+token.lemma_
        all_ents.extend(input_ents)
    return all_ents

def gpt_conclude(keys):
    i=keys[0]
    count=keys[1]
    vcg_entities_train=keys[2]

    lock=keys[3]
    a=df1[i:i+1]
    data_dict=list(a.T.to_dict().values())[0]
    input_seq = ' '.join([data_dict['event'],data_dict['place']])
    input_ents = extract_entity(nlp,[input_seq])
    before_ents = extract_entity(nlp,data_dict['before'])
    intent_ents = extract_entity(nlp,data_dict['intent'])
    after_ents = extract_entity(nlp,data_dict['after'])
    lock.acquire()
    vcg_entities_train[i] = {'input':input_ents,'before':before_ents,'intent':intent_ents,'after':after_ents}
    count.value=count.value+1
    # if count.value==10:
    #     save_dict={}
    #     for temp in tqdm(vcg_entities_train.keys()):
    #         save_dict[temp]=vcg_entities_train[temp]
    #     np.save('/GPFS/data/yutongmeng/VCG_entities/vcg_entities_train.npy',vcg_entities_train)
    # if count.value>111794:
    #     with open('/GPFS/data/yutongmeng/VCG_entities/vcg_entities_train.txt','a+') as f:
    #         f.write(str(vcg_entities_train))

    lock.release()
    

# def init(nlp):
#     global lock 
#     global bar
#     global vcg_entities_train
#     global count
#     global vcg_entities_train
data_list = [i for i in range(df1.shape[0])]



with Manager() as manager:
    count=manager.Value('i',0)
    vcg_entities_train=manager.dict()
    lock = manager.Lock() 
    nlp = spacy.load("en_core_web_sm",disable=[])

    widgets = [ 'My progress1 :',' [', progressbar.Timer(), '] ',progressbar.Bar('&'), ' (', progressbar.ETA(), ') ']

    pool = Pool(20, initializer=None, initargs=(nlp,))
    _=list(tqdm(pool.imap(gpt_conclude, [(i,count,vcg_entities_train,lock) for i in data_list]),total=len(data_list) ))
    print('excecuted here   2')
    pool.close()
    pool.join()
    print('excecuted here   3')
    # with open('/GPFS/data/yutongmeng/VCG_entities/vcg_entities_train.txt','a+') as f:
    #     f.write(str(vcg_entities_train))
    save_dict={}
    for temp in tqdm(vcg_entities_train.keys()):
        save_dict[temp]=vcg_entities_train[temp]
    np.save('/GPFS/data/yutongmeng/VCG_entities/vcg_entities_train.npy',save_dict)
