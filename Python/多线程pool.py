import os
from multiprocessing import Pool, Lock
import torch
import numpy as np
import time
import pandas as pd
import torch
from tqdm import tqdm
import os
import openai
from progress.bar import Bar
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool

df1=pd.read_json(os.path.join('/Users/yutong/Desktop','val_annots.json'))

api_keys=[

]


def gpt_conclude(i):
    # print(i)
    global count
    # print('enter_thread')
    lock.acquire()
    count=count+1
    # print(count)
    
    openai.api_key = api_keys[count%len(api_keys)]
    lock.release()
    a=df1[i:i+1]
    data_dict=list(a.T.to_dict().values())[0]
    prompt="Given {Event: [\"2 was doing his homework\"]; Place: [\"school\"]}; we can guess {intent,before,after} as {Intent: [\"read his notes over\",\"review his text and study\"]; Before: [\"sat down at the table\",\"opened his notebook\",\"put his books on the table\",\"sit down with his friends\",\"see 1 waving the paper\"], After: [\"sigh in frustration\",\"find somewhere quieter to sit\",\"pick up a paper\",\"read the paper he's touching\"]}. \
Given {Event: [\"3 is sitting at the end of the table looking concerned\"], Place: [\"in a restaurant\"]}; we can guess {intent,before,after} as {Intent: [\"listen to 8's story\",\"wait for the check\",\"get another drink\"], Before: [\"invite everyone over for dinner\",\"sit down at the table\",\"come to the dinner\",\"sit down\",\"start eating\"], After: [\"finish eating dinner\",\"stand up and clear the table\",\"listen to 8's story\",\"react to the story\",\"finish his meal\"]}. \
Given {Event: [\"1 is sitting in a suit and waiting for someone\"], Place: [\"in a cafe\"]}; we can guess {intent,before,after} as {Intent: [\"have coffee with his girlfriend\",\"go on a blind date\",\"look nice\"], Before: [\"walk into the bar\",\"walk up to a barstool\",\"go to the bar\",\"set a date to meet them\"], After: [\"speak to the server\",\"place an order\",\"meet someone there\",\"leave the bar\"]}. \
Given {Event: [\"1 is sitting at a table at a busy night club\"], Place: [\"in a nightclub\"]}; we can guess {intent,before,after} as {Intent: [\"to go out with drinks with her friends\",\"share a date with a romantic love interest\"], Before: [\"style her hair\",\"put on makeup\",\"go to the night club\",\"be seated by a waiter\",\"be asked out on a date\",\"put on makeup\"], After: [\"talk to her date\",\"order a drink\",\"order a drink\",\"talk to her date\"]}. \
Given {Event: [\"a group of soldiers are on horses\"], Place: [\"a battlefield\"]}; we can guess {intent,before,after} as {Intent: [\"ride into battle\",\"kill the enemy\"], Before: [\"put on their armor\",\"mount their horses\",\"ride towards the enemy army\",\"ride to the battle field\",\"see the enemy\"], After: [\"listen to their commander speaking\",\"follow their commander's lead\",\"charge at the enemy\",\"yell a battle cry\"]}. "
    prompt_=prompt+"Please fill the following sentence according to 5 examples given."+  ' Given {Event: '+f'[\"{str(data_dict["event"])}\"], Place: [\"{str(data_dict["place"])}\"]'+'}; we can guess {intent,before,after} as '

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt_,
    temperature=1.0,
    max_tokens=512,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    # print('get response!!!!!!!!!!')
    time.sleep(1)
    
    lock.acquire()
    # print('to here lock')
    bar.next()
    gpt_conclusion[i]=response.choices[0].text.replace('\n','')
    # print(gpt_conclusion[i])
    # print('to here count')
    #global count  ##不声明会卡住
    
    # print('to here count2')
    # print(count)

    if count%20==0:
        np.save('/Users/yutong/Desktop/gpt3_icl_val.npy',gpt_conclusion)
    lock.release()
    # print('lock release')

print('here reading history result')
if os.path.exists('/Users/yutong/Desktop/gpt3_icl_val.npy'):
    gpt_conclusion=np.load('/Users/yutong/Desktop/gpt3_icl_val.npy',allow_pickle=True).item()
else:
    gpt_conclusion={}
lock=Lock()
global count
count=0
# data_list = [[i] for i in range(df1.shape[0])]
data_list = list(set([i for i in range(df1.shape[0])])-set(gpt_conclusion.keys()))
print('bar here')
bar = Bar('Processing', max=len(data_list),suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')

print('building threads')
pool = ThreadPool(processes=40)
res = pool.starmap(gpt_conclude, [[i] for i in data_list])
pool.close()
pool.join()
bar.finish()
print('save all')
np.save('/Users/yutong/Desktop/gpt3_icl_val.npy',gpt_conclusion)







