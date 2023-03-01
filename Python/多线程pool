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

df1=pd.read_json(os.path.join('/GPFS/data/yutongmeng/VCR/vcg_annots/visualcomet','test_annots.json'))



def gpt_conclude(i):
    # print(i)
    openai.api_key = api_keys[i%len(api_keys)]
    a=df1[i:i+1]
    data_dict=list(a.T.to_dict().values())[0]
    prompt="Event: [\"2 was doing his homework\"],\
Place: [\"school\"],\
Intent: [\"read his notes over\", \"review his text and study\"],\
Before: [\"sat down at the table\", \"opened his notebook\", \"put his books on the table\", \"sit down with his friends\", \"see 1 waving the paper\"],\
After: [\"sigh in frustration\", \"find somewhere quieter to sit\", \"pick up a paper\", \"read the paper he\"s touching\"],\
Event: [\"3 is sitting at the end of the table looking concerned\"],\
Place: [\"in a restaurant\"],\
Intent: [\"listen to 8's story\", \"wait for the check\", \"get another drink\"],\
Before: [\"invite everyone over for dinner\", \"sit down at the table\", \"come to the dinner\", \"sit down\", \"start eating\"],\
After: [\"finish eating dinner\", \"stand up and clear the table\", \"listen to 8's story\", \"react to the story\", \"finish his meal\"],\
Event: [\"1 is sitting in a suit and waiting for someone\"],\
Place: [\"in a cafe\"],\
Intent: [\"have coffee with his girlfriend\", \"go on a blind date\", \"look nice\"],\
Before: [\"walk into the bar\", \"walk up to a barstool\", \"go to the bar\", \"set a date to meet them\"],\
After: [\"speak to the server\", \"place an order\", \"meet someone there\", \"leave the bar\"],\
Event: [\"1 is sitting at a table at a busy night club\"],\
Place: [\"in a nightclub\"],\
Intent: [\"to go out with drinks with her friends\", \"share a date with a romantic love interest\"],\
Before: [\"style her hair\", \"put on makeup\", \"go to the night club\", \"be seated by a waiter\", \"be asked out on a date\", \"put on makeup\"],\
After: [\"talk to her date\", \"order a drink\", \"order a drink\", \"talk to her date\"],\
Event: [\"a group of soldiers are on horses\"],\
Place: [\"a battlefield\"],\
Intent: [\"ride into battle\", \"kill the enemy\"],\
Before: [\"put on their armor\", \"mount their horses\", \"ride towards the enemy army\", \"ride to the battle field\", \"see the enemy\"],\
After: [\"listen to their commander speaking\", \"follow their commander's lead\", \"charge at the enemy\", \"yell a battle cry\"],\
Event: [\"3 is sharing his plans with 1 and 2\"],\
Place: [\"school lunch\"],\
Intent: [\"be the leader\", \"ask for help\", \"be a show off\"],\
Before: [\"look at 1 and 2\", \"sit down\", \"want to tell his plans to others\", \"come up with plans\"],\
After: [\"stand up\", \"walk away\", \"ask 1 and 2 to do something\", \"leave 1 and 2\"],"
    prompt_=prompt+f'Event: [\"{str(data_dict["event"])}\"],Place: [\"{str(data_dict["place"])}\"],'
    #    ,f"Intent: {str(data_dict['intent'])}",f"Intent: {str(data_dict['before'])}" ,f"After: {str(data_dict['after'])}" ,"Please describe the scene in about 30 words:"])

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
    # print('to here count')
    global count  ##不声明会卡住
    count=count+1
    # print('to here count2')
    # print(count)
   
    if count%100==8:
        np.save('/GPFS/data/yutongmeng/GPT_3/gpt3_icl_test.npy',gpt_conclusion)
    lock.release()
    # print('lock release')

gpt_conclusion=np.load('/GPFS/data/yutongmeng/GPT_3/gpt3_icl_test.npy',allow_pickle=True).item()
lock=Lock()
global count
count=0
# data_list = [[i] for i in range(df1.shape[0])]
data_list = list(set([i for i in range(df1.shape[0])])-set(gpt_conclusion.keys()))

bar = Bar('Processing', max=len(data_list),suffix='%(index)d/%(max)d - %(percent).1f%% - %(elapsed)ds- %(eta)ds')


pool = ThreadPool(processes=50)
res = pool.starmap(gpt_conclude, [[i] for i in data_list])
pool.close()
pool.join()
bar.finish()
np.save('/GPFS/data/yutongmeng/GPT_3/gpt3_icl_test.npy',gpt_conclusion)





