
import re
import collections
from collections import Counter
from tqdm.notebook import tqdm
from counter_utils import json_save, json_read, normal_to_special, special_to_normal #*  

#josa1=[c for c in '가는은를을의에']
#josa2 = ['에게','에서', '에는' ] #, josa 3 : '이다', '이고', '하고', '하여', '해서', 되어, '하게', '한다', '했다']
#sec_josa = ['하고', '했다', '였다', '하여', '로', '과', '도', '이며', '에서는', '와', '으로', '이', '에서도', '해', '했고', '하는', '된', '이다']


# 이, 들, 으로, 부터, 로, 에, 와, 과, 로는, 에서는, 에서도, 으로는, 으로도, 로부터, 으로부터, 
#이런 경우 n 개 이상에 해당되는 경우 ==> 1차 추출, 
#다시 앞에 이 단어들이 나오는 형태 중에 
#뒤에 josa3 가 따라오면 분절하는 방법 ==> 2차 추출, 이때 새로운 조사들도 찾아 추가
#1차 추출 후 스타트는 '_' 에서 '__' (더블 언더스코어), 추출된 조사 앞에 '_' 추가  => 나중에 한개인 경우 없에고 두개인 경우만 잔존
#2차 추출 후에도 추출된 조사 앞에 '_' 추가
# 중복 제거는 2만 ~4만 사이에서 
# 혹은 몇차례 반복 !! ---예를들어 1만 iteration 마다 적용
# 1차 추출 단어 리스트 print
# 2차 추출 단어 리스트 print
# 숫자와 영어는 분절하지 않고 잔존.. 영어 잔존 ! 그 자체로 한개의 단어화 =>convert
# 처음 부터 새로 

# 들의 처리 
# 맨뒤에 오는 가는은을의에 등이 있을 수 있고 
# 그 앞에 들이 있을 수 있음
# 들 앞의 글자가 리스트에 있을 경우 
# 분리

# 분리의 의미는 
# dict 에서  기존의 키를 없애고  두개로 분리 
# 기존의 키 중에 같은 키가 있으면 밸류를 합산 
# 합산의 의미는 1차적으로 키의 축소 계산 부담의 축소의 의미가 있을 수 있을 것

# from utils import *

def voc_file_call(path, step_from_start):

    pre_vocabs =json_read('./generated_Data4/vocabs_'+str(step_from_start)+'.json')
    vocs, _ = voc_combined(pre_vocabs)
    
    #vocabs_read =json_read(path +'vocabs_'+str(step_from_start)+'.json')
    #vocs, _ = voc_combined(vocabs_read)
    #vocs, _ = voc_combined(number_attach(vocabs_read))
    
    return pre_vocabs, sorted(vocs.items(), key=lambda x:x[1], reverse=True)


def adjust_word_to_tune(words_to_tune, num_of_letters):
    
    if num_of_letters == 2:
        p=re.compile(r'.*[0-9가-힣\-][0-9가-힣\-]') # 2 글자 이상
    elif num_of_letters == 3:
        p=re.compile(r'.*[0-9가-힣\-][0-9가-힣\-][0-9가-힣\-]') # 3 글자 이상
    elif num_of_letters > 3:
        p=re.compile(r'.*[0-9가-힣\-][0-9가-힣\-][0-9가-힣\-][0-9가-힣\-]') # 4 글자 이상
    else:
        p=re.compile(r'.*[0-9가-힣\-]')
        
    adjusted = [(k,v) for k,v in words_to_tune.items() if p.match(k)]
    return [item[0] for item in sorted(adjusted, key = lambda x:x[1], reverse=True)]


def find_to_tune(sorted_vocabs,josa1,josa2,josa3,key_vars,cutline, num_of_letters): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    words_with_s = []
    candidate_with_j = {}
    p=re.compile(r'[0-9가-힣]')
    for s,v in tqdm(sorted_vocabs):
        k = special_to_normal(s,key_vars)
        if (len(k)>3):
            if k[-1] in josa1:
                if k[:-1] in candidate_with_j.keys():
                    candidate_with_j[k[:-1]] += 1
                elif p.match(k[-2]): 
                    candidate_with_j[k[:-1]] = 1
            elif k[-2:] in josa2:
                if k[:-2] in candidate_with_j.keys():
                    candidate_with_j[k[:-2]] += 1
                elif p.match(k[-3]): 
                    candidate_with_j[k[:-2]] = 1
            elif k[-3:] in josa3:
                if k[:-3] in candidate_with_j.keys():
                    candidate_with_j[k[:-3]] += 1
                elif p.match(k[-4]): 
                    candidate_with_j[k[:-3]] = 1                
                    
                    
        #if k[0]=='_':
        #    words_with_s.append(k[1:])      

        #if ((len(k) >2) and (k[1] in nums)) or (k[0] in nums):
        #    numbers.append(k)
        
    words_to_tune = {}
    for k,v in candidate_with_j.items():
        if (v >cutline):
            words_to_tune[k] = v

    return adjust_word_to_tune(words_to_tune, num_of_letters) # 두 글자 이상만 반영   
    #return words_to_tune

def find_to_tune_2(sorted_vocabs,josa_list,key_vars,cutline, num_of_letters): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    j_len = len(josa_list)
    words_with_s = []
    candidate_with_j = {}
    p=re.compile(r'[0-9가-힣]')
    for s,v in tqdm(sorted_vocabs):
        k = special_to_normal(s,key_vars)
        #print(k)
        if (len(k)>3):
            for i in reversed(range(1,j_len)):
                 
                #print(i)
                if i>len(k)-1:continue
                if k[-i:] in josa_list[str(i)]:
                    if k[:-i] in candidate_with_j.keys():
                        candidate_with_j[k[:-i]] += 1
                    elif p.match(k[-(i+1)]): 
                        candidate_with_j[k[:-i]] = 1
                    break
               

    words_to_tune = {}
    for k,v in candidate_with_j.items():
        if (v >cutline):
            words_to_tune[k] = v

    return adjust_word_to_tune(words_to_tune, num_of_letters) # 두 글자 이상만 반영   
    #return words_to_tune



def find_overlab_words(vocabs, key_vars): #vocabs : list of tuples
    
    overlab_words = []
    for s, v  in tqdm(vocabs):
        k = special_to_normal(s,key_vars)
        if k in words_with_s:
            overlab_words.append(k)
    
    return overlab_words


def dict_merge(src_dict, tgt_dict):
    
    updated_items = 0
    added_items = 0
    
    for k,v in tqdm(src_dict.items()):
        if k in tgt_dict.keys():
            tgt_dict[k] =  tgt_dict[k] + v 
            updated_items += 1
        else:
            tgt_dict[k] = v
            added_items += 1
    
    print("updated items : {},  added items : {}".format(updated_items, added_items)) 
    
    return updated_items, added_items


def dict_subtract(src_dict, tgt_dict):
    
    deleted_items = 0
    for k,v in tqdm(src_dict.items()):        
        if k in tgt_dict.keys():
            del tgt_dict[k]
            deleted_items += 1
            
    print("deleted items : {}".format(deleted_items))    
    
    return deleted_items

def add_dict_item(key,val,dict):
    if key in dict.keys():
        dict[key] = dict[key] + val
    else:
        dict[key] = val
    return dict

def add_start_ch(bpe_vocab):
    double_starter ={}
    for k,v in bpe_vocab.items():
        double_starter["걟"+k] = v
    return double_starter


def vocab_tunning2(vocabs, to_tune, josa, key_vars):
    #to_tune = [normal_to_special(k,key_vars) for k,v in words_to_tune.items()]
    #to_tune = [k for k,v in words_to_tune.items()]
    #josa = [normal_to_special(j,key_vars) for j in josa1 +josa2]
    #josa = josa1 +josa2
    tuned_vocabs1 = {}
    tuned_vocabs2 = {}
    to_delete = {}
    separated_terms = []
    #=re.compile(r'[0-9가-힣]')
    for k,v in tqdm(vocabs.items()):
        w = []
        num_tune = 0
        for sc in k.split(' '):
            c = special_to_normal(sc,key_vars)
            #print(c)
            if len(c) > 2:
                n = 0
                for i in range(1,len(josa)):
                    if (c[:-i] in to_tune) and (c[-i:] in josa[i]):
                        w +=[normal_to_special(c[:-i],key_vars), "_걟"+normal_to_special(c[-i:],key_vars)]
                        separated_terms.append(c)
                        num_tune += 1
                        n += 1
                        break
                    if i>len(c)-1:
                        break
                
                if n == 0:w.append(sc)
                
            else:
                w.append(sc)

        if num_tune > 0:
            k1,k2 = ' '.join(w).split(' _')

            tuned_vocabs1 = add_dict_item(k1,v,tuned_vocabs1)
            tuned_vocabs2 = add_dict_item(k2,v,tuned_vocabs2)

            to_delete[k] = v

    return tuned_vocabs1, tuned_vocabs2, separated_terms, to_delete

def vocab_tunning(vocabs, to_tune, josa1, josa2, key_vars):
    #to_tune = [normal_to_special(k,key_vars) for k,v in words_to_tune.items()]
    #to_tune = [k for k,v in words_to_tune.items()]
    #josa = [normal_to_special(j,key_vars) for j in josa1 +josa2]
    josa = josa1 +josa2
    tuned_vocabs1 = {}
    tuned_vocabs2 = {}
    to_delete = {}
    separated_terms = []
    #=re.compile(r'[0-9가-힣]')
    for k,v in tqdm(vocabs.items()):
        w = []
        num_tune = 0
        for sc in k.split(' '):
            c = special_to_normal(sc,key_vars)
            #print(c)
            if len(c) > 2:
                if (c[:-2] in to_tune) and (c[-2:] in josa2):
                    w +=[normal_to_special(c[:-2],key_vars), "_걟"+normal_to_special(c[-2:],key_vars)]
                    separated_terms.append(c)
                    num_tune += 1
                    continue
                elif (c[:-1] in to_tune) and (c[-1:] in josa1):
                    w +=[normal_to_special(c[:-1],key_vars), "_걟"+normal_to_special(c[-1:],key_vars)]
                    separated_terms.append(c)
                    num_tune += 1
                    continue  
                else:
                    w.append(sc)
            else:
                w.append(sc)

        if num_tune > 0:
            k1,k2 = ' '.join(w).split(' _')
            """
            tuned_vocabs1.append(k1)
            tuned_vocabs2.append(k2)
            to_delete.append((k,v))
            """

            tuned_vocabs1 = add_dict_item(k1,v,tuned_vocabs1)
            tuned_vocabs2 = add_dict_item(k2,v,tuned_vocabs2)

            to_delete[k] = v
          
            
    return tuned_vocabs1, tuned_vocabs2, separated_terms, to_delete


def find_sec_josa(vocabs, words_tuned,key_vars): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    
    words_with_j = []


    for w in tqdm(words_tuned):
        
        for c,v in vocabs:
            s = special_to_normal(c,key_vars)
            if w in s:
                l = s.find(w)+len(w)
                words_with_j.append((s[:l],s[l:]))
              
    return words_with_j


def vocab_sec_tunning(vocabs, to_tune, sec_josa, key_vars):

    tuned_vocabs1 = {}
    tuned_vocabs2 = {}
    to_delete = {}

    for k,v in tqdm(vocabs.items()):
        w = []
        num_tune = 0
        for sc in k.split(' '):
            
            n = 0
            
            s = special_to_normal(sc,key_vars)
            if len(s.replace('_','')) < 3: 
                w += [sc]
                continue 

            
            for ws in to_tune:
                if ws in s:
                    l = s.find(ws)+len(ws)
                    if s[l:] in sec_josa:
                        print(s, s[:l],s[l:])
                        w +=[normal_to_special(s[:l],key_vars), "_걟"+normal_to_special(s[l:],key_vars)]
                        n += 1
                        break
            if n == 0:
                w += [sc]
            
            num_tune += n

        if num_tune > 0:
            k1,k2 = ' '.join(w).split(' _')
            
            tuned_vocabs1 = add_dict_item(k1,v,tuned_vocabs1)
            tuned_vocabs2 = add_dict_item(k2,v,tuned_vocabs2)

            to_delete[k] = v
                      
    return tuned_vocabs1, tuned_vocabs2, to_delete


def call_words_to_tune(file_name):
    with open(file_name, 'r') as f:
        to_tune = f.read()
    return to_tune



