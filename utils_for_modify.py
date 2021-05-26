
from counter_utils import normal_to_special #json_save, json_read, special_to_normal, dict_merge
from xutils_for_key_vars import make_key_vars
from tqdm import tqdm
import re

key_vars = make_key_vars()

def dict_add(k,v,tgt):

    if k in tgt.keys():
        tgt[k] += v
    else:
        tgt[k] = v
    return tgt

def dict_subtract_part(to_del,vocX,rate):
    for k,v in to_del.items():
        if k in vocX.keys():
            vocX[k] = vocX[k]-int(v*rate)
    return vocX

def vocab_split(vocX,to_split,rate):
    
    to_del ={}
    X = {}
    kwds = to_split
    #to_skip = [normal_to_special(w,key_vars) for w in ['구한말']]
    for k,v in kwds.items():
        ks = normal_to_special(k,key_vars)
        vs = normal_to_special(v,key_vars)
        if ks in vocX.keys():
            for x in vs.split(' '):
                _ = dict_add(x,vocX[ks],X)  #k,v,tgt
            _ = dict_add(ks,vocX[ks],to_del)
    
    _ = dict_subtract_part(to_del,vocX,rate)
    
    return vocX

def vocab_adjust(vocX,to_adjust,rate,to_skip=[''], to_match = 0): #단어 첫 캐릭터 부터 검색하지 않는 경우의 수
                                                                 #to_match=0 의 의미는 '모두 첫 캐릭터 부터 검색'
    to_del ={}
    aa = {}
    bb = {}
    cc = {}

    kwds = to_adjust
    pre_kwds = ['']*to_match+['^']*(len(kwds[to_match:]))
    to_skip = [normal_to_special(w,key_vars) for w in to_skip]
    for i,wk in enumerate(tqdm(kwds)):
        wks = normal_to_special(wk,key_vars)
        p = re.compile(pre_kwds[i]+wks)
        nS = 0
        for k,v in vocX.items():
            
            if p.search(k) is not None and k not in to_skip:
                #print(special_to_normal3(k,key_vars,keep_double= False))
                sl = k.find(wks)
                el = sl +len(wks)                       
                sects = [k[:sl], k[sl:el],k[el:]]
                #print(special_to_normal3(k,key_vars,keep_double= False), sects)
                for ik,dt in enumerate([aa,bb,cc]):
                    _ = dict_add(sects[ik],v,dt)
                dict_add(k,v,to_del)
                nS += 1
        
        if nS==0:
             _ = dict_add(wks,20,bb)
                
    vocX = dict_subtract_part(to_del,vocX,rate) 
    
    return vocX, aa, bb, cc


def counter_ind_char3(idx, key_vars,keep_double=True):
    
    ch_len = key_vars['ch_len']
    CHOSUNG = key_vars['CHOSUNG']
    JUNGSUNG = key_vars['JUNGSUNG']
    BASE_CODE = key_vars['BASE_CODE']
    kor_alpha = key_vars['kor_alpha']
    d_to_single = key_vars['d_to_single']    
    word_comb = []
    nc = [CHOSUNG, JUNGSUNG, 1]
    w_idx = BASE_CODE
    n_ind = 0
    d_vowel = 0
    
    for i,id in enumerate(idx):
        if d_vowel == 1:
            d_vowel = 0
            id = dbl_id
            
        if id > 83 or ((id//ch_len > 0) and (n_ind == 0)):
            #print(id)
            if id > len(kor_alpha)-1 or id < 0 : 
                print("list index out of range, id = {}".format(id))
                word_comb.append(' ') 
            else:
                word_comb.append(kor_alpha[id])  
            w_idx = BASE_CODE
            n_ind = 0
            continue
        if id<0:
            word_comb.append(' ')
            w_idx = BASE_CODE
            n_ind = 0
            continue
            
        w_idx += nc[id//ch_len] * (id%ch_len)    #초,중,종성 구분 + 각 분류에서 몇번째 인지
        n_ind += 1
                       
        if i == len(idx)-1:                      #맨 마지막이면..
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            n_ind = 0
        
        # or (idx[i+1]>ch_len*3):             한 글자가 끝나거나 다음에 글자가 아닌 것이 오면 ..
        elif (idx[i+1]//ch_len==0) or (idx[i+1] < 0) or (idx[i+1] > 83):            
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            w_idx = BASE_CODE
            n_ind = 0

        elif ((id//ch_len == 1) and (idx[i+1]//ch_len ==1)): # 모음이 겹치면 
            if keep_double:           
                word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
                w_idx = BASE_CODE
                n_ind = 0 
            else:
                d_vowel = 1
                if (id%ch_len,idx[i+1]%ch_len) in d_to_single.keys():
                    dbl_id = d_to_single[(id%ch_len,idx[i+1]%ch_len)]+ch_len                   
                else:                   
                    dbl_id = id%ch_len + ch_len #  임시로 추후 수정할 것
                    print( '###########  key error : {} ###########'.format((id%ch_len,idx[i+1]%ch_len)))
                    
                w_idx = w_idx - nc[id//ch_len] *  (id%ch_len)
            
    return ''.join(word_comb) 
 
    
def special_to_normal3(s,key_vars,keep_double=True):
    BASE_CODE = key_vars['BASE_CODE']

    return counter_ind_char3([ord(c)-BASE_CODE for c in s], key_vars,keep_double)