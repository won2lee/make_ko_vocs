# insert_voc(dct,k,v)
# extract_bpe_sentences_from_normal2(vocabs, num_files=2)
# recursive_forward(w,vocabs,sub_words)
# to_bpe_sents3(sentences, vocabs,key_vars, sum_voc_vals)
# to_bpe_sents2(sentences, vocabs,key_vars)
# dict_subtract(src_dict, tgt_dict)
# dict_replace(src_dict, tgt_dict)
# counter_vocab_initialize3(counters,key_vars)
# counter_vocab_initialize2(counters,key_vars)
# bpe_vocab_iteration(path, iter_size,step_from_start, step_size = 250)
# vocab_select2(vocabs,vocabs_freq, path,iter_size, step_size,step_from_start,save_cyc)
# dpe_iteration2(vocab, vocabs_freq, iter_num)

# find_sec_josa2(vocabs, words_tuned,key_vars): #vocabs: list of tuple
# counter_convert2(test_keyword, key_vars):
# normal_to_special2(word, key_vars):
# bpe_josa_by_len(josa_dict,n_syllable=1):
# bpe_vocab_tunning3(vocabs, to_tune, josa, key_vars):
# bpe_vocab_sec_tunning(vocabs, to_tune, sec_josa, key_vars)
# json_save(data, f_name)
# json_read(f_name)
# merge_dict_of_list(dict1,dict2):
# to_find_verb_josa_through_ha_dangha_siki(pre_bpe_vocabs):
# josa_for_verb_noun(verb_josa_sorted) # '당하 하 시키' 추가 하는 함수
# josa_by_len(josa_list)
# vocab_tunning3(vocabs, to_tune, josa, key_vars
# bpe_find_to_tune_4(sorted_vocabs,josa_list,key_vars,cutline, num_of_letters)
# bpe_find_to_tune_3(sorted_vocabs,josa_list,key_vars,cutline, num_of_letters)
# adjust_word_to_tune2(words_to_tune, num_of_letters)
# find_to_tune_3(sorted_vocabs,josa_list,key_vars,cutline, num_of_letters)

import re
from collections import Counter
from tqdm.notebook import tqdm
#from to_k_alpha import convert
#from utils import *
#from utils import *
from counter_utils import get_filelist, voc_combined, call_sentences  #, *
#from counter_vocab_tuning import *
#from vocab_preprocess import call_sentences #,*
#from vocab_tuning_after_bpe import *

import numpy as np


def insert_voc(dct,k,v):
    if k in dct.keys():
        dct[k] += v
    else:
        dct[k] = v

def extract_bpe_sentences_from_normal2(vocabs, num_files=2):
    file_list = get_filelist()
    sentences = call_sentences(file_list, num_files)
    #vocabs = json_read(path+file)
    vocs, _ = voc_combined(vocabs)
    vocabs_to_get_sents = {}
    p = re.compile(r'걟걟$')
    for k, v in vocs.items():
        insert_voc(vocabs_to_get_sents, p.sub('걟',k), v)
    return sentences, vocabs_to_get_sents

def recursive_forward(w,vocabs,sub_words):
    wlen = len(subword)
    for i in reversed(range(1,wlen+1)):
        if w[:i] in vocabs[str(i)].keys():
            sub_word.append(w[:i])
            if i<wlen:
                sub_words = like_beam_forward(w[i:],vocabs,sub_words)
                return sub_words
            else:
                return sub_words
    return sub_words  
                        
"""
def backward_check(s_w,vocabs,fre_sum):

    n_bw = [s_w]

    if len(s_w) > 1: 
        sub_sum = -1000000
        temp =[]

        for j in range(1,len(s_w)):
            if (s_w[:j] in vocabs.keys()) and (s_w[j:] in vocabs.keys()):
                temp = [s_w[:j], s_w[j:]]
                sub_sum = sum([np.log(vocabs[s]/fre_sum) for s in temp])
                n_bw = temp if np.log(vocabs[s_w]/fre_sum) < sub_sum 
                break                                                    
    return n_bw
"""

def forward_check(subword,vocabs,fre_sum,forward=True):

    ws_fw = [subword]
    if len(subword) > 1: 
        sub_sum = -1000000
        temp =[]
        
        dir_to_check = reversed(range(1,len(subword))) #if forward else range(1,len(subword))

        for j in dir_to_check:
            if (subword[:j] in vocabs.keys()) and (subword[j:] in vocabs.keys()):
                temp = [subword[:j], subword[j:]] if forward else [subword[j:],subword[:j]]
                sub_sum = sum([np.log(vocabs[s]/fre_sum) for s in temp])
                if (np.log(vocabs[subword]/fre_sum)+1.*(1-np.sqrt(len(subword)/5)) < sub_sum):
                    ws_fw = temp  
                break

    return ws_fw


def to_bpe_sents4(sentences, vocabs,key_vars, sum_voc_vals):

    vocabs.pop('')
    all_s = []
    #fre_sum = sum(vocabs.values())
    fre_sum = sum_voc_vals
    p = re.compile(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9_\-]+')
    p2 = re.compile(r'[가-힣]')
    q = re.compile(r'\.$')
    r = re.compile(r'\s+')
    #u = re.compile(r'(?P<to_fix>[,\(\)\'\"\<\>])')
    u = re.compile(r'(?P<to_fix>[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ0-9_\-\.])')
    uek = re.compile(r'(?P<en>[A-Za-z])(?P<ko>[가-힣ㄱ-ㅎ])')
    uke = re.compile(r'(?P<ko>[가-힣ㄱ-ㅎ])(?P<en>[A-Za-z])')
    z = re.compile(r'걟+')
    #u1 = re.compile(r'‘')
    #u2 = re.compile(r'”')
    #ord('“'),ord('‘'),ord('"'),ord("'")
    #(8220, 8216, 34, 39)
                   
    for ii,s in enumerate(tqdm(sentences)):
        st = []
        for c in s:
            if ord(c) == 8220:
                st.append(chr(34))
            elif ord(c) == 8216:
                st.append(chr(39))
            else:
                st.append(c)
        st = ''.join(st) 
        st = uek.sub('\g<en> \g<ko>', st)
        st = uke.sub('\g<ko> \g<en>', st)
        st = u.sub(' \g<to_fix> ',st)
        #st = u1.sub(' ‘',st)
        #st = u2.sub(' ”',st)
        st = q.sub(' .',st)
        st = r.sub(' ',st)
        st = [wd for wd in st.strip().split(' ')]
        """
        sent = []
        for wd in st.strip().split(' '):
            if wd[-1] == '_':
                sent += [wd]
            else:
                sent += [wd,'_']
        """
        #st = r.sub(' ',' '.join([u.sub(' \g<to_fix> ',wd) for wd in st])).split(' ')
            
        if ii<20:
            print("s : {}".format(st))
        sentence = []
        for w in st:
            if (w == '') or (w ==' '):continue

            if p.search(w) is not None:                       
                sentence += [w,'_']
                continue                    
            #print(w)
            n_fw = []
            n_bw = []

            bpe_w= ''.join([chr(i+BASE_CODE) for i in counter_convert2(w,key_vars)[1]])
            
            s_w = bpe_w + '걟'
            
            if s_w in vocabs.keys():
                n_bf = [s_w]
            elif z.sub('',s_w[:-1]) in vocabs.keys():
                n_bf = [z.sub('',s_w[:-1])]
            else:
                n_bf = []
            
            ifw = 2
            while len(z.sub('',s_w)):
                #print(s_w)
                pre_sw = s_w
                #print("s_w : {}".format(s_w))

                for i in reversed(range(1,len(s_w)-ifw+1)):   # len(s_w) -2 로 한 것은 n_bf와 중복을 피하기 위해 ...

                    if s_w[:i] in vocabs.keys():
                        
                        n_fw += forward_check(s_w[:i],vocabs,fre_sum)
                        """
                        if len(s_w[:i]) > 1: 
                            sub_sum = -1000000
                            temp =[]

                            for j in reversed(1,len(s_w[:i])):
                                if (s_w[:j] in vocabs.keys()) and (s_w[j:i] in vocabs.keys()):
                                    temp = [s_w[:j], s_w[j:i]]
                                    sub_sum = sum([np.log(vocabs[s]/fre_sum) for s in temp])
                                    n_fw += [s_w[:i]] if np.log(vocabs[s_w[:i]]/fre_sum) > sub_sum else temp
                                    break
                                 
                        else:
                            n_fw += [s_w[:i]] 

                        """    
                        #n_fw.append(s_w[:i])
                        #s_w = '' if i == len(s_w)-1 else s_w[i:]
                        s_w = s_w[i:]
                        break
                ifw = 0                
                if s_w == pre_sw:
                    n_fw.append("<" + s_w +">")
                    insert_voc(vocabs,"<" + s_w +">",1)
                    break


            """                                                    
            s_w = bpe_w + '걟'
            ibw = 1
            while len(s_w):
                #print(s_w)
                pre_sw = s_w
                #print("s_w : {}".format(s_w))
                
                for i in range(ibw,len(s_w)):  # start를 1로 한 것은 n_bf와 중복을 피하기 위해...
                    #print(s_w[i:])
                    
                    if s_w[i:] in vocabs.keys():
                        n_bw += forward_check(s_w[i:],vocabs,fre_sum,forward=False)
                        s_w = [] if i == 0 else s_w[:i]
                        break
                    elif z.sub('',s_w[i:]) in vocabs.keys():
                        n_bw += forward_check(z.sub('',s_w[i:]),vocabs,fre_sum, forward=False)
                        s_w = [] if i == 0 else s_w[:i]
                        break
                                                      
                ibw = 0
                print([special_to_normal(k,key_vars) for k in n_bw])
                if s_w == pre_sw:
                    n_bw.append("<" + s_w +">")
                    insert_voc(vocabs,"<" + s_w +">",1)
                    break

            """    
            bf = sum([np.log(vocabs[s]/fre_sum) for s in n_bf]) if n_bf !=[] else -100000
            fw = sum([np.log(vocabs[s]/fre_sum) for s in n_fw])         
            bw = sum([np.log(vocabs[s]/fre_sum) for s in n_bw]) if n_bw !=[] else -100000 
            
            #print("fw :{}, bw :{}".format(fw,bw))
            if bf + 1.*(1-np.sqrt(len(bpe_w)/5)) < max(fw,bw):
                n_w = n_fw if fw > bw else n_bw[::-1]
            else:
                n_w = n_bf
            print([special_to_normal(k,key_vars) for k in n_w], [special_to_normal(k,key_vars) for k in n_bf],
                 [special_to_normal(k,key_vars) for k in n_fw], [special_to_normal(k,key_vars) for k in n_bw])                            
            sentence += n_w + ['걟'] if n_w[-1][-1] !='걟' else n_w
        all_s.append(sentence)

    sents = []
    for s in tqdm(all_s):
        sent = []
        for w in s:
            to_add = special_to_normal(w,key_vars) if p2.search(w) is not None else w
            sent.append(to_add)    
        #sents.append([special_to_normal(w,key_vars) for w in s])
        sents.append(sent)
    
    return sents 
    

def to_bpe_sents3(sentences, vocabs,key_vars, sum_voc_vals):

    all_s = []
    #fre_sum = sum(vocabs.values())
    fre_sum = sum_voc_vals
    p = re.compile(r'[^가-힣ㄱ-ㅎㅏ-ㅣ0-9_\-]+')
    p2 = re.compile(r'[가-힣]')
    q = re.compile(r'\.$')
    r = re.compile(r'\s+')
    #u = re.compile(r'(?P<to_fix>[,\(\)\'\"\<\>])')
    u = re.compile(r'(?P<to_fix>[^A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ0-9_\-\.])')
    uek = re.compile(r'(?P<en>[A-Za-z])(?P<ko>[가-힣ㄱ-ㅎ])')
    uke = re.compile(r'(?P<ko>[가-힣ㄱ-ㅎ])(?P<en>[A-Za-z])')
    z = re.compile(r'걟+')
    #u1 = re.compile(r'‘')
    #u2 = re.compile(r'”')
    #ord('“'),ord('‘'),ord('"'),ord("'")
    #(8220, 8216, 34, 39)
                   
    for ii,s in enumerate(tqdm(sentences)):
        st = []
        for c in s:
            if ord(c) == 8220:
                st.append(chr(34))
            elif ord(c) == 8216:
                st.append(chr(39))
            else:
                st.append(c)
        st = ''.join(st) 
        st = uek.sub('\g<en> \g<ko>', st)
        st = uke.sub('\g<ko> \g<en>', st)
        st = u.sub(' \g<to_fix> ',st)
        #st = u1.sub(' ‘',st)
        #st = u2.sub(' ”',st)
        st = q.sub(' .',st)
        st = r.sub(' ',st)
        st = [wd for wd in st.strip().split(' ')]
        """
        sent = []
        for wd in st.strip().split(' '):
            if wd[-1] == '_':
                sent += [wd]
            else:
                sent += [wd,'_']
        """
        #st = r.sub(' ',' '.join([u.sub(' \g<to_fix> ',wd) for wd in st])).split(' ')
            
        if ii<20:
            print("s : {}".format(st))
        sentence = []
        for w in st:
            if (w == '') or (w ==' '):continue

            if p.search(w) is not None:                       
                sentence += [w,'_']
                continue                    

            n_fw = []
            n_bw = []

            bpe_w= ''.join([chr(i+BASE_CODE) for i in counter_convert2(w,key_vars)[1]])
            
            s_w = bpe_w + '걟'
            
            if s_w in vocabs.keys():
                n_bf = [s_w]
            elif s_w[:-1] in vocabs.keys():
                n_bf = [s_w[:-1]]
            else:
                n_bf = []
            
            ifw = 2
            while len(z.sub('',s_w)):
                #print(s_w)
                pre_sw = s_w
                #print("s_w : {}".format(s_w))

                for i in reversed(range(1,len(s_w)-ifw+1)):   # len(s_w) -2 로 한 것은 n_bf와 중복을 피하기 위해 ...

                    if s_w[:i] in vocabs.keys():
                        n_fw.append(s_w[:i])
                        #s_w = '' if i == len(s_w)-1 else s_w[i:]
                        s_w = s_w[i:]
                        break
                
                if s_w == pre_sw:
                    n_fw.append("<" + s_w +">")
                    insert_voc(vocabs,"<" + s_w +">",1)
                    break
                ifw = 0

            s_w = bpe_w + '걟'
            ibw = 1
            while len(s_w):
                #print(s_w)
                pre_sw = s_w
                #print("s_w : {}".format(s_w))
                
                for i in range(ibw,len(s_w)):  # start를 1로 한 것은 n_bf와 중복을 피하기 위해...
                    #print(s_w[i:])
                    if s_w[i:] in vocabs.keys():
                        
                        n_bw.append(s_w[i:])
                        s_w = [] if i == 0 else s_w[:i]
                        break

                    if z.sub('',s_w[i:]) in vocabs.keys():
                        
                        n_bw.append(z.sub('',s_w[i:]))
                        s_w = [] if i == 0 else s_w[:i]
                        break                
                if s_w == pre_sw:
                    n_bw.append("<" + s_w +">")
                    insert_voc(vocabs,"<" + s_w +">",1)
                    break
                ibw = 0
                
            bf = sum([np.log(vocabs[s]/fre_sum) for s in n_bf]) if n_bf !=[] else -100000
            fw = sum([np.log(vocabs[s]/fre_sum) for s in n_fw])         
            bw = sum([np.log(vocabs[s]/fre_sum) for s in n_bw])  
            
            #print("fw :{}, bw :{}".format(fw,bw))
            if bf < max(fw,bw):
                n_w = n_fw if fw > bw else n_bw[::-1]
            else:
                n_w = n_bf
                                        
            sentence += n_w + ['걟'] if n_w[-1][-1] !='걟' else n_w
        all_s.append(sentence)

    sents = []
    for s in tqdm(all_s):
        sent = []
        for w in s:
            to_add = special_to_normal(w,key_vars) if p2.search(w) is not None else w
            sent.append(to_add)    
        #sents.append([special_to_normal(w,key_vars) for w in s])
        sents.append(sent)
    
    return sents

    


def to_bpe_sents2(sentences, vocabs,key_vars):
    all_s = []
    for s in tqdm(sentences):
        print("s : {}".format(s))
        sentence = []
        for w in s.split(' '):
            if w == '':continue
            n_w = []

            s_w = ''.join([ch_start]+[chr(i+BASE_CODE) for i in counter_convert2(w,key_vars)[1] if i !=' '])
            while len(s_w):
                #print(s_w)
                pre_sw = s_w
                #print("s_w : {}".format(s_w))

                for i in reversed(range(1,len(s_w)+1)):

                    if s_w[:i] in vocabs.keys():
                        n_w.append(s_w[:i])
                        s_w = [] if i == len(s_w) else s_w[i:]
                        break
                
                if s_w == pre_sw:
                    n_w.append("<" + s_w +">")
                    break

            sentence += n_w
        all_s.append(sentence)

    sents = []
    for s in tqdm(all_s):
        sents.append([special_to_normal(w,key_vars) for w in s])
    
    return sents




def dict_subtract(src_dict, tgt_dict):
    
    deleted_items = 0
    for k,v in tqdm(src_dict.items()):        
        if k in tgt_dict.keys():
            del tgt_dict[k]
            deleted_items += 1
            
    print("deleted items : {}".format(deleted_items))    
    
    return deleted_items


def dict_replace(src_dict, tgt_dict):

    deleted_items = 0
    replaced_items = 0
    for k,v in tqdm(src_dict.items()):        
        if k in tgt_dict.keys():
            if v>0:
                tgt_dict[k] = v
                replaced_items += 1
            else:
                del tgt_dict[k]
                deleted_items += 1
            
            
    print("deleted items : {}".format(deleted_items))    
    
    return deleted_items, replaced_items


def counter_vocab_initialize3(counters,key_vars): # counters : LIST(TUPLE)
    
    BASE_CODE = key_vars['BASE_CODE']
    ch_start = key_vars['ch_start']
    
    vocab = {}

    
    for k,v in tqdm(counters):
        _, idx, to_check = counter_convert2(k,key_vars)
        
        #if len(to_check) > 0 : print('to_check : {}'.format(to_check))
        #kw = ''.join([chr(i+BASE_CODE) for i in idx]+[ch_start]*2)
        
        kw = ''.join([chr(i+BASE_CODE) for i in idx])
 
        vocab[kw] = v

    return vocab


def counter_vocab_initialize2(counters,key_vars): # counters : LIST(TUPLE)
    
    BASE_CODE = key_vars['BASE_CODE']
    ch_start = key_vars['ch_start']
    
    vocab = {}

    
    for k,v in tqdm(counters):
        _, idx, to_check = counter_convert2(k,key_vars)
        
        #if len(to_check) > 0 : print('to_check : {}'.format(to_check))
        kw = ''.join([ch_start]*2+[chr(i+BASE_CODE) for i in idx])
 
        vocab[kw] = v

    return vocab

def bpe_vocab_iteration(path, iter_size,step_from_start, step_size = 250):

    save_cyc =1000

    vocabs =json_read(path+'vocabs' + str(step_from_start)+'.json')
    vocabs_freq =json_read(path+'vocs_freq' + str(step_from_start)+'.json')
    new_bpe_vocabs,vocabs_f,_,_ = vocab_select2(vocabs, vocabs_freq, path, iter_size, step_size, step_from_start, save_cyc)
    return new_bpe_vocabs, vocabs_f


def vocab_select2(vocabs,vocabs_freq, path,iter_size, step_size,step_from_start,save_cyc):
    pre_voc_volume = 0
    n = 0
    for i in tqdm(range(iter_size//step_size)):
        
        vocabs, vocabs_freq = dpe_iteration2(vocabs, vocabs_freq, step_size)
        _, voc_volume = voc_combined(vocabs) 
        print("iter_number :{}, voc_volume : {}".format((i+1)*step_size+step_from_start, voc_volume))
        iter_n = (i+1) * step_size 
        if iter_n % save_cyc == 0:
            json_save(vocabs, path+ 'vocabs' + str(step_from_start+iter_n))
            json_save(vocabs_freq, path+ 'vocs_freq'+str(step_from_start+iter_n))
           
        if voc_volume < pre_voc_volume:n+=1
        pre_voc_volume = voc_volume
        if n>3:break
       
    step_from_start += iter_size
    
    return vocabs, vocabs_freq, voc_volume, step_from_start


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
""" 
vocab = {'l o w </w>' : 5,
         'l o w e r </w>' : 2,
         'n e w e s t </w>':6,
         'w i d e s t </w>':3
         }
"""


def dpe_iteration2(vocab, vocabs_freq, iter_num):
    num_merges = iter_num
    new_best = []
    for i in tqdm(range(num_merges)):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        vocabs_freq[''.join(best)] = (best, pairs[best])
        new_best.append(best)
        if i%50 == 49:
            print([[special_to_normal(q,key_vars) for q in p] for p in new_best])
            new_best = []
    return vocab, vocabs_freq    


def find_sec_josa2(vocabs, words_tuned,key_vars): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    
    words_with_j = []
        
    for c,v in tqdm(vocabs):
        s = special_to_normal(c,key_vars)
        for w in words_tuned:
            if w in s:
                l = s.find(w)+len(w)
                words_with_j.append((s[:l],s[l:]))
    josa = {}
    for s,v in words_with_j:
        if v in josa.keys():
            josa[v] +=1
        else:
            josa[v] =1
    sorted_josa_list = sorted(josa.items(), key =lambda x:x[1], reverse = True)    
    
    return sorted_josa_list, words_with_j


def counter_convert2(test_keyword, key_vars, all_char=False):
        
    ch_len = key_vars['ch_len']
    CHOSUNG = key_vars['CHOSUNG']
    JUNGSUNG = key_vars['JUNGSUNG']
    BASE_CODE = key_vars['BASE_CODE']
    kor_alpha = key_vars['kor_alpha'] 
    CHOSUNG_LIST  = key_vars['CHOSUNG_LIST'] 
    JUNGSUNG_LIST = key_vars['JUNGSUNG_LIST']
    JONGSUNG_LIST = key_vars['JONGSUNG_LIST'] 
    double_vowel = key_vars['double_vowel'] 
    s_to_double = key_vars['s_to_double']
    k_alpha_to_num = key_vars['k_alpha_to_num']
        
    split_keyword_list = list(test_keyword)
    #print(split_keyword_list)

    result = list()
    indexed = list()
    to_check = list()
    num_etc = {}
    for i,s in enumerate('0123456789-_'):
        num_etc[s] = i+84
    n_check = 0  #  이전에 [가-힣]이 없었다  
    p = re.compile(r'.*[가-힣]+.*')
    q = re.compile(r'[ㄱ-ㅎㅏ-ㅣ]')
    
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if len(keyword)>1:print(keyword, split_keyword_list)
        if p.match(keyword) is not None:
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            indexed.append(char1)
            result.append(CHOSUNG_LIST[char1])
            #print('초성 : {}'.format(CHOSUNG_LIST[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            #print("char2 in double_vowel :{}".format(char2 in double_vowel))
            if char2 in double_vowel:                
                indexed += [s + ch_len for s in s_to_double[char2]]
            else:
                indexed.append(char2 + ch_len)
            result.append(JUNGSUNG_LIST[char2])
            #print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            
            if char3==0:
                result.append('#')
            else:
                result.append(JONGSUNG_LIST[char3])
                indexed.append(char3 +ch_len*2)
            n_check  = 1
            #print('종성 : {}'.format(JONGSUNG_LIST[char3]))
        else:
            result.append(keyword)
            if keyword in '0123456789-_':
                 indexed.append(num_etc[keyword])
            elif keyword == ' ':
                indexed.append(keyword)
            elif q.match(keyword) is not None:
                if n_check == 0: #앞에 정상적인 한글이 없었다면  ㄱ ㄴ 등을 종성으로 인식
                    if keyword in k_alpha_to_num[1].keys():
                        indexed.append(k_alpha_to_num[1][keyword])
                    elif keyword in k_alpha_to_num[0].keys():
                        indexed.append(k_alpha_to_num[0][keyword])

                else:            #바로 앞에 정상적인 한글이 있었다면  ㄱ ㄴ 등을 초성으로 인식
                    if keyword in k_alpha_to_num[0].keys():
                        indexed.append(k_alpha_to_num[0][keyword])
                    elif keyword in k_alpha_to_num[1].keys():
                        indexed.append(k_alpha_to_num[1][keyword])              
            else:
                if all_char:
                    indexed.append(keyword)
                else:
                    #indexed.append('') 
                    to_check.append(keyword)
            
            n_check  = 0    
                
            #if ord(keyword) <127:
            #    indexed.append(ord(keyword)+54)  # 한글이 아닌 경우 잠정적으로 +54  숫자, 영문 대소문자 포함
               
                
    # result
    return "".join(result), indexed, to_check    #indexed는 자음(초성과 종성 구분), 모음 구분하여 인덱스 각 => [ㄱ ㅏ ㄱ], [0,28,57]

def normal_to_special2(word, key_vars,all_char=False, show_num = False):
    
    #print('normal_to_special')
    
    BASE_CODE = key_vars['BASE_CODE']
    kor_alpha = key_vars['kor_alpha']
    
    special = []
    for i in counter_convert2(word,key_vars, all_char)[1]:
        if type(i) == str:
            special.append(i) 
        elif show_num and i > 83:
            special.append(kor_alpha[i])
            
        else:
            special.append(chr(i+BASE_CODE))

            
        #if i != ' ':
        #    special.append(chr(i+BASE_CODE))
        #else:
        #    special.append(' ')       
    
    return ''.join(special)

""" 
def bpe_josa_by_len2(josa_dict,n_syllable=0):  
    
    josa_list = []
    for k,v in josa_dict.items():
        josa_list += v
    
    josa_list = list(set(josa_list))
    josa_by_len = {}
    print(len(josa_list))
    
    if n_syllable > 1:
        josa_list = [normal_to_special2(s, key_vars) for s in josa_list if len(s) > 1]
    else:
        josa_list = [normal_to_special2(s, key_vars) for s in josa_list if len(s) > 0]
        
    max_len = min(max([len(s) for s in josa_list]),15)
    
    for i in range(max_len+3):
        josa_by_len[str(i)] = []
    
    for j in josa_list:
        if (len(j) > max_len) or (j ==''):
            continue
        josa_by_len[str(len(j)+2)]+=[j+'걟걟'] 
        
    
    return josa_by_len  
"""

def bpe_josa_by_len(josa_dict,key_vars, n_syllable=0):  
    
    josa_list = []
    for k,v in josa_dict.items():
        josa_list += v 

    josa_list = list(set(josa_list))
    if '' in josa_list:
        josa_list.remove('')
    josa_by_len = {}
    print(len(josa_list))
 
    if n_syllable > 1:
        josa_list = [normal_to_special2(s, key_vars) for s in josa_list if len(s) > 1]
    else:
        josa_list = [normal_to_special2(s, key_vars) for s in josa_list if len(s) > 0]
          
    max_len = min(max([len(s) for s in josa_list]),15)
    
    for i in range(max_len+1):
        josa_by_len[str(i)] = []
    
    for j in josa_list:
        if len(j) > max_len:
            continue
        josa_by_len[str(len(j))]+=[j] 
        
    
    return josa_by_len


def bpe_vocab_tunning4(vocabs, to_tune, josa, cutline, key_vars, adj_rate = 0.5):
    #to_tune = [normal_to_special(k,key_vars) for k,v in words_to_tune.items()]
    #to_tune = [k for k,v in words_to_tune.items()]
    #josa = [normal_to_special(j,key_vars) for j in josa1 +josa2]
    #josa = josa1 +josa2
    tuned_vocabs1 = {}
    tuned_vocabs2 = {}
    separated_terms = []
    to_replace = {}
    prt_n = 0
    mc1 = 0 if cutline >2 else adj_rate ** (cutline+1)
    mc2 = 1 if cutline >2 else (1-adj_rate **(cutline+1))   # cutline adjusted v
    
    #=re.compile(r'[0-9가-힣]')
    for c,v in tqdm(vocabs.items()):
        """ 
        num_tune = 0
        #for sc in k.split(' '):
        for c in k.split(' '):
            #c = special_to_normal(sc,key_vars)
            #print(c)
        """ 
        w = []
        
        if len(c) > 2:
            #n = 0
            for i in reversed(range(1,len(josa))):
                if i>len(c)-1:
                    continue
                if (c[:-i] in to_tune) and (c[-i:] in josa[str(i)]):
                    # w +=[normal_to_special(c[:-i],key_vars), "_걟"+normal_to_special(c[-i:],key_vars)]
                    if prt_n <20:
                        print(c, special_to_normal(c[:-i],key_vars),special_to_normal(c[-i:], key_vars))
                    prt_n +=1
                    #w +=[c[:-i], "_걟"+c[-i:]]
                    w = [c[:-i], c[-i:]]
                    separated_terms.append(c)
                    #num_tune += 1
                    #n += 1
                    break


            #if n == 0:w.append(c)

        #else:
        #    w.append(c)

        #if num_tune > 0:
        #k1,k2 = ' '.join(w).split(' _걟')
        #k1,k2 = w
        #k1,k2 = ' '.join(w).split(' _')
        
        if len(w) > 0:
            tuned_vocabs1 = add_dict_item(w[0],v * mc2,tuned_vocabs1)
            tuned_vocabs2 = add_dict_item(w[1],v * mc2,tuned_vocabs2)

            to_replace[c] = v * mc1
          
            
    return tuned_vocabs1, tuned_vocabs2, separated_terms, to_replace 



def bpe_vocab_tunning3(vocabs, to_tune, josa, cutline, key_vars, adj_rate = 0.5):
    #to_tune = [normal_to_special(k,key_vars) for k,v in words_to_tune.items()]
    #to_tune = [k for k,v in words_to_tune.items()]
    #josa = [normal_to_special(j,key_vars) for j in josa1 +josa2]
    #josa = josa1 +josa2
    tuned_vocabs1 = {}
    tuned_vocabs2 = {}
    separated_terms = []
    to_replace = {}
    prt_n = 0
    mc1 = 0 if cutline >2 else adj_rate ** (cutline+1)
    mc2 = 1 if cutline >2 else (1-adj_rate **(cutline+1))   # cutline adjusted v
    
    #=re.compile(r'[0-9가-힣]')
    for k,v in tqdm(vocabs.items()):
        w = []
        num_tune = 0
        #for sc in k.split(' '):
        for c in k.split(' '):
            #c = special_to_normal(sc,key_vars)
            #print(c)
            #if len(c) > 2:
            if len(c) > 3:
                n = 0
                for i in range(1,len(josa)):
                    if (c[:-i] in to_tune) and (c[-i:] in josa[str(i)]):
                        # w +=[normal_to_special(c[:-i],key_vars), "_걟"+normal_to_special(c[-i:],key_vars)]
                        if prt_n <20:
                            print(c, special_to_normal(c[:-i],key_vars),special_to_normal(c[-i:], key_vars))
                        prt_n +=1
                        w +=[c[:-i], "_걟"+c[-i:]]
                        separated_terms.append(c)
                        num_tune += 1
                        n += 1
                        break
                    if i>len(c)-1:
                        break
                
                if n == 0:w.append(c)
                
            else:
                w.append(c)

        if num_tune > 0:
            k1,k2 = ' '.join(w).split(' _')
            
            tuned_vocabs1 = add_dict_item(k1,v * mc2,tuned_vocabs1)
            tuned_vocabs2 = add_dict_item(k2,v * mc2,tuned_vocabs2)
            
            to_replace[k] = v * mc1
          
            
    return tuned_vocabs1, tuned_vocabs2, separated_terms, to_replace 


def bpe_vocab_sec_tunning3(vocabs, to_tune, sec_josa, cutline, key_vars, adj_rate = 0.5):

    tuned_vocabs1 = {}
    tuned_vocabs2 = {}
    to_tune = sorted(to_tune, key=lambda x:len(x))
    minl = min([len(x) for x in to_tune])    
    to_replace = {}
    mc1 = 0 if cutline >2 else adj_rate ** (cutline+1)
    mc2 = 1 if cutline >2 else (1-adj_rate **(cutline+1))   # cutline adjusted v
    prt_n =0
    
    for s,v in tqdm(vocabs.items()):

        if len(s) < minl + 1:
            continue

        w =[]    
        for ws in to_tune:
            if len(ws) > len(s)-1: break
            if ws in s:
                l = s.find(ws)+len(ws)
                if s[l:] in sec_josa:
                    if prt_n <100:
                        print(s, special_to_normal(s[:l],key_vars),special_to_normal(s[l:], key_vars))
                    prt_n +=1                        

                    # w +=[normal_to_special(s[:l],key_vars), "_걟"+normal_to_special(s[l:],key_vars)]
                    w = [s[:l], s[l:]]
                    break
                                   
        if len(w)>0:
            tuned_vocabs1 = add_dict_item(w[0],v * mc2,tuned_vocabs1)
            tuned_vocabs2 = add_dict_item(w[1],v * mc2,tuned_vocabs2)

            to_replace[s] = v * mc1
            
            
    return tuned_vocabs1, tuned_vocabs2, to_replace



def bpe_vocab_sec_tunning2(vocabs, to_tune, sec_josa, cutline, key_vars, adj_rate = 0.5):

    tuned_vocabs1 = {}
    tuned_vocabs2 = {}

    to_replace = {}
    mc1 = 0 if cutline >2 else adj_rate ** (cutline+1)
    mc2 = 1 if cutline >2 else (1-adj_rate **(cutline+1))   # cutline adjusted v
    prt_n =0
    
    for k,v in tqdm(vocabs.items()):
        w = []
        num_tune = 0
        for s in k.split(' '):
            
            n = 0
            
            #s = special_to_normal(sc,key_vars)
            #if len(s.replace('_','')) < 3: 
            if len(s.replace('걟','')) < 5:
                # w += [sc]
                w += [s]
                continue 

            
            for ws in to_tune:
                if ws in s:
                    l = s.find(ws)+len(ws)
                    if s[l:] in sec_josa:
                        if prt_n <100:
                            print(s, special_to_normal(s[:l],key_vars),special_to_normal(s[l:], key_vars))
                        prt_n +=1                        
                        
                        # w +=[normal_to_special(s[:l],key_vars), "_걟"+normal_to_special(s[l:],key_vars)]
                        w +=[s[:l], "_걟"+s[l:]]
                        n += 1
                        break
            if n == 0:
                w += [s]
            
            num_tune += n

        if num_tune > 0:
            k1,k2 = ' '.join(w).split(' _')
            
            tuned_vocabs1 = add_dict_item(k1,v * mc2,tuned_vocabs1)
            tuned_vocabs2 = add_dict_item(k2,v * mc2,tuned_vocabs2)
            
            to_replace[k] = v * mc1
            
            
    return tuned_vocabs1, tuned_vocabs2, to_replace

def merge_dict_of_list(dict1,dict2):

    for k,l in dict1.items():
        if k in dict2.keys():
            dict2[k] = list(set(dict2[k] + l))
        else:
            dict2[k] = list(set(l))
            
    return dict2  

def put_to_dict(k,v,dic):  # for to_find_verb_josa_through_ha_dangha_siki(pre_bpe_vocabs)
    if k in dic.keys():
        dic[k][1] += v
    else:
        dic[k] = [1,v]

def to_find_verb_josa_through_ha_dangha_siki(pre_bpe_vocabs):
        
    verb_josa  = {}
    verb_josa2 = {}

    p = re.compile(r'^_+[가-힣]+당하')
    q = re.compile(r'^_+[가-힣]+하')
    r = re.compile(r'^_+[가-힣]+시키')

    for k,v in tqdm(pre_bpe_vocabs.items()):

        n_k = special_to_normal(k, key_vars)

        if p.match(n_k):
            m_k = p.sub('',n_k)
            put_to_dict(m_k,v,verb_josa)        
        elif q.match(n_k):
            m_k = q.sub('',n_k) 
            put_to_dict(m_k,v,verb_josa)         
        elif r.match(n_k):
            m_k2 = r.sub('',n_k) 
            put_to_dict(m_k2,v,verb_josa2)  # 시키 로 시작되는 경우 
        else:
            continue

    verb_josa_sorted = sorted(verb_josa.items(), key = lambda x:x[1][0], reverse=True) #당하, 하 로 찾은 조사
    verb_josa2_sorted = sorted(verb_josa2.items(), key = lambda x:x[1][0], reverse=True) # 시키  로 찾은 조사
    # to_find_josa로 잦은 조사   
    
    return verb_josa_sorted, verb_josa2_sorted



def josa_for_verb_noun(verb_josa_sorted, verb_josa2_sorted): # 당하 하 시키 추가 하는 함수
    josa_find_nouns =[]

    for k,v in verb_josa_sorted[:200]:
        if k !='':
            josa_find_nouns += ['당하'+ k, '하' +k] 
    for k,v in verb_josa2_sorted:
        if k !='':
            josa_find_nouns += ['시키'+ k] 


    josa_check_nouns = josa_find_nouns.copy()

    p = re.compile(r'^이ㅓ')
    q = re.compile(r'^ㅣ')
    r = re.compile(r'^어ㅆ')
    s = re.compile(r'^돼ㅆ')
    for k,v in verb_josa_sorted[:200]:
        if k =='': 
            continue
        if q.match(k):
            josa_check_nouns += [s.sub('됐',q.sub('돼',k))]         

        else:    
            josa_check_nouns += ['되'+r.sub('었',p.sub('어',k))] 
            josa_check_nouns += ['이'+r.sub('었',p.sub('어',k))]


    josa_to_search_nouns = josa_by_len(josa_find_nouns)
    josa_to_check_nouns = josa_by_len(josa_check_nouns)
    
    return josa_to_search_nouns, josa_to_check_nouns

  
    
def josa_by_len(josa_list):  
    
    josa_by_len = {}
    max_len = min(max([len(s) for s in josa_list]),15)
    print(len(josa_list))
    for i in range(max_len):
        josa_by_len[str(i)] = []
    
    for j in josa_list:
        if len(j) > max_len - 1:
            continue
        josa_by_len[str(len(j))]+=[j] 
        
    josa_by_len[str(0)] = [] 
    
    return josa_by_len

def vocab_tunning3(vocabs, to_tune, josa, key_vars):
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
                    if (c[:-i] in to_tune) and (c[-i:] in josa[str(i)]):
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

def bpe_find_to_tune_4(sorted_vocabs,josa_list,key_vars,cutline, num_of_letters, min_josa_len = 2): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    j_len = len(josa_list)
    words_with_s = []
    candidate_with_j = {}
    bc = key_vars['BASE_CODE']
    p=re.compile(r'[0-9ㄱ-ㅎㅏ-ㅣ가-힣]')
    #q=re.compile(r'걟+')
    for k,v in tqdm(sorted_vocabs):
        #if len(q.sub('',k)) < num_of_letters * 2 + 1 : continue
        if len(k) < num_of_letters * 2 + 1 : continue
        # k = special_to_normal(s,key_vars)
        #print(k)

        for i in reversed(range(min_josa_len,j_len)):
                 
            #print(i)
            if i>len(k)-1:continue
            bbb= josa_list[str(i)]
            if k[-i:] in bbb:
                if k[:-i] in candidate_with_j.keys():
                    candidate_with_j[k[:-i]] += 1
                elif p.match(k[-(i+1)]): 
                    candidate_with_j[k[:-i]] = 1
                break
               

    words_to_tune_special = [k for k,v in candidate_with_j.items() if v>cutline]
    longer_than =[]
    for k in words_to_tune_special:
        if len(k)>2*num_of_letters:
            longer_than.append(k)
        elif (len(k)== 2*num_of_letters) and (ord(k[-1])-bc>27) and (ord(k[-1])-bc< 49): #종성으로 끝나지 않는경우
            longer_than.append(k)
            
    #words_to_tune_special = [k for k in words_to_tune_special if len(q.sub('',k))>2*num_of_letters]
    words_to_tune_normal = [special_to_normal(k,key_vars) for k in longer_than]
 
    return longer_than, words_to_tune_normal





def bpe_find_to_tune_3(sorted_vocabs,josa_list,key_vars,cutline, num_of_letters): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    j_len = len(josa_list)
    words_with_s = []
    candidate_with_j = {}
    bc = key_vars['BASE_CODE']
    p=re.compile(r'[0-9ㄱ-ㅎㅏ-ㅣ가-힣]')
    q=re.compile(r'걟+')
    for k,v in tqdm(sorted_vocabs):
        if len(q.sub('',k)) < num_of_letters * 2 + 1 : continue
        # k = special_to_normal(s,key_vars)
        #print(k)
        if (len(k)>3):
            for i in reversed(range(1,j_len)):
                 
                #print(i)
                if i>len(k)-1:continue
                bbb= josa_list[str(i)]
                if k[-i:] in bbb:
                    if k[:-i] in candidate_with_j.keys():
                        candidate_with_j[k[:-i]] += 1
                    elif p.match(k[-(i+1)]): 
                        candidate_with_j[k[:-i]] = 1
                    break
               

    words_to_tune_special = [k for k,v in candidate_with_j.items() if v>cutline]
    longer_than =[]
    for k in words_to_tune_special:
        if len(q.sub('',k))>2*num_of_letters:
            longer_than.append(k)
        elif (len(q.sub('',k))== 2*num_of_letters) and (ord(k[-1])-bc>27) and (ord(k[-1])-bc< 49): #종성으로 끝나지 않는경우
            longer_than.append(k)
            
    #words_to_tune_special = [k for k in words_to_tune_special if len(q.sub('',k))>2*num_of_letters]
    words_to_tune_normal = [special_to_normal(k,key_vars) for k in longer_than]
 
    return longer_than, words_to_tune_normal


def adjust_word_to_tune2(words_to_tune, num_of_letters):
    
    if num_of_letters == 2:
        p=re.compile(r'.*[0-9ㄱ-ㅎㅏ-ㅣ가-힣\-][0-9ㄱ-ㅎㅏ-ㅣ가-힣\-]') # 2 글자 이상
    elif num_of_letters == 3:
        p=re.compile(r'.*[0-9ㄱ-ㅎㅏ-ㅣ가-힣\-][0-9ㄱ-ㅎㅏ-ㅣ가-힣\-][0-9ㄱ-ㅎㅏ-ㅣ가-힣\-]') # 3 글자 이상
    elif num_of_letters > 3:
        p=re.compile(r'.*[0-9ㄱ-ㅎㅏ-ㅣ가-힣\-][0-9ㄱ-ㅎㅏ-ㅣ가-힣\-][0-9ㄱ-ㅎㅏ-ㅣ가-힣\-][0-9ㄱ-ㅎㅏ-ㅣ가-힣\-]') # 4 글자 이상
    else:
        p=re.compile(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣]*[0-9ㄱ-ㅎㅏ-ㅣ가-힣\-]')
        
    return [k for k,v in words_to_tune.items() if p.match(k)]






def find_to_tune_3(sorted_vocabs,josa_list,key_vars,cutline, num_of_letters): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    j_len = len(josa_list)
    words_with_s = []
    candidate_with_j = {}
    p=re.compile(r'[0-9ㄱ-ㅎㅏ-ㅣ가-힣]')
    q=re.compile(r'걟+')
    for s,v in tqdm(sorted_vocabs):
        if len(q.sub('',s)) < num_of_letters * 2 + 2 : continue
        k = special_to_normal(s,key_vars)
        #print(k)
        if (len(k)>3):
            for i in reversed(range(1,j_len)):
                 
                #print(i)
                if i>len(k)-1:continue
                bbb= josa_list[str(i)]
                if k[-i:] in bbb:
                    if k[:-i] in candidate_with_j.keys():
                        candidate_with_j[k[:-i]] += 1
                    elif p.match(k[-(i+1)]): 
                        candidate_with_j[k[:-i]] = 1
                    break
               

    words_to_tune = {}
    for k,v in candidate_with_j.items():
        if (v >cutline):
            words_to_tune[k] = v

    return adjust_word_to_tune2(words_to_tune, num_of_letters) # 두 글자 이상만 반영   
    #return words_to_tune
