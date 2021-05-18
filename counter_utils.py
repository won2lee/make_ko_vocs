

import re
import collections
from collections import Counter
from tqdm.notebook import tqdm
from xutils_for_key_vars import make_key_vars


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


def counter_vocab_initialize(counters,key_vars): # counters : LIST(TUPLE)
    
    BASE_CODE = key_vars['BASE_CODE']
    ch_start = key_vars['ch_start']
    
    vocab = {}

    
    for k,v in tqdm(counters):
        _, idx = counter_convert(k,key_vars)

        kw = ''.join([ch_start]*2+[chr(i+BASE_CODE) for i in idx])
 
        vocab[kw] = v

    return vocab

def double_vowel_lookup2(JUNGSUNG_LIST):
    vowels = JUNGSUNG_LIST
    s = [c for c in'ㅕㅘㅝㅟㅐㅑㅒㅖㅙㅚㅛㅞㅠㅢ']
    d = 'ㅣㅓ ㅗㅏ ㅜㅓ ㅜㅣ ㅏㅣ ㅣㅏ ㅣㅐ ㅣㅔ ㅗㅐ ㅗㅣ ㅣㅗ ㅜㅔ ㅣㅜ ㅡㅣ'.split(' ')
    s_to_d = {}

    for k,v in list(zip(s,d)): 
        s_to_d[k] = v

    ch_to_num = {}
    for i, c in enumerate(vowels):
        ch_to_num[c] = i
    #print(ch_to_num)
    
    num_to_ch = {}
    for k, v in ch_to_num.items():
        num_to_ch[v] = k

    s_to_double = {}
    d_to_singleN = {}
    for k,v in s_to_d.items():
        s_to_double[ch_to_num[k]] = tuple([ch_to_num[ch] for ch in v])

    for k,v in s_to_double.items():
        d_to_singleN[v] = k

    return s_to_double, d_to_singleN, ch_to_num, num_to_ch

def double_vowel_lookup(JUNGSUNG_LIST):
    vowels = JUNGSUNG_LIST
    s = [c for c in'ㅕㅘㅝㅟㅐㅑㅒㅖㅙㅚㅛㅞㅠㅢ']
    d = 'ㅣㅓ ㅗㅏ ㅜㅓ ㅜㅣ ㅏㅣ ㅣㅏ ㅣㅐ ㅣㅔ ㅗㅐ ㅗㅣ ㅣㅗ ㅜㅔ ㅣㅜ ㅡㅣ'.split(' ')
    s_to_d = {}

    for k,v in list(zip(s,d)): 
        s_to_d[k] = v

    ch_to_num = {}
    for i, c in enumerate(vowels):
        ch_to_num[c] = i
    #print(ch_to_num)
    
    num_to_ch = {}
    for k, v in ch_to_num.items():
        num_to_ch[v] = k

    s_to_double ={}
    for k,v in s_to_d.items():
        s_to_double[ch_to_num[k]] = [ch_to_num[ch] for ch in v]

    return s_to_double, ch_to_num, num_to_ch


def counter_convert(test_keyword, key_vars):
    
    #print(key_vars.keys()) #임시로 만든 것 
        
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
    num_etc = {}
    for i,s in enumerate('0123456789-_'):
        num_etc[s] = i+84
    n_check = 0  #  이전에 [가-힣]이 없었다
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if len(keyword)>1:print(keyword, split_keyword_list)
        if re.match('.*[가-힣]+.*', keyword) is not None:
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
            elif n_check == 0:
                if keyword in k_alpha_to_num[1].keys():
                    indexed.append(k_alpha_to_num[1][keyword])
                elif keyword in k_alpha_to_num[0].keys():
                    indexed.append(k_alpha_to_num[0][keyword])
                else:
                    indexed.append(' ')
            else:
                if keyword in k_alpha_to_num[0].keys():
                    indexed.append(k_alpha_to_num[0][keyword])
                elif keyword in k_alpha_to_num[1].keys():
                    indexed.append(k_alpha_to_num[1][keyword])
                else:
                    indexed.append(' ')               
            
            n_check  = 0    
                
            #if ord(keyword) <127:
            #    indexed.append(ord(keyword)+54)  # 한글이 아닌 경우 잠정적으로 +54  숫자, 영문 대소문자 포함
               
                
    # result
    return "".join(result), indexed    #indexed는 자음(초성과 종성 구분), 모음 구분하여 인덱스 각 => [ㄱ ㅏ ㄱ], [0,28,57]
  

    
def counter_ind_char(idx, key_vars):
    
    ch_len = key_vars['ch_len']
    CHOSUNG = key_vars['CHOSUNG']
    JUNGSUNG = key_vars['JUNGSUNG']
    BASE_CODE = key_vars['BASE_CODE']
    kor_alpha = key_vars['kor_alpha']
   
    word_comb = []
    nc = [CHOSUNG, JUNGSUNG, 1]
    w_idx = BASE_CODE
    n_ind = 0
    
    for i,id in enumerate(idx):
        if type(id) !=  int:
            word_comb.append(id)
            w_idx = BASE_CODE
            n_ind = 0
            continue
            
        if id > 83 or ((id//ch_len > 0) and (n_ind == 0)):
            #print(id)
            if id > len(kor_alpha)-1 or id < 0 : print("list index out of range, id = {}".format(id))
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
        

        elif type(idx[i+1]) ==str:     ##### 추가된 부분       
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            w_idx = BASE_CODE
            n_ind = 0        
        
        elif (idx[i+1]//ch_len==0) or (idx[i+1] < 0) or (idx[i+1] > 83):            
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            w_idx = BASE_CODE
            n_ind = 0

        elif ((id//ch_len == 1) and (idx[i+1]//ch_len ==1)): # 모음이 겹치면 
            
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            w_idx = BASE_CODE
            n_ind = 0            
            
    return ''.join(word_comb) 

def json_save(data, f_name):
    import json
    json = json.dumps(data)
    with open(f_name +".json","w") as f:
        f.write(json) 

def json_read(f_name):
    import json
    with open(f_name) as f:
        return json.load(f)


def get_filelist():
    #path = '../wikiextractor/text/AA/wiki_'
    path = '../wikiextractor/text/A'
    fileN = '0123456789'
    fileABC = 'ABCDEFG'
    #file_list = "subtxt.txt,subtxt2.txt".split(',')
    file_list = []
    for a in fileABC:
        for n in fileN:
            if (a=='G') and (n=='5'):
                file_list += [path+a+'/wiki_' + n + s for s in fileN[:9]]
                break
            else: 
                file_list += [path+a+'/wiki_' + n + s for s in fileN]
    return file_list     


def get_text(file):
    with open(file, 'r') as f:
        data = f.read()
    p = re.compile(r'\<.*\>')
    #q = re.compile(r'[^0-9a-zA-Z가-힣\-]')
    #######################################################################################
    q = re.compile(r'[^0-9가-힣\-]') #'_' 가 들어갈 경우 별도조치 필요 : start_ch 와 중복!!!!!!!!!!
    #######################################################################################
    d_out = p.sub(' ', data)
    d_out = q.sub(' ', d_out)
    return re.sub('\s{1,}',' ', d_out)  # sentence list

def get_sentences(file):
    with open(file, 'r') as f:
        data = f.read()

    p = re.compile(r'\<.*\>')
    d_out = p.sub(' ', data)

    q = re.compile(r'[^0-9a-zA-Z가-힣\-\.]')
    d_out = q.sub(' ', d_out)
    d_out = re.sub('\s{1,}',' ', d_out)

    
    sentences = d_out.split('. ')
    return [re.sub('\.',' ',s) for s in sentences]

def call_sentences(file_list, num_files):
    sentences =[]
    for f in file_list[:num_files]:    
        sentences +=  get_sentences(f)
    return sentences

def sentences_from_filelist(file_list, num_files):
    sentences =[]
    for f in file_list[:num_files]:    
        sentences +=  get_sentences(f)
    return sentences

def count_from_file(file_list):
    c = Counter()
    for f in tqdm(file_list):
        d_out =  get_text(f)
        c = c + Counter(d_out.split(' '))
    return c



"""
kor_alpha = CHOSUNG_LIST + ['#']*(ch_len - len(CHOSUNG_LIST)) + JUNGSUNG_LIST + ['#']*(ch_len - 
len(JUNGSUNG_LIST))+JONGSUNG_LIST
kor_alpha = kor_alpha+['_','$']
etc = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 
       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
       'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 
       'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
       '{', '|', '}', '~']
#alpha = [C for C in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]+[c for c in 'abcdefghijklmnopqrstuvwxyz']
kor_alpha += etc
"""

# from util_function----------------
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

# ---------------end of from util_function

# 스페셜 글자를 정상적인 글자로 전환
def special_to_normal2(s,key_vars,keep_double=True):
    BASE_CODE = key_vars['BASE_CODE']

    return counter_ind_char2([ord(c)-BASE_CODE for c in s], key_vars,keep_double)  

def special_to_normal(s,key_vars):
    BASE_CODE = key_vars['BASE_CODE']
    to_convert = []
    for c in s:
        ch_num = ord(c)-BASE_CODE
        if ch_num > -1 and ch_num <96:
            to_convert.append(ch_num)
        else:
            to_convert.append(c)
    return counter_ind_char(to_convert, key_vars)    
    
    #return counter_ind_char([ord(c)-BASE_CODE for c in s], key_vars) 

def normal_to_special(word, key_vars):
    
    #print('normal_to_special')
    
    BASE_CODE = key_vars['BASE_CODE']
    
    special = []
    for i in counter_convert(word,key_vars)[1]:
        if i != ' ':
            special.append(chr(i+BASE_CODE))
        else:
            special.append(' ')       
    
    return ''.join(special)
    

#######################################################
#  자소 인덱스를 글자로 변환하여 글자를 식별할 수 있도록 
#######################################################

def ind_char(idx, key_vars):
    
    ch_len = key_vars['ch_len']
    CHOSUNG = key_vars['CHOSUNG']
    JUNGSUNG = key_vars['JUNGSUNG']
    BASE_CODE = key_vars['BASE_CODE']
    kor_alpha = key_vars['kor_alpha']
   
    word_comb = []
    nc = [CHOSUNG, JUNGSUNG, 1]
    w_idx = BASE_CODE
    n_ind = 0
    
    for i,id in enumerate(idx):
        if id > 83:
            #print(id)
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
            #n_ind = 0
        elif (idx[i+1]//ch_len==0) or (idx[i+1] < 0): # or (idx[i+1]>ch_len*3):             한 글자가 끝나면..
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            w_idx = BASE_CODE
            n_ind = 0

    return ''.join(word_comb) 


def convert(test_keyword, key_vars):
    
    ch_len = key_vars['ch_len']
    CHOSUNG = key_vars['CHOSUNG']
    JUNGSUNG = key_vars['JUNGSUNG']
    BASE_CODE = key_vars['BASE_CODE']
    kor_alpha = key_vars['kor_alpha'] 
    CHOSUNG_LIST  = key_vars['CHOSUNG_LIST'] 
    JUNGSUNG_LIST = key_vars['JUNGSUNG_LIST']
    JONGSUNG_LIST = key_vars['JONGSUNG_LIST']    
        
    split_keyword_list = list(test_keyword)
    #print(split_keyword_list)

    result = list()
    indexed = list()
    num_etc = {}
    for i,s in enumerate('0123456789-_'):
        num_etc[s] = i+84
    
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            indexed.append(char1)
            result.append(CHOSUNG_LIST[char1])
            #print('초성 : {}'.format(CHOSUNG_LIST[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            indexed.append(char2 + ch_len)
            result.append(JUNGSUNG_LIST[char2])
            #print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            
            if char3==0:
                result.append('#')
            else:
                result.append(JONGSUNG_LIST[char3])
                indexed.append(char3 +ch_len*2)
            #print('종성 : {}'.format(JONGSUNG_LIST[char3]))
        else:
            result.append(keyword)
            if keyword in '0123456789-_':
                 indexed.append(num_etc[keyword])
            elif keyword == ' ':
                indexed.append(keyword)
                
            #if ord(keyword) <127:
            #    indexed.append(ord(keyword)+54)  # 한글이 아닌 경우 잠정적으로 +54  숫자, 영문 대소문자 포함
               
                
    # result
    return "".join(result), indexed    #indexed는 자음(초성과 종성 구분), 모음 구분하여 인덱스 각 => [ㄱ ㅏ ㄱ], [0,28,57]
  




###########################################################
###########################################################

def vocab_initialize(counters,key_vars):
    
    BASE_CODE = key_vars['BASE_CODE']
    ch_start = key_vars['ch_start']
    
    vocab = {}

    
    for k,v in tqdm(counters):
        _, idx = counter_convert(k,key_vars)

        kw = ' '.join([ch_start]+[chr(i+BASE_CODE) for i in idx])
 
        vocab[kw] = v

    return vocab


def number_attach(counters):

    vocab = {}
    p = re.compile(r"(?P<num1>[걔-걝]+)\s(?P<num2>[걔-걝]+.*)")  
    q = re.compile(r"(?P<num3>[걟][걔-걝]*)\s(?P<num4>[걔-걝]+.*)")
    num_att = []  # 체크용
    for kw,v in counters.items():
        i = 0 # 체크용
        while p.search(kw):
            kw = p.sub('\g<num1>\g<num2>', kw) 
            i+=1  # 체크용
        kw = q.sub('\g<num3>\g<num4>', kw)
        vocab[kw] = v
        num_att.append(i) # 체크용
    print(len(num_att), sum([1 for i in num_att if i>0]), max(num_att)) 
    return vocab




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


def dpe_iteration(vocab, iter_num):
    num_merges = iter_num
    for i in tqdm(range(num_merges)):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        #print(best)
    return vocab    
 
    
def voc_combined(vocab):
    vocab_sub = {}
    #for k, v in vocab:
    for k, v in vocab.items():
        for s in k.split(' '): 
            if s in vocab_sub.keys():
                vocab_sub[s] += v
            else:
                vocab_sub[s] = v
    return vocab_sub, len(vocab_sub)


def bpe_voc_in_normal(bpe_vocabs):
    key_vars = make_key_vars()
    voc_ex, _ = voc_combined(bpe_vocabs)
    sorted_voc_ex = sorted(voc_ex.items(), key=lambda x:x[1], reverse=True)
    readable = [(special_to_normal(k,key_vars),v) for k, v in sorted_voc_ex]
    return readable, sorted_voc_ex


def vocab_select(vocabs,path,iter_size, step_size,step_from_start,save_cyc):
    pre_voc_volume = 0
    n = 0
    for i in tqdm(range(iter_size//step_size)):
        
        vocabs = dpe_iteration(vocabs,step_size)
        _, voc_volume = voc_combined(vocabs) 
        print("iter_number :{}, voc_volume : {}".format((i+1)*step_size+step_from_start, voc_volume))
        iter_n = (i+1) * step_size 
        if iter_n % save_cyc == 0:
            json_save(vocabs, path+ str(step_from_start+iter_n))
           
        if voc_volume < pre_voc_volume:n+=1
        pre_voc_volume = voc_volume
        if n>3:break
       
    step_from_start += iter_size
    #json_save(vocabs, path+'vocabs_'+ str(step_from_start))
    #print("iter_number :{}, voc_volume : {}".format(step_from_start, voc_volume))
    
    return vocabs, voc_volume, step_from_start

#iter_size, step_size,step_from_start = 6,2,0   
#vocs, v_vol, from_start = vocab_select(vocab, iter_size, step_size,step_from_start)  





