import re
#from tqdm.notebook import tqdm
from tqdm import tqdm
from counter_utils import special_to_normal, add_dict_item

#from counter_vocab_tuning import add_dict_item


def bpe_josa_by_len2(josas,key_vars, n_syllable=0):  

    if type(josas) is dict:
        josa_list = []
        for k,v in josas.items():
            josa_list += v 
        josa_list = list(set(josa_list))
    else:
        josa_list = josas
        
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


def bpe_vocab_tunning5(vocabs, to_tune, josa, cutline, key_vars, adj_rate = 0.5): # dict, list, list
    #to_tune = [normal_to_special(k,key_vars) for k,v in words_to_tune.items()]
    #to_tune = [k for k,v in words_to_tune.items()]
    #josa = [normal_to_special(j,key_vars) for j in josa1 +josa2]
    #josa = josa1 +josa2
    tuned_vocabs1 = {}
    tuned_vocabs2 = {}
    separated_terms = []
    to_replace = {}
    prt_n = 0
    mc1 = 0 if cutline >1 else adj_rate ** (cutline+1)
    mc2 = 1 if cutline >1 else (1-adj_rate **(cutline+1))   # cutline adjusted v
    
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
                if len(josa[str(i)]) ==0:continue
                if (c[-i:] in josa[str(i)]) and (c[:-i] in to_tune):
                #if (c[:-i] in to_tune) and (c[-i:] in josa[str(i)]):
                    # w +=[normal_to_special(c[:-i],key_vars), "_걟"+normal_to_special(c[-i:],key_vars)]
                    if prt_n <10 or prt_n%1000==0:
                        print(c, special_to_normal(c[:-i],key_vars))
                        print(special_to_normal(c[-i:], key_vars))
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

def bpe_find_to_tune_5(sorted_vocabs,josa_list,key_vars,cutline, num_of_letters, to_check, min_josa_len = 2): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    j_len = len(josa_list)
    nch = len(to_check)
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
                
            if nch > 0:
                nc = 0
                for c in to_check:
                    if k[-(i+len(c)):-i] in to_check:
                        nc +=1
                        break
                if nc == 0:continue
            
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


def bpe_find_to_tune_recursive_5(tuned_vocabs,josa_list,key_vars,cutline, num_of_letters, min_josa_len = 2): #vocabs: list of tuple
    #nums = [n for n in '0123456789']
    #numbers =[]
    j_len = len(josa_list)
    words_with_s = []
    candidate_with_j = {}
    bc = key_vars['BASE_CODE']
    p=re.compile(r'[0-9ㄱ-ㅎㅏ-ㅣ가-힣]')
    #q=re.compile(r'걟+')
    
    key_words = []
    longer_than =[]
    
    for k in tqdm(tuned_vocabs):
        #if len(q.sub('',k)) < num_of_letters * 2 + 1 : continue
        #if len(k) < num_of_letters * 2 + 1 : continue
        # k = special_to_normal(s,key_vars)
        #print(k)
        nc = 0

        for i in reversed(range(min_josa_len,j_len)):
                 
            #print(i)
            
            if i>len(k)-1:continue
            bbb= josa_list[str(i)]
            if k[-i:] in bbb:
                print(special_to_normal(k,key_vars))
                nc =1

                if len(k[:-i])>2*num_of_letters:
                    longer_than.append(k[:-i])
                elif (len(k[:-i])== 2*num_of_letters) and (ord(k[-(i+1)])-bc>27) and (ord(k[-(i+1)])-bc< 49): #종성으로 끝나지 않는경우
                    longer_than.append(k[:-i])

                else:
                    key_words.append(k)
                break
        if nc == 0:
            key_words.append(k)
            
            
    #words_to_tune_special = [k for k in words_to_tune_special if len(q.sub('',k))>2*num_of_letters]
    
    print([special_to_normal(k,key_vars) for k in longer_than])
    key_words = list(set(key_words + longer_than))
    words_to_tune_normal = [special_to_normal(k,key_vars) for k in key_words]
 
    return key_words, words_to_tune_normal
