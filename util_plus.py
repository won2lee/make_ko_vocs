from counter_utils import voc_combined, special_to_normal
from xutils_for_key_vars import make_key_vars

def bpe_vocab_initialize():
    
    path = './generated_Data4/'
    file_list = get_filelist()
    bpe_vocabs = vocab_build(file_list,key_vars,from_file=True)
    json_save(bpe_vocabs, path+'vocabs_'+ str(0))
    return bpe_vocabs

#bpe_vocabs = bpe_vocab_initialize


def bpe_voc_in_normal(bpe_vocabs):
    key_vars = make_key_vars()
    voc_ex, _ = voc_combined(bpe_vocabs)
    sorted_voc_ex = sorted(voc_ex.items(), key=lambda x:x[1], reverse=True)
    readable = [(special_to_normal(k,key_vars),v) for k, v in sorted_voc_ex]
    return readable, sorted_voc_ex


def to_make_josa_list(words_with_josa):
    
    josa_found = [v for k,v in words_with_josa]
    josa ={}
    for j in josa_found:
        if j in josa.keys():
            josa[j] +=1
        else:
            josa[j] =1

    josa_to_apply = sorted(josa.items(), key=lambda x:x[1], reverse=True)
    josa_to_apply = [k for k,v in josa_to_apply]
    josa_list ={}
    josa_to_apply = josa_to_apply[:300]
    max_l = max([len(j) for j in josa_to_apply])
    for i in range(max_l+1):
        josa_list[i] =[]
    for j in josa_to_apply:   
        josa_list[len(j)].append(j)
    
    return josa_list

""" 
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
                if k[-i:] in josa_list[i]:
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
"""
            #tuned_vocabs1.append(k1)
            #tuned_vocabs2.append(k2)
            #to_delete.append((k,v))
"""

            tuned_vocabs1 = add_dict_item(k1,v,tuned_vocabs1)
            tuned_vocabs2 = add_dict_item(k2,v,tuned_vocabs2)

            to_delete[k] = v
          
            
    return tuned_vocabs1, tuned_vocabs2, separated_terms, to_delete    
    
"""    
def extract_bpe_sentences_from_normal():
    file_list = get_filelist()
    sentences = call_sentences(file_list, num_files=2)
    vocs, _ = voc_combined(pre_bpe_vocabs)
    vocabs_to_get_sents = {}
    p = re.compile('^걟')
    for k, v in vocs.items():
        vocabs_to_get_sents[p.sub('',k)] = v
    return sentences, vocabs_to_get_sents


def to_eval_whether_to_split3(voc_f,vocs_1ch,vocs_count, iter_sum, pre_sum, km, kd, key_vars):

    print('start process of [to_eval_whether_to_split]')
    new_vocs = {}
    new_voc_f = {}
    vf ={} 
    vc ={}
    #selected = ''
    after_comb = np.zeros((2,))
    before_comb = np.zeros((2,))
    eval_comb = []
    s_vocs = sorted(voc_f.items(), key=lambda x:x[1][1], reverse=True)
    for ix,(k,v) in enumerate(tqdm(s_vocs)):
        #print(v[1], voc_f[v[0][0]][1], voc_f[v[0][1]][1])
        #print(special_to_normal(v[0][0], key_vars), special_to_normal(v[0][1], key_vars))
        for i,c in enumerate([v[0][0],v[0][1]]):
            vf[i] = voc_f[c][1] if len(c) >1 else vocs_1ch[c]     # voc_f : {key:[[v01,vo2],v1]}
            vc[i] = vocs_count[c] if len(c) >1 else vocs_1ch[c] 
        after_comb[0] = np.log(v[1]/iter_sum)
        before_comb[0] = np.log(vf[0]/iter_sum) + np.log(vf[1]/iter_sum)
        after_comb[1] = np.log(vocs_count[k]/pre_sum) 
        before_comb[1] = np.log(vc[0]/pre_sum)+np.log(vc[1]/pre_sum)
        eval_comb.append(np.concatenate((after_comb, before_comb),0))
        
        if ix < 70000:
            new_vocs[k] = v[1]
            new_voc_f[k] = [(v[0][0],v[0][1]),v[1]]
        elif ((after_comb[0] > before_comb[0]+1.5/70000* ix-0.5) 
              or (after_comb[1] > before_comb[1]+2.5/70000* ix-1.3)):
            new_vocs[k] = v[1]
            new_voc_f[k] = [(v[0][0],v[0][1]),v[1]]

            
        #print("preprocessed : {}, {}, {}".format(after_comb[0], before_comb[0], selected)) 
        #print("original_bpe : {}, {}".format(after_comb[1], before_comb[1]))
        #selected = ''
    deleted_items, replaced_items = dict_merge(vocs_1ch, new_vocs)
    return  new_vocs, new_voc_f,eval_comb

def get_new_vocs():
    import numpy as np

    iter_num = '70000'
    f_path ='./generated_Data11/bpe/vocs_freq'+iter_num+'.json'
    voc_f = json_read(f_path)

    bpe_vocabs, split_bpe_vocabs = get_split_bpe_vocabs()
    voc_1ch = get_single_char_freq(split_bpe_vocabs)
    voc_1ch['걟'] = sum(json_read('./generated_Data11/preproc/josa_vocabs_008_after_n_c0_n1.json').values())

    vocs_count = json_read('./generated_Data11/bpe/voc_count_70000.json')
    iter_vocabs =json_read('./generated_Data11/bpe/vocabs'+iter_num+'.json')

    iter_sum = sum(iter_vocabs.values())
    pre_sum = sum(bpe_vocabs.values())
    km = 1.
    kd = 4000

    new_vocs, new_voc_f, eval_comb = to_eval_whether_to_split3(
        voc_f,voc_1ch,vocs_count, iter_sum, pre_sum, km, kd, key_vars)

    vcs = [[],[],[],[]]
    for i in range(4):
        vcs[i] = [s[i] for s in eval_comb]
    
    return new_vocs, vocs_count

def extract_bpe_sentences_from_normal3(vocabs, num_files=2):
    file_list = get_filelist()
    sentences = call_sentences(file_list, num_files)
    #vocabs = json_read(path+file)
    vocs, _ = voc_combined(vocabs)
    """
    vocabs_to_get_sents = {}
    p = re.compile(r'걟걟$')
    for k, v in vocs.items():
        insert_voc(vocabs_to_get_sents, p.sub('걟',k), v)
    """
    return sentences, vocs
