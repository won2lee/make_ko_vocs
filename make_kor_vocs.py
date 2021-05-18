import re
import time
import json
from collections import Counter
from tqdm.notebook import tqdm
import copy
#from to_k_alpha import convert
#from utils import *

from counter_utils import (json_save, json_read, normal_to_special, special_to_normal, 
                           dict_merge, dict_subtract, bpe_voc_in_normal, bpe_josa_by_len, normal_to_special2) #clean !!
from xutils_for_key_vars import make_key_vars  #clean !!
from utils_from_vocProcess10 import bpe_find_to_tune_5, bpe_vocab_tunning5, bpe_find_to_tune_recursive_5 #clean!!!

path0 = '/home/john/Notebook/preProject/create_vocabs/data/'
file1 = 'vocabs_10.json'
file2 = 'josa_dict_2020_03_14.json'
path2 = 'out_data/'
path = path0+path2

def extract_nouns_verbs_subs():
    
    #path0 = '/home/john/Notebook/preProject'
    #pre_bpe_vocabs = json_read(path0+'/generated_Data11/pre_bpe_vocabs.json')
    pre_bpe_vocabs = json_read(path0+file1)
    #josa_dict = json_read('./generated_Data11/preproc/josa_dict_2020_03_14.json')
    bpe_vocabs = pre_bpe_vocabs.copy()
    new_readable_vocabs, bpe_vocs = bpe_voc_in_normal(bpe_vocabs)

    import time
    date_now = time.strftime('%Y_%m_%d', time.localtime(time.time()))
    #date_now = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    date_now
    key_vars = make_key_vars()
    #path = path0+'/generated_Data11/preproc_0311/'
    #path = path0+'/NMT_new/folder_Utils/test_data_20210514/'
    path = path0+path2


    josa_dict = json_read(path0+file2)

    #bpe_vocabs = pre_bpe_vocabs.copy()

    josa_d ={}
    josa_d['n'] = ['josa_to_find_nouns_adj_0206', 'josas_for_noun_sec'] #'josa_to_find_nouns']
    josa_d['nt'] = ['josa_to_find_nouns_adjusted_for_verb_noun', 'josas_for_noun_sec'] #'josa_to_find_nouns']
    josa_d['uht'] = ['josas_uh_uht_first', 'josas_uh_uht_sec']
    josa_d['aht'] = ['josas_ah_aht_first', 'josas_ah_aht_sec']
    josa_d['ht'] = ['josas_ss_ht','josas_ss_ht_sec']
    josa_d['ah_uh'] = ['josas_ah_uh_sec', 'josas_ah_uh_sec']
    josa_d['ha'] = ['josa_ha','josa_to_find_nouns']
    josa_d['dangha'] = ['josa_dangha','josa_dangha']
    josa_d['nn'] = ['josas_for_noun_first', 'josas_for_noun_sec']
    josa_d['ae_eh'] = ['josas_ae_ht_sec','josas_ae_ht_sec']
    josa_d['eu_eu'] = ['josas_eu_eu_sec','josas_eu_eu_sec']
    josa_d['ha_ee'] = ['josas_ha_ee_uh_sec','josas_ha_ee_uh_sec']
    josa_d['ha_ht'] = ['josas_ss_ht','josas_ss_ht']
    josa_d['doe_ht'] = ['josas_ss_ht','josas_ss_ht']
    josa_d['ee_ht'] = ['josas_ss_ht','josas_ss_ht']

    josa_d['v']= ['good_to_find_verbs_adjusted', 'good_after_find_verbs_adjusted']



    srch_cond = {'uht':[normal_to_special(s,key_vars)[-1] for s in ['루','리']],
                'aht' :[normal_to_special(s,key_vars) for s in ['보','오','모','노']],
                'ha_ee':[normal_to_special(s,key_vars)[-2:] for s in ['하']],
                'ht' :[normal_to_special(s,key_vars)[-1] for s in ['가','러']], 
                'ha_ht' :[normal_to_special(s,key_vars) for s in ['하','하ㅣ','하이ㅓ']],
                'doe_ht' :[normal_to_special(s,key_vars) for s in ['되','돼','되어']],
                'ee_ht' :[normal_to_special(s,key_vars) for s in ['이','이ㅓ','이어']],
                'ae_eh' :[normal_to_special(s,key_vars) for s in ['ㅏㅣ','ㅔ','뚜ㅣ','돼','되']],
                'eu_eu' :[normal_to_special(s,key_vars) for s in ['ㅡ']],
                'ah_uh' :[normal_to_special(s,key_vars)[-1] for s in [
                    '없','싶','옅','겪','넘','좁','녹','죽','밟','밝','신','길','늦','놓','갚','않','받','젊','쫓','앉','솟','앓']]+[
                    normal_to_special(s,key_vars)[-3:] for s in ['있']]
                }


    new_bpe_vocabs ={}
    extracted_vocabs = {}
    tuned_bpe_vocabs = {}
    josa_bpe_vocabs = {}
    repl_bpe_vocabs = {}
    seq = 0

    tuned_dict = {}
    n_iter = 0
    c_n = [('dangha',2,2,0.5),('nt',5,2,0.5),('n',3,2,0.5),('n',1,2,0.0),('nn',1,2,0.0),('n',3,1,0.0),
           ('ha',1,2,0.0),('ha_ee',2,1,0.0),('ae_eh',2,1,0.0),('uht',2,1,0.0),('aht',2,1,0.0),('ah_uh',2,1,0.0),
           ('ht',2,1,0.0),('eu_eu',2,1,0.0),
           ('dangha',1,2,0.0),('nt',1,2,0.0),('nn',0,2,0.0),('ha',1,1,0.0),
           ('ha_ee',1,1,0.0),('uht',1,1,0.0),('nt',0,1,0.0)]

           #('ae_eh',1,1,0.0),('aht',1,1,0.0),('ht',1,1,0.0),('eu_eu',1,1,0.0),('dangha',0,1,0.0),
           #('nn',0,1,0.0),('ha',0,1,0.0), ('ah_uh',0,1,0.0),
    #('ha_ee',0,1,0.5),('ae_eh',0,1,0.0),('uht',0,1,0.0),('aht',0,1,0.0),('ah_uh',0,1,0.0),
    #       ('ht',0,1,0.0),('eu_eu',0,1,0.0),]
    #n_iter = 13

    #0.5===> 0.0 으로

    #c_n = [('dangha',0,2,0.5),('nt',0,2,0.5),('n',0,2,0.5),('nn',0,2,0.5),('nn',0,1,0.5)]
    #n_iter = 18
    #c_n = [('ha',0,1,0.0),('ha_ee',0,1,0.5),('uht',0,1,0.5),('aht',0,1,0.5),('ah_uh',0,1,0.5),('ht',0,1,0.5),]

    #c_n = [('nn',1,2,0.5),('uht',1,2,0.5), ('aht',1,2,0.5)] #('nn',5,2,0.5),
    #c_n = [('dangha',2,2,0.5),('n',7,2,0.5), ('n',3,2,0.5), ('ha',2,2,0.5),('n',1,2,0.5) ]
    #c_n = [('dangha',2,2,0.5),('nt',5,2,0.5),('n',5,2,0.5),('n',3,2,0.5),('n',1,2,0.5),('nn',1,2,0.5),('nn',1,1,0.5),
    #       ('ha',0,2,0.0), ('uht',1,2,0.0), ('aht',1,2,0.0),('ha_ht',0,2,0.0),('doe_ht',0,2,0.0),('ee_ht',1,2,0.0),
    #       ('ht',0,2,0.0)]

    delete_list = {'dangha': ['우ㅣ풍당'], 'aht':['만호'],'n':['지난','으'],'nn':['가이']} #'하ㅣ야']}

    for kw,cutline,num_of_letters,adj_rate in tqdm(c_n):

        n_iter += 1
        n_step = str(n_iter)

        print("\n\n#### start [n_step_{}] {} step 1 ####".format(n_step,kw))

        josas_list = bpe_josa_by_len(josa_dict[josa_d[kw][0]],key_vars, n_syllable=1)

        to_check = srch_cond[kw] if kw in srch_cond.keys() else []

        word_to_tune_s,word_to_tune = bpe_find_to_tune_5(
            bpe_vocs,josas_list,key_vars,cutline, num_of_letters, to_check) #vocabs: list of tuple

        if kw == 'ha':
            print("len(word_to_tune) : {}".format(len(word_to_tune)))

            word_to_tune_s,word_to_tune =bpe_find_to_tune_recursive_5(
                word_to_tune_s,josas_list,key_vars,0, num_of_letters) #vocabs: list of tuple

        to_tune = word_to_tune_s

        print("len(word_to_tune) : {}".format(len(word_to_tune)))

        if kw in delete_list.keys():
            for s in delete_list[kw]:
                x = normal_to_special2(s, key_vars)
                for wx in [w for w in to_tune if w[-len(x):]==x]:
                    to_tune.remove(wx)
                    print("deleted word : {}".format(special_to_normal(wx,key_vars)))

        ni = len(word_to_tune)//400 + 1
        print([w for i,w in enumerate(word_to_tune) if i%ni==0])

        tuned_dict[kw+'_c'+str(cutline)+'_n'+str(num_of_letters)] = list(zip(word_to_tune,word_to_tune_s))
        json_save(tuned_dict, path+'tuned_dict') #cutline =4 num of letters = 2 

        tuned_vocabs1, tuned_vocabs2, separated_terms, to_replace = bpe_vocab_tunning5(
            bpe_vocabs, to_tune, josas_list, cutline, key_vars, adj_rate) # 수정 없으면 4로 복귀

        print([len(f) for f in [tuned_vocabs1, tuned_vocabs2, separated_terms, to_replace]])
        #merge 하지 말고 별도 저장할 것 dict로  delete 부분도 별도 dict 만들것
        updated_items, added_items = dict_merge(tuned_vocabs1, tuned_bpe_vocabs)
        updated_items, added_items = dict_merge(tuned_vocabs2, josa_bpe_vocabs)
        updated_items, added_items = dict_merge(to_replace, repl_bpe_vocabs)

        extracted_vocabs[kw+n_step] ={k:{} for k in "keyword_1 josa_1 repl_1 keyword_2 josa_2 repl_2".split(' ')}

        updated_items, added_items = dict_merge(tuned_vocabs1, extracted_vocabs[kw+n_step]['keyword_1'])
        updated_items, added_items = dict_merge(tuned_vocabs2, extracted_vocabs[kw+n_step]['josa_1'])
        updated_items, added_items = dict_merge(to_replace, extracted_vocabs[kw+n_step]['repl_1']) 
        """ 
        extracted_vocabs[kw+n_step]['keyword_1'] += tuned_vocabs1
        extracted_vocabs[kw+n_step]['josa_1'] += tuned_vocabs2
        extracted_vocabs[kw+n_step]['repl_1'] += to_replace
        """    
        deleted_items = dict_subtract(to_replace, bpe_vocabs)  

        for kk in "keyword_2 josa_2 repl_2".split(' '):
            extracted_vocabs[kw+n_step][kk] = {}

        for i in range(1,len(josa_d[kw])):

            """
            sec_sorted_josa = []

            for k,l in bpe_josa_by_len(josa_dict[josa_d[kw][i]],key_vars, n_syllable=0).items():
                sec_sorted_josa+=l
            tuned_vocabs_sec1, tuned_vocabs_sec2, to_replace2 = bpe_vocab_sec_tunning3(
                bpe_vocabs, to_tune, sec_sorted_josa, cutline, key_vars,adj_rate)  
            """

            print("#### start {} step {} ####\n".format(kw, i+1))

            josas_list = bpe_josa_by_len(josa_dict[josa_d[kw][i]],key_vars, n_syllable=1)

            tuned_vocabs1_2, tuned_vocabs2_2, separated_terms, to_replace_2 = bpe_vocab_tunning5(
                bpe_vocabs, to_tune, josas_list, cutline, key_vars, adj_rate) 

            ni = len(tuned_vocabs1_2)//200 + 1
            print([special_to_normal(w,key_vars) for i,w in enumerate(tuned_vocabs1_2) if i%ni==0])   

            updated_items, added_items = dict_merge(tuned_vocabs1_2, tuned_bpe_vocabs)
            updated_items, added_items = dict_merge(tuned_vocabs2_2, josa_bpe_vocabs)
            updated_items, added_items = dict_merge(to_replace_2, repl_bpe_vocabs)

            updated_items, added_items = dict_merge(tuned_vocabs1_2, extracted_vocabs[kw+n_step]['keyword_2'])
            updated_items, added_items = dict_merge(tuned_vocabs2_2, extracted_vocabs[kw+n_step]['josa_2'])
            updated_items, added_items = dict_merge(to_replace_2, extracted_vocabs[kw+n_step]['repl_2'])  

            deleted_items = dict_subtract(to_replace_2, bpe_vocabs)
            """
            extracted_vocabs[kw]['keyword_2'] = tuned_vocabs_sec1
            extracted_vocabs[kw]['josa_2'] = tuned_vocabs_sec2
            extracted_vocabs[kw]['repl_2'] = to_replace2
            """

        #deleted_items = dict_subtract(to_replace2, bpe_vocabs)

        print([len(f) for f in [bpe_vocabs, tuned_bpe_vocabs, josa_bpe_vocabs, repl_bpe_vocabs]])
        if seq < 10:
            sub_fname = 'vocabs_00'+str(seq)+'_after_'+kw+'_c'+str(cutline)+'_n'+str(num_of_letters)
        else:
            sub_fname = 'vocabs_0'+str(seq)+'_after_'+kw+'_c'+str(cutline)+'_n'+str(num_of_letters)


        json_save(bpe_vocabs,path+'bpe_' + sub_fname)
        json_save(tuned_bpe_vocabs,path+'tuned_' + sub_fname)
        json_save(josa_bpe_vocabs,path+'josa_' + sub_fname)
        json_save(repl_bpe_vocabs,path+'repl_' + sub_fname)
        json_save(extracted_vocabs,path+'extracted_vocabs'+date_now)

        new_readable_vocabs, bpe_vocs = bpe_voc_in_normal(bpe_vocabs)
        seq += 1
    
    return extracted_vocabs

 

def make_to_noun_verb_subs(extracted_vocabs):
    Xdata = extracted_vocabs
    p=re.compile(r'[0-9]+$')
    to_X = {}
    noun_W = {}
    for k in Xdata.keys():
        if p.sub('',k) in ['dangha', 'nt', 'n', 'nn', 'ha']:
            for kj in ['josa_1','josa_2']:
                to_X[k+'_'+kj] = Xdata[k][kj]
            for kk in ['keyword_1','keyword_2']:
                noun_W[k+'_'+kk] = Xdata[k][kk] 

    sub_W = {}
    verb_W = {}
    for k in Xdata.keys():
        if p.sub('',k) in ['ha_ee', 'ae_eh', 'uht', 'aht', 'ah_uh', 'ht', 'eu_eu']:
            for kj in ['josa_1','josa_2']:
                sub_W[k+'_'+kj] = Xdata[k][kj]
            for kk in ['keyword_1','keyword_2']:
                verb_W[k+'_'+kk] = Xdata[k][kk] 
    """
    to_extract = {}
    nouns = {}
    verbs = {}
    subs = {}
    """
    mkeys = ['to_extract', 'nouns','verbs','subs']
    make_to = {k:{} for k in mkeys}
    make_from = [to_X, noun_W, verb_W, sub_W]

    for i,dcts in enumerate(make_from):
        for k in dcts.keys():
            aa,bb = dict_merge(dcts[k],make_to[mkeys[i]])
    
    return make_to


def insert_dct(item,dct):  #item:tuple
    if item[0] in dct.keys():
        dct[item[0]] += item[1]
    else:
        dct[item[0]] = item[1]
def subtrct_dct(item,dct):
    if item[0] in dct.keys():
        del dct[item[0]]



def tuning_NVS(make_to):
    
    import copy
    
    key_vars = make_key_vars()
    dkeys = ['to_extract', 'subs', 'nouns', 'verbs']


    top_nk = {}
    top_vk = {}
    top_k = [top_nk, top_vk]

    mid_k = {}
    sub_k = {}

    klist = {}
    klist['to_extract'] = "당하이ㅓㅆ 당하ㅣㅆ 당하이ㅓ 당하ㅣ 당하 하이ㅓㅆ 하ㅣㅆ 하이ㅓ 하ㅣ 하 되었 됐 되어 돼 되 시키ㅓㅆ 시키ㅓ 시키 들".split(' ') 
    klist['to_extract'] += "이었 이ㅓㅆ 이어 이ㅓ 이".split(' ')
    klist['subs'] = '들 이었 이ㅓㅆ 이어 이ㅓ 이 었 았 ㅓㅆ ㅏㅆ ㅣㅆ ㅆ 어 아 ㅓ ㅏ ㅣ'.split(' ')
    klist['verbs'] = klist['to_extract'].copy() + ['게','받','시키ㅓㅆ','시키ㅓ','시키']
    klist['nouns'] = '들'.split(' ')


    for nk in dkeys[:2]:

        X = copy.deepcopy(make_to[nk])
        print(len(X))

        for w in klist[nk]:
            print(w)
            ck = normal_to_special(w,key_vars)
            cl = len(ck)

            XX = [(k,v) for k,v in X.items() if k[:cl] == ck]
            if w =='들':
                print([(special_to_normal(s, key_vars),v) for s,v in XX])        

            for x,v in XX:
                insert_dct((x[:cl],v),mid_k)
                insert_dct((x[cl:],v),sub_k)
                subtrct_dct((x,v),X)

            print(len(X)) 

        aa,bb = dict_merge(X,sub_k)

    to_check = "보이 붙이 모이 벌리 놓이 줄이 높이 들이 쓰이 싸이 죽이 덮이 섞이 쌓이 묶이 기울이".split(' ') 
    to_check += "먹이 녹이 짜이 꺾이 씌이 꼬이 망설이 고이 늘이 트이 깎이 조이".split(' ')
    to_check = [normal_to_special(s, key_vars) for s in to_check]

    for i,nk in enumerate(dkeys[2:]):

        X = copy.deepcopy(make_to[nk])
        print(len(X))

        for w in klist[nk]:

            ck = normal_to_special(w,key_vars)
            cl = len(ck)

            XX = [(k,v) for k,v in X.items() if k[-cl:] == ck]
            if w =='이':
                print([(special_to_normal(s, key_vars),v) for s,v in XX])

            for x,v in XX:
                if w!= '이':
                    insert_dct((x[:-cl],v), top_k[i])
                    insert_dct((x[-cl:],v), mid_k)
                    subtrct_dct((x,v),X)
                else:
                    if x[-4:] in to_check or x[-5:] in to_check or x[-7:] in to_check or x[-8:] in to_check:
                        continue
                    else:
                        insert_dct((x[:-cl],v), top_k[i])
                        insert_dct((x[-cl:],v), mid_k)
                        subtrct_dct((x,v),X)

            print(len(X)) 

        aa,bb = dict_merge(X,top_k[i])
    
    return top_nk, top_vk, mid_k, sub_k

def save_extracted_vocs(top_nk, top_vk, mid_k, sub_k):
    #path0 = '/home/john/Notebook/preProject'
    path = path0+path2 #'/NMT_new/folder_Utils/test_data/'
    
    extracted_vocs = {'nouns' : top_nk, 'verbs':top_vk, 'mids': mid_k, 'subs':sub_k}
    json_save(extracted_vocs,path+'extracted_vocabs_nvs')
    
    return extracted_vocs

  

def make_new_vocs(extracted_vocs):
    #path0 = '/home/john/Notebook/preProject'
    #path = path0+path2 #'/NMT_new/folder_Utils/test_data/'
    
    #srvvd_bpe = json_read('./generated_Data11/preproc_0311/bpe_vocabs_027_after_ha_c0_n1.json')
    srvvd_bpe = json_read(path+ 'bpe_vocabs_020_after_nt_c0_n1.json')
    new_vocs = {}
    for k,dct in extracted_vocs.items():
        aa,bb = dict_merge(dct,new_vocs)
    new_vocs_small = copy.deepcopy(new_vocs)
    new_vocs_all = copy.deepcopy(srvvd_bpe)
    aa,bb = dict_merge(new_vocs_small,new_vocs_all)
    
    return new_vocs_small, new_vocs_all


def make_my_vocabs():

    extracted_vocabs = extract_nouns_verbs_subs()
    make_to = make_to_noun_verb_subs(extracted_vocabs)
    top_nk, top_vk, mid_k, sub_k = tuning_NVS(make_to)
    extracted_vocs = save_extracted_vocs(top_nk, top_vk, mid_k, sub_k)

    #path = path0+path2
    #extracted_vocs = json_read(path+'extracted_vocabs_nvs.json')
    new_vocs_small, new_vocs_all = make_new_vocs(extracted_vocs)
    with open(path+"new_vocs_small.json","w") as f:
        f.write(json.dumps(new_vocs_small))
    with open(path+"new_vocs_all.json","w") as f:
        f.write(json.dumps(new_vocs_all))
    
    return new_vocs_small, new_vocs_all

if __name__ == '__main__':
    
    #extract_nouns_verbs_subs()
    #with open(path+"extracted_vocabs2021_05_14.json","r") as f:
    #    extracted_vocabs = json.load(f)
    make_my_vocabs()
              
    
    
    
    
    
