
import pandas as pd
import copy
import time
import re
from counter_utils import json_save, json_read, normal_to_special, special_to_normal, dict_merge
from utils_for_modify import vocab_adjust, special_to_normal3, vocab_split
from xutils_for_key_vars import make_key_vars  #clean !!
from tqdm import tqdm

def save_sents(sentences,fn,save_op):
    p=re.compile(r'\.$')
    q = re.compile(r'\s+')
    z = re.compile(r'_ ') #영어로 번역할 때는 '_' 무시
    u = re.compile(r'(?P<to_fix>[0-9]+)')
                 
    sents = []
    for s in sentences:
        sents.append(q.sub(' ',u.sub(' \g<to_fix> ',z.sub(' ',p.sub(' .',' '.join(s))))))
        
        #sents.append(z.sub(' ',q.sub(' ',p.sub(' .',' '.join(s)))))
        
    to_save = '\n'.join(sents)   

    with open(fn,save_op) as f:
        f.write(to_save)  
    
    return to_save

def dict_max(src1,src2):
    import copy
    tgt = copy.deepcopy(src2)
    for k,v in src1.items():
        if k in tgt.keys():
            tgt[k] = max(v,tgt[k])
        else:
            tgt[k] = v
    return tgt

def modify_ko_vocs():
    key_vars = make_key_vars()
    #vocabs = to_get_vocabs2()
    date_now = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    path = "./data/"

    new_vocs = json_read('./data/out_data/new_vocs_small.json')
    new_vocs_small = new_vocs

    extracted_vocs = json_read('./data/out_data/extracted_vocabs_nvs.json')


    to_adjust = ['스러우','스럽','시끄러우','따가우','까끄러우','안쓰러우','부드러우','껄끄러우','보드라우','어려우','싱그러우','부끄러우','무서우']
    to_adjust += ['움직이','반짝이','담그',',담기','결정지','가지','데치','무치','걷히','열리','사라지','덧붙이','오르','줄어드']
    to_adjust += ['치러지','내리','거세지','기울이','깨우치','이어지','놓이','만들','들이받','머뭇거리','끈질기','빠지','졸리','견디']
    to_adjust += ['어긋나','뒤늦','엉기','엉키','고꾸라지','돌이키','모이','지피']
    to_adjust_2 = ['테크노','밸리','피플','플랫','사커','큐빅','클린','베이','레이서','유니버시티','도널드','메타포','데일리','잉글리시']
    to_adjust_2 += ['슬리핑','숄더','도그','코펜하겐','며느리','종로','국영','민영','고법','병원','임시','방공식별','김정은','겨자']
    to_adjust_2 += ['훈','관','영','근','인','순', '그들', '이들','말치레']
    to_adjust_2 += ['국제','식민','콰이어','코러스','최고','최저','최소','최대','별도','통째','일련','애잔','예비','즉각','어처구니','휴양']
    to_adjust_3 = ['일수록','므로', '나마','야말로','자마자','만큼','토록']
    to_skip = ['구한말']
    to_split = {'자가':'자 가','자의':'자 의','제로':'제 로','성과':'성 과','적인':'적 인','ㅁ을':'ㅁ 을', '도가':'도 가',
                '세에':'세 에','위에':'위 에','수의':'수 의','력이':'력 이','권에':'권 에','ㅁ이':'ㅁ 이','결의':'결 의',
                '양이':'양 이','상이':'상 이','진이':'진 이','특유의':'특유 의','가의':'가 의','차로':'차 로','사가':'사 가'
            }
    to_adjust = list(set(to_adjust))
    to_adjust_2 = list(set(to_adjust_2))

    vocabs = new_vocs

    # new_vocs_small 을 적용한 것은 vocabs 를 적용하는 경우에 비해 교란 요인을 줄이기 위함 임
    _, nouns, verbs, subs = vocab_adjust(new_vocs_small,to_adjust, 0.9,to_skip,to_match=2) #to_match:'스러우' 등 중간에 들어가는 표현 수

    dkey = [ 'nouns', 'verbs', 'subs']
    for i,dct in enumerate([nouns, verbs, subs]):
        _,_ = dict_merge(dct,extracted_vocs[dkey[i]])
        
    _, _, nouns, _ = vocab_adjust(new_vocs_small,to_adjust_2, 0.5)
    _, _, subs, _ = vocab_adjust(new_vocs_small,to_adjust_3, 0.7, to_match=len(to_adjust_3))

    dkey = [ 'nouns', 'subs']
    for i,dct in enumerate([nouns, subs]):
        _,_ = dict_merge(dct,extracted_vocs[dkey[i]])
        
        
    ############# Total Vocabs => 50000 + epsilone + single_words

    vocabs = {k:v/100. for k,v in vocabs.items() if v>300}

    #extracted_vocs['nouns'] = {k:v for k,v in extracted_vocs['nouns'].items()}
    vocabs.update(extracted_vocs['nouns'])

    vocabs.pop('',1)

    new_X = copy.deepcopy(vocabs)
    for k,v in new_X.items(): 
        normX = special_to_normal3(k,key_vars,keep_double=False)
        if len(normX)>1 and normX[-1] in ['제','의','로','이','은','도']:
            vocabs[k] = vocabs[k] /10.
        if ord(normX[0]) in range(12593, 12644) or ord(normX[-1]) in range(12593, 12644):
            vocabs.pop(k,1)

    for k in ['ㄴ다','ㄴ다는','ㄴ지','ㄴ데','ㄴ다면','ㄴ지는','ㅁ으로써','ㅁ에','ㅁ을','ㄹ지', 'ㄹ지는','ㅣ','ㅣㅆ','ㅆ다', 'ㅆ는','여','였']:
        extracted_vocs['subs'].pop(normal_to_special(k,key_vars),1)
        extracted_vocs['mids'].pop(normal_to_special(k,key_vars),1)

    extracted_vocs['verbs'] = {k:v*5 for k,v in extracted_vocs['verbs'].items()} 
    extracted_vocs['subs'] = {k:v*2 for k,v in extracted_vocs['subs'].items()} 
    extracted_vocs['mids'] = {k:v*2 for k,v in extracted_vocs['mids'].items()} 

    vocabs.update(extracted_vocs['verbs'])
    vocabs.update(extracted_vocs['subs'])
    vocabs.update(extracted_vocs['mids'])

    for k in ['내려','담겨있','후보이','기에']:
        vocabs.pop(normal_to_special(k,key_vars),1)
        
    vocabs.pop('',1)    
    new_X = copy.deepcopy(vocabs)
    for k,v in new_X.items(): 
        normX = special_to_normal3(k,key_vars,keep_double=False)
        if normX[0] in ['눠','춰','눴','췄','봐','봤','였']:
            vocabs[k] = vocabs[k] /100.
        if len(normX)>1 and normX[0] in '않 앉 없 있 받 싶 찾 잦 갚'.split(' '):
            vocabs[k] = vocabs[k] /100.

    single_words = json_read('./data/ko_single_words.json')
    vocabs.update({normal_to_special(k,key_vars):max(v,100) for k, v in single_words.items() 
                if ord(k) not in range(12593, 12644)})

    vocabs.pop('',1)

        
    for k in 'ㄴ ㄹ ㅁ 쳐'.split(' '):
        vocabs[normal_to_special(k,key_vars)] = vocabs[normal_to_special(k,key_vars)] /10.

    #############


    w_to_tune = '가 이 는 여 를 론 일 있 않 앉 없 받 싶 순 찾 갖 맞 잦 갚 녹 간 주의 김 성장 과제 절실 도시 이유 특유 거뜬 ㅅ'.split(' ')
    w_to_tune += "전략 매우 다소 아름 모든 되레 이용 대거 나름대로 어떻게 더욱 역시 결정 아들 직접 간접 마음껏 결코 통째 흔히".split(' ')
    w_to_tune += "이미 미리 반면 여러 임시 바로 종일 저절로 따로 즉각 아무리 다시 또한 아울러 절레 착한 개관".split(' ')
    w_to_tune += "과연 조만간 점점 오로지 어쩌면 오히려 전혀 또는 그냥 재차 이를테면 예를들면".split(' ')

    w_to_tune += ['제'+str(i) for i in range(10)]
    w_to_tune = list(set(w_to_tune))

    for w in w_to_tune:
        ws = normal_to_special(w,key_vars)
        #print(ws, w)
        if ws in vocabs.keys():
            vocabs[ws] = max(1000,vocabs[ws]*30) 
        else:
            vocabs[ws] = 1000 
    for w in ['로','의']:
        ws = normal_to_special(w,key_vars)
    
    vocabs.pop('',1)
    _ = vocab_split(vocabs,to_split, 1.0)

    vocabs.pop(normal_to_special('러운',key_vars),1)
    vocabs['NotInVocabs'] = 1

    json_save(extracted_vocs,path+'modified_extracted')
    json_save(vocabs,path+'vocabs')

if __name__ == "__main__":
    modify_ko_vocs()