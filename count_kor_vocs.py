import json
import re 
import pandas as pd
from collections import Counter
from counter_utils import json_save


"""
    초성 중성 종성 분리 하기
	유니코드 한글은 0xAC00 으로부터
	초성 19개, 중성21개, 종성28개로 이루어지고
	이들을 조합한 11,172개의 문자를 갖는다.
	한글코드의 값 = ((초성 * 21) + 중성) * 28 + 종성 + 0xAC00
	(0xAC00은 'ㄱ'의 코드값)
	따라서 다음과 같은 계산 식이 구해진다.
	유니코드 한글 문자 코드 값이 X일 때,
	초성 = ((X - 0xAC00) / 28) / 21
	중성 = ((X - 0xAC00) / 28) % 21
	종성 = (X - 0xAC00) % 28
	이 때 초성, 중성, 종성의 값은 각 소리 글자의 코드값이 아니라
	이들이 각각 몇 번째 문자인가를 나타내기 때문에 다음과 같이 다시 처리한다.
	초성문자코드 = 초성 + 0x1100 //('ㄱ')
	중성문자코드 = 중성 + 0x1161 // ('ㅏ')
	종성문자코드 = 종성 + 0x11A8 - 1 // (종성이 없는 경우가 있으므로 1을 뺌)
"""

# 유니코드 한글 시작 : 44032, 끝 : 55199
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', '']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
ch_len = max([len(s) for s in [CHOSUNG_LIST,JUNGSUNG_LIST, JONGSUNG_LIST]])

len_alpha =86+BASE_CODE
ch_start = chr(len_alpha-2)
ch_end = chr(len_alpha-1)

def get_text(file, f_list=True):
    if f_list:
        with open(file, 'r') as f:
            data = f.read()
    else:
        data = file
    p = re.compile(r'\<.*\>')
    q = re.compile(r'[^0-9a-zA-Z가-힣\-]')
    d_out = p.sub(' ', data)
    d_out = q.sub(' ', d_out)
    return re.sub('\s{1,}',' ', d_out)

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

def count_from_file(file_list):
    c = Counter()
    for f in file_list:
        d_out =  get_text(f)
        c = c + Counter(d_out.split(' '))
    return c

def convert(test_keyword):
    split_keyword_list = list(test_keyword)
    #print(split_keyword_list)

    result = list()
    indexed = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
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
            if ord(keyword) <127:
                indexed.append(ord(keyword)+54)  # 한글이 아닌 경우 잠정적으로 +54  숫자, 영문 대소문자 포함
    # result
    return "".join(result), indexed    #indexed는 자음(초성과 종성 구분), 모음 구분하여 인덱스 각 => [ㄱ ㅏ ㄱ], [0,28,57]

"""
def ind_char(idx):
    #######################################################
    #  자소 인덱스를 글자로 변환하여 글자를 식별할 수 있도록 
    #######################################################
    ch_len = 28
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
        w_idx += nc[id//ch_len] * (id%ch_len)    #초,중,종성 구분 + 각 분류에서 몇번째 인지
        n_ind += 1
        if i == len(idx)-1:                      #맨 마지막이면..
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            #n_ind = 0
        elif (idx[i+1]//ch_len==0) or (idx[i+1]>ch_len*3):              #한 글자가 끝나면..
            word_comb.append(chr(w_idx)) if n_ind>1 else word_comb.append(kor_alpha[id])
            w_idx = BASE_CODE
            n_ind = 0

    return ''.join(word_comb) 
"""

def pre_bpe_vocab_initialize(counters):

    vocab = {}
    for k,v in counters:
        _, idx = convert(k)
        vocab[''.join([chr(i+BASE_CODE) for i in idx])] = v 
    return vocab

def vocab_initialize(counters):
    
    """
    띄워쓰기로 분할된 단어(빈도가 카운트된)들을 음절로 분리
    """
    vocab = {}
    #for k,v in counters.items():
    for k,v in counters:
        _, idx = convert(k)
        #vocab[' '.join([ch_start]+[chr(i+BASE_CODE) for i in idx] + [ch_end])] = v 
        vocab[''.join([chr(i+BASE_CODE) for i in idx])] = v 
        
    return vocab

def main():
    path = '/home/john/Notebook/wikiextractor_ko/text/A'
    fileN = '0123456789'
    fileABC = 'ABCDEFG'

    file_list = []
    for a in fileABC:
        for n in fileN:
            if (a=='G') and (n=='5'):
                file_list += [path+a+'/wiki_' + n + s for s in fileN[:9]]
                break
            else: 
                file_list += [path+a+'/wiki_' + n + s for s in fileN]

    counters = count_from_file(file_list)
    #with open("counters_wiki.json", "r") as f:
    #    counters = json.load(f)
    
    """
    sentences =[]
    for f in file_list[:5]:    
        sentences +=  get_sentences(f)
    """

    sorted_count = sorted(counters.items(), key=lambda x:x[1], reverse=True)
    sorted_10 = [x for x in sorted_count if x[1] > 10] 

    vocab = pre_bpe_vocab_initialize(sorted_10)
    #vocab = vocab_initialize(sorted_20)
    
    with open('vocabs_10_check.json', 'w' ) as f:
        f.write(json.dumps(vocab))

def count_en(to_count):
    from glob import glob
    path = ""
    path2 = "data/"
    ko_wk = Counter()
    if 1 in to_count:      
        print('start reading wiki_files')
        files = glob(path+'wikiK/*.txt')
        ko_wk = count_from_file(files)
        """
        for fl in tqdm(files):
            with open(fl,'r') as f:
                X = f.read().split('\n')    
            ko_wk = word_count(X, ko_wk)     
        """        
        print("len(ko_wk) :{}".format(len(ko_wk)))
        json_save(ko_wk,path+path2+'counted_vocs_ko_wk')

    """
    en_news = Counter()
    if 2 in to_count:
        print('start reading en_news_files')      
        for i in tqdm(range(1,4)):
            df = pd.read_csv(path+'newsEn/articles'+str(i)+'.csv') 
            en_news = word_count(df['content'], en_news)
        print("len(en_news) :{}".format(len(en_news)))
        json_save(en_news,path+path2+'counted_vocs_en_news')
    """
    ko_news = Counter()
    if 2 in to_count:
        print('start reading translated_ko_news_files')  
        files = glob(path+'newsKo/*.xlsx')     
        for fn in files:
            df = pd.read_excel(fn)
            parallel = df['ID 원문 번역문'.split(' ')]
            d_out =  get_text(' '.join(list(parallel['원문'])), f_list=False)
            ko_news += Counter(d_out.split(' '))
        print("len(ko_news) :{}".format(len(ko_news)))  
        json_save(ko_news,path+path2+'counted_vocs_ko_news')

    counters = ko_news+ko_wk
    json_save(counters,path+path2+'counted_vocs')
    sorted_count = sorted(counters.items(), key=lambda x:x[1], reverse=True)
    sorted_10 = [x for x in sorted_count if x[1] > 10] 

    vocab = pre_bpe_vocab_initialize(sorted_10)
    #vocab = vocab_initialize(sorted_20)
    
    with open(path2+'vocabs_10.json', 'w' ) as f:
        f.write(json.dumps(vocab))

if __name__ == '__main__':
    to_count = [1,2]
    count_en(to_count)
