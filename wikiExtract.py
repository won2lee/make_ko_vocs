import argparse
#import sys
import os
from os.path import join, exists
import urllib.request
import re
import random

from xml.etree import ElementTree as ET
import re

import wget
import bz2

#p = re.compile('^\s*[\|\!\/\:\;\*\,\{\#]')
p0 = re.compile('\s*[\|\!\/\:\;\*\,\{\#\<]') #'\;\#]')
p1 = re.compile('\<[Rr]ef(?!\<\/ref).*\<\/ref\>')
p3 = re.compile('\{\{[^\}\{]{1,100}\}\}')
p5 = re.compile('\{\{[^\}\{]{1,500}\}\}')
p7 = re.compile('\<div(?!\<\/div).{1,300}\<\/div\>')
p9 = re.compile('\[http[^\]]+\]')
p11 = re.compile('\<\!\-\-(?!\-\-\>).*\-\-\>')
#p13 = re.compile('\[\[[^\]\|]+\]\]')
p15 = re.compile('\<nowiki(?!\<\/).*\<\/nowiki\>')
p17 = re.compile('\[\[([^\|\]]+\|)*(?P<t17>[^\]]{1,120})\]\]')
p19 = re.compile('\&nbsp\;')
p21 = re.compile('\{\{\s*convert\|(?P<t21>[0-9]+)\|(?P<t21_1>[^\|]+)\|[^\}]+\}\}')
p23 = re.compile("\'\'\'")
p25 = re.compile("\'\'")
p27 = re.compile('\<ref(?!\/\>).*\/\>')
p29 = re.compile('[0-9\,\.\-\+\=\*]{30,}')
p31 = re.compile('\{\{\#tag(?!\}\}).*(\}\})*')
p33 = re.compile('\<\/*blockquote\>')
p34 = re.compile('(?P<t34>(al|etc)\.)\s') #보호 문자 첨부 ħ
p35 = re.compile('(?P<t35>[a-z가-힣][a-z가-힣\)][\.\?\!][\"]*)\s+')
p36 = re.compile('ħ') # 보호 문자 해제 ħ
p37 = re.compile('(COMMENT\:*|Ris\s*[0-9][0-9]|File\:|Bot\:)')  #패러그랩 단위에서 체크
p39 = re.compile('(\(UTC\)|\{\{|\}\}|\<\/*math\>)') # 문장단위에서 체크
#p39 = re.compile('\(talk\)(?!\(UTC\)).*\(UTC\)') # 문장단위에서 체크
p41 = re.compile('(?P<t41a>[0-9][0-9])\.\s(?P<t41b>[A-Z])')
p43 = re.compile('[\.\?\!]\"*$')
p45 = re.compile('\<ref.*')
p46 = re.compile('\<\/*[suipU]\>')
p47 = re.compile('\<\/*(sub|em|code|sup|ref|REF|SMALL|small|big|br|nowiki|span|var|tt|kbd|chem|cite|i|abbr|syntax)[^\>]*\>')
p48 = re.compile('[\[\]]')
#p47 = re.compile('(\<\/*[su]\>|[\[\]]|\<\/*sub\>|\<\/*em\>|\<\/*code\>|\<\/*sup\>|\<\/*ref\>|\<\/*SMALL\>)')
p49 = re.compile('https*\:[^\s]+')
p51 = re.compile('(\(\s*\,[^\)]*\)|\([^\,]*\,\s*\)|\(\s*\))')
p53 = re.compile('[^\s]{30,}')
p55 = re.compile('File\:')


z = re.compile('\s+')

edW = """Hello Welcome Hi Wikipedia wikipedia You you I My my page Edit edit Article article submission 
Please please Thank thank Typography Font font redirect Categor Redirect User user talk 
log Talk Remov remov message delet website am"""
edW = z.sub(' ',edW).split(' ')
p_m = [re.compile(w) for w in edW]

edWK = """위키 백과 문서 편집 작성 관리자 사용자 니다"""
edWK = z.sub(' ',edWK).split(' ')
p_m2 = [re.compile(w) for w in edWK]


def read_wiki(args,lang):
    url = "https://dumps.wikimedia.org/"+lang+"wiki/latest/"
    file = urllib.request.urlopen(url)
    p = re.compile('.+'+lang+'wiki-latest-pages-articles-multistream(?P<toExt>[0-9]{1,2}\.xml-p[0-9]+p[0-9]+\.bz2)\".+')
    q = re.compile('(?P<indx>[0-9]+)\.xml.*')

    f_toEx = []
    for i,line in enumerate(file):
        decoded_line = line.decode("utf-8")
        if p.search(decoded_line) is not None:
            #print(decoded_line)
            f_toEx.append(p.sub('\g<toExt>', decoded_line)[:-1])

    if args.opt == 0:
        return random.sample(f_toEx, args.num)
    else:
        #print(args.articles)
        return [f for f in f_toEx if q.sub('\g<indx>',f) in args.articles]
        
def main(args):
    lang = args.lang  
    p57 = re.compile('[가-힣\s]') if lang == 'ko' else re.compile('[a-zA-Z\s]')
    #k = 2             #num of files to read
    step =1000
    fstep = 1000*10
    if args.path != '':
        path = args.path
    else:
        path = "wikiE/" if lang =='en' else "wikiK/"
    if path != '' and not exists(path):
        os.makedirs(path)
    srcpath = lang+"wiki-latest-pages-articles-multistream"

    txtX = []
    n_iter = 0*fstep #1
    with open(path+'wiki_'+lang+str(n_iter//fstep)+'.txt','w') as f:
        f.write('')

    wiF = read_wiki(args,lang) #["16.xml-p18960153p20460152"]
    print(f'wiF : {wiF}')
    #sys.exit()
    pppp = "{http://www.mediawiki.org/xml/export-0.10/}"
    txtX = []

    for iw,fw in enumerate(wiF):
        X = ET.parse(bz2.open(wget.download("https://dumps.wikimedia.org/"+lang+"wiki/latest/"+srcpath+fw), mode='rb'))    
        print("start X")
        
        root = X.getroot()
        rows = []

        for doc in root.findall(pppp+"page"):

            n_iter += 1

            if doc.findall(pppp+'revision') is not None:

                for ic,child in enumerate(doc.findall(pppp+'revision')):

                    for gchild in child.findall(pppp+'text'):
                        s = gchild.text
                        if s == '' or type(s) is not str:
                            print(f"s : {s}")
                            continue

                        s = p51.sub('',p49.sub('',p48.sub('',p47.sub('',p46.sub('',
                                p45.sub('',p33.sub('',p31.sub('',p27.sub('',
                                p25.sub('',p23.sub('',p19.sub(' ',
                                p17.sub('\g<t17>',p15.sub('',
                                p11.sub('',p9.sub('',p7.sub('',
                                p1.sub('',p5.sub('',p3.sub('',
                                p21.sub('\g<t21> \g<t21_1>',s)))))))))))))))))))))

                        s = [x for x in s.split('\n')]
                        for x in s:
                            if (p0.match(x) is None and len(x)>200 and p29.search(x) is None and
                                p37.search(x) is None and
                                sum([1 if pp.search(x[:300 if lang == 'en' else 200]) is not None else 0 for pp in p_m]) <3):
                                
                                xx = [s for s in ('\n'+z.sub(' ',p41.sub('\g<t41a>.Ħ\g<t41b>',
                                                    p36.sub('',p35.sub('\g<t35>Ħ',
                                                    p34.sub('\g<t34>ħ ',x)))))
                                                    .strip()).split('Ħ') 
                                        if (len(s)<500 if lang == 'en' else 350) and p39.search(s) is None and  
                                                p53.search(s) is None and p43.search(s) is not None]
                                
                                xx = [s for s in xx if len(p57.findall(s[:100]))/len(s[:100]) >0.7]
                                #print("second starting xx")
                                #print(xx,'\n')
                                xl = []
                                for i,s in enumerate(xx):
                                    if i>0 and len(xl[-1]) + len(s) < 100:
                                        xl[-1]+=' '+s
                                    else:
                                        xl.append(s)
                                #print("starting xl")
                                #print(xl,'\n')

                                txtX += xl               
                                #sum([1 if pp.search(x[:400]) is not None else 0 for pp in p_m]) <3):
                                #print(z.sub(' ',x.strip()), '\n')
                                #txtX.append(z.sub(' ',x.strip()))


            else:
                print("doc is None")


            if n_iter % step == 0:
                #with open('wiki_en.txt','a') as f:
                with open(path+'wiki_'+lang+str(n_iter//fstep)+'.txt','a') as f:
                    f.write('\n'.join(txtX)+'\n')

                txtX = [] 

                if n_iter % fstep == 0:
                    with open(path+'wiki_'+lang+str(n_iter//fstep)+'.txt','w') as f:
                        f.write('')

    with open(path+'wiki_'+lang+str(n_iter//fstep)+'.txt','a') as f:
        f.write('\n'.join(txtX))

if  __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='program to extract sentences from Wikipedia'
    )
    parser.add_argument('--lang', type=str, default = 'ko', help="language,  'ko': korean, 'en': english")
    parser.add_argument('--opt', type=int, default = 0, help='from what index,  0: random, 1: by indexes')
    parser.add_argument('--num', type=int, default = 2, help='num of files to download')
    parser.add_argument('--articles', nargs="+", default = [3], help='article group ids to download')
    parser.add_argument('--path', default= '', help='directory to write output')
    args = parser.parse_args()
    main(args)