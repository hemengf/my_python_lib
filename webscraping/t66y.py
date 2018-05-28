# -*- encoding: utf-8 -*-
import urllib
import cfscrape
from bs4 import BeautifulSoup
import re
n = 1
f = open('result.html','w+')
f.write('<!DOCTYPE html>')
f.write('<html>')
f.write('<body>')
for page in range(1,50):
    site15 ="http://t66y.com/thread0806.php?fid=15&search=&page=%d"%page
    site2 ="http://t66y.com/thread0806.php?fid=2&search=&page=%d"%page
    site4 ="http://t66y.com/thread0806.php?fid=4&search=&page=%d"%page
    site8 ="http://t66y.com/thread0806.php?fid=8&search=&page=%d"%page
    site7 ="http://t66y.com/thread0806.php?fid=7&search=&page=%d"%page
    for site in [site15, site2,site4,site8,site7]:
    #for site in [site7]:
        scraper = cfscrape.create_scraper()  # returns a CloudflareScraper instance
        # Or: scraper = cfscrape.CloudflareScraper()  # CloudflareScraper inherits from requests.Session
        html = scraper.get(site)
        soup = BeautifulSoup(html.content,'html.parser')
        trs = soup.findAll('tr',{'class','tr3 t_one tac'},limit=None)

        for tr in trs[3:]:
            url =  'http://t66y.com/'+tr.find('td',{'class','tal'}).find('a').get('href')
            s = tr.find('td',{'class','tal'}).get_text().encode('utf8')
            keywords = ['Beginningofkeywords',\
                    '佐々木',\
                    '佐佐木',\
                    'sasaki',\
                    'Sasaki',\
                    '白木',\
                    '松岡　ちな',\
                    '春原未来',\
                    'Chanel',\
                    'Karla Kush',\
                    'pantyhose',\
                    'Pantyhose',\
                    'Stockings',\
                    '絲襪',\
                    '丝袜',\
                    '黑丝',\
                    '襪',\
                    '小島',\
                    '神纳花',\
                    #'FHD',\
                    'EndofKeywords'\
                    ]
                    
            for keyword in keywords:
                if keyword in s:
                    linktext = '<a href="{x}">{y}</a>'.format(x=url,y=s)
                    print linktext
                    f.write('<p>'+linktext+'</p>')
                    #print(s),url,'page =',page,'fid =',site[site.index('=')+1:site.index('&')]
            #print n
            n+=1


f.write('</body>')
f.write('</html>')
f.close()
