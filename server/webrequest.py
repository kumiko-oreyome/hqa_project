import re
import time
import asyncio
import httpx
from urllib.parse import quote
from  jieba import analyse
from lxml import etree,html
from pyppeteer import launch


GOOGLE_SEARCH_URL = 'https://www.google.com/search' 

class Downloader():
    def __init__(self,timeout=5000,max_try=5):
        self.timeout = timeout
        self.max_try = max_try
    def get(self,url):
        try:
            r = httpx.get(url,timeout=self.timeout)
        except httpx.exceptions.ReadTimeout:
            print('timeout url is %s'%(url))
        return r.text
    def post(self,url):
        pass

    async def  async_get(self,url):
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(url,timeout=self.timeout)
            except httpx.exceptions.ReadTimeout:
                print('timeout url is %s'%(url))
  
            
        return r.text



class Query():
    def __init__(self,query,word_dict=[]):
        self.query = query
        self.word_dict = word_dict


    def extract_keywords(self,k=3):
        return analyse.extract_tags(self.query,topK=3)

    def extract_keywords_with_diciotnary(self):
        keywords = []
        for w in  self.word_dict:
            if w in self.query:
                keywords.append(w)
        return keywords

class HealthArticlePage():
    def __init__(self,html_text):
        self.html_text = html_text
        self.tree = html.fromstring(html_text)
    def get_title(self):
        ele = self.tree.xpath("//h1[@class='title']")[0]
        return ele.text
    def get_body(self):
        body_ele = self.tree.xpath("//div[@class='essay']")[0]
        return html.tostring(body_ele,pretty_print=True).decode()
    def to_json(self):
        return {'body':self.get_body(),'title':self.get_title()}

class HealthArticleRequest():
    def __init__(self,url,event_loop=None):
        self.url = url
        self.article_id = self.parse_nid() 
        self.event_loop = event_loop

    def parse_nid(self):
        m = re.search(r'article\.action\?nid=(\d+)',self.url)
        if m is None:
            return None
        return m.group(1)

    def send(self):
        assert self.article_id is not None
        html = Downloader().get(self.url)
        page = HealthArticlePage(html)
        return page

    async def _async_send(self):
        assert self.article_id is not None
        html = await Downloader().async_get(self.url)
        page = HealthArticlePage(html)
        return page

    async def _async_send_render(self):
        assert self.article_id is not None
        browser = await launch()
        page = await browser.newPage()
        await page.goto(self.url)
        html = await page.content()
        return HealthArticlePage(html)

    def async_send_render(self):
        loop = asyncio.get_event_loop()
        page = loop.run_until_complete(self._async_send_render())
        return page

    def async_send(self):
        if self.event_loop is None:
            self.event_loop = asyncio.get_event_loop()
        page = self.event_loop.run_until_complete(self._async_send())
        return page



class GoogleSearchResultPage():
    def __init__(self,html_text):
        self.html_text = html_text
        self.tree = html.fromstring(self.html_text)
    def get_result_links(self):
        a_elememts = self.tree.xpath("//div[@id='search']//div[@class='rc']/div[@class='r']/a")
        links = []
        for a in a_elememts:
            links.append(a.get("href"))
        print('len of links %d'%(len(links)))
        return links
    def get_next_page_link(self):
        e = self.tree.xpath("//div[@id='foot']//a[@class='pn']")
        if len(e) == 0:
            return None
        return '%s%s'%('https://www.google.com',e[0].get("href"))
    

class GoogleSearchRequest():
    def __init__(self,keywords=[],site_url=None,loop=None):
        if type(keywords) == str:
            self.keywords = [keywords]
        self.keywords = keywords
        self.site_url =site_url
        self.url = self.get_url()
        self.loop = loop

    def join_keywords_to_query(self,keywords):
        return '+'.join(keywords)

    def get_url(self):
        query = self.join_keywords_to_query(self.keywords)
        if self.site_url is not None:
            site_part = quote('site:%s'%(self.site_url))
            query = self.join_keywords_to_query([site_part,query])
        return '%s?q=%s'%(GOOGLE_SEARCH_URL,query)

    def send(self):
        html = Downloader().get(self.url)
        #with open('aaa.html','w',encoding='utf-8') as f:
        #    f.write(html)
        return GoogleSearchResultPage(html)
    async def _async_send(self):
        browser = await launch(   handleSIGINT=False,handleSIGTERM=False,handleSIGHUP=False)
        page = await browser.newPage()
        await page.goto(self.url)
        html = await page.content()
        return GoogleSearchResultPage(html)

    def async_send(self):
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        return self.loop.run_until_complete(self._async_send())

class YahooAnswerQuestionPage():
    def __init__(self,html_text):
        self.html_text = html_text
        self.tree = html.fromstring(html_text)
    def get_title(self):
        ele = self.tree.xpath("//h1[contains(@class,'Question__title___')]")[0]
        return ele.text
    def get_body(self):
        body_ele = self.tree.xpath("//div[contains(@id,'qnaContainer-')]")
        if body_ele is None:
            return None
        body_ele = body_ele[0]
        return html.tostring(body_ele,pretty_print=True).decode()
    def to_json(self):
        return {'body':self.get_body(),'title':self.get_title()}

class YahooAnswerQuestionRequest():
    def __init__(self,url,event_loop=None):
        self.url = url
        self.event_loop = event_loop
       
    def send(self):
        html = Downloader().get(self.url)
        page = YahooAnswerQuestionPage(html)
        return page

    async def _async_send(self):
        html = await Downloader().async_get(self.url)
        page = YahooAnswerQuestionPage(html)
        return page

    #async def _async_send_render(self):
    #    browser = await launch()
    #    page = await browser.newPage()
    #    await page.goto(self.url)
    #    html = await page.content()
    #    return YahooAnswerQuestionPage(html)

    #def async_send_render(self):
    #    loop = asyncio.get_event_loop()
    #    page = loop.run_until_complete(self._async_send_render())
    #    return page

    def async_send(self):
        if self.event_loop is None:
            self.event_loop = asyncio.get_event_loop()
        page = self.event_loop.run_until_complete(self._async_send())
        return page


class CMKBLibraryPage():
    def __init__(self,html_content):
        self.html_content = html_content
        self.tree = html.fromstring(html_content)
    def get_body(self):
        body_ele = self.tree.xpath("//div[@class='nm-post-content']/article")[0]
        return html.tostring(body_ele,pretty_print=True).decode()

    def get_title(self):
        ele = self.tree.xpath("//div[@class='nm-post-content']/article//div[@class='text-content']/h2")[0]
        return ele.text

    

class CMKBKeywordSearchPage():
    def __init__(self,keyword,loop):
        self.keyword = keyword
        self.url = self.get_keyword_serch_url(keyword)
        self.loop = loop
        self.page = self.loop.run_until_complete(self.open_keyword_serch_page())
        self.current_page_infos = []

        
    @classmethod
    def  get_keyword_serch_url(cls,keyword):
        return 'https://kb.commonhealth.com.tw/library/search?keyword=%s'%(keyword)

    @classmethod
    def get_more_btn_xpath(cls):
        return "//div[@class='resultList-box']/div[@class='box-btn']/a[@class='btn']"

    @classmethod  
    def _get_result_item_xpath(cls):
        return "//div[@class='box-data']/div[@class='result-item']"

    @classmethod  
    def keyword_search_request(cls,keyword):
        html_content = Downloader().get(cls. get_keyword_serch_url(keyword))
        return cls.get_library_page_infos(html_content)
    @classmethod 
    def get_library_page_infos(cls,html_content):
        html_content = html_content
        tree = html.fromstring(html_content)
        l   = []
        for node in tree.xpath(cls._get_result_item_xpath()):
            link = node.xpath('./a/@href')[0]
            title = node.xpath('./a/h3/text()')[0]
            tags =  node.xpath('./a/p/text()')[0]
            l.append({'url':link,'title':title,'tags':[tags]})
        return l

    
    def update_page_infos(self,new_page_infos):
        current_n  = len(self.current_page_infos)
        newer_n = len(new_page_infos)
        assert newer_n >= current_n
        self.current_page_infos.extend(new_page_infos[current_n:])

    async def  expand_page_until(self,expand_num=1000):
        for _ in range(expand_num):
            res = await self._get_more_items_by_click_btn()
            if res is None:
                break
            print('expand page : current item num %d'%(len(self.current_page_infos)))
        print('expaned : total %d'%(len(self.current_page_infos)))


    async def open_keyword_serch_page(self):
        browser = await launch({'headless': True,'timeout':1000*360})
        page = await browser.newPage()
        await page.goto(self.url)
        await page.setJavaScriptEnabled(enabled=True)
        return page


    async def _parse_current_page(self):
        page_infos = await self.page.evaluate("() =>{\
             let l = [];\
             for(let node of document.getElementsByClassName('result-item')){ \
                   let url = node.getElementsByTagName('a')[0].href;\
                   let title =   node.querySelector('a h3').innerText;\
                   let tags =   node.querySelector('a p').innerText;\
                   l.push({url:url,title:title,tags:[tags]});\
                 }return l;}")
        return page_infos

    async def _get_more_items_by_click_btn(self):
        #if self.first_seen:
        #    self.first_seen = False
        #    await self.page.setRequestInterception(True)
        #    async def intercept(request):
        #        if not request.url.startswith('https://kb.commonhealth'):
        #            await request.abort()
        #        else:
        #            print(request.url)
        #            await request.continue_()
        #    self.page.on('request', lambda req: asyncio.ensure_future(intercept(req)))
        #    self.page.on('response', lambda resp: asyncio.ensure_future(self.xhr_handler(resp)))
        flag = await self.page.evaluate("() =>{if(document.querySelector('div.box-btn a.btn').style.display=='none'){return false;}document.querySelector('div.box-btn a.btn').click();return true;}")
        if not flag:
            return None
        await self.page.waitFor(1000)
        page_infos = await self._parse_current_page()
        self.update_page_infos(page_infos)
        return page_infos
        
    async def xhr_handler(self,resp):
        print( resp.request.url)
        # = await resp.text()
        #("tmp.html",'w',encoding='utf-8').write(text)
        if 'https://kb.commonhealth.com.tw/library' in resp.request.url:
            print('on xhr handle')
            print( await resp.json())
    

class CMKBRequest():
    @classmethod
    def html_parsing_paragraph(cls,html_text):
        def remove_white_spaces(texts):
            texts = list(map(lambda  x: x.rstrip().lstrip(),texts))
            texts = list(map(lambda  x: re.sub("[\n\t\r]","",x),texts))
            texts = list(filter(lambda  x: len(x)>0,texts))
            return texts

        parser = etree.HTMLParser(remove_blank_text=True)
        tree = etree.HTML(html_text,parser)
        
        type1_paragraphs = []
        type2_paragraphs = []
        for pnode in tree.xpath("//div[@class='text-content']/p"):
            if pnode.text is None:
                continue
            type1_paragraphs.append(pnode.text)
        
        for node in tree.xpath("//div[@class='post-tab']/div[@class='panel-group']//div[@class='panel panel-default']"):
            title = node.xpath(".//h4[@class='panel-title']")[0].text
            if title == '貼心提醒':
                continue
            body = "".join(node.xpath(".//div[@class='panel-body']//text()"))
            type2_paragraphs.append('%s\n%s'%(title,body))
        return remove_white_spaces(type1_paragraphs+type2_paragraphs)



    def __init__(self,loop=None):
        self.loop = loop
    def  get_library_page(self,url):
        html_content = Downloader().get(url)
        return CMKBLibraryPage(html_content)

    def crawl_keyword_search_page(self,keyword,click_more_num=1000):
        from .dao import cmkb 
        assert self.loop is not None
        async def a ():
            await page.expand_page_until(click_more_num)
        print('create keyword search page')
        page = CMKBKeywordSearchPage(keyword,self.loop)
        self.loop.run_until_complete(a())
        doc_jsons = []
        for item in page.current_page_infos:
            url,tags,title = item["url"], item["tags"], item["title"]
            print('request : %s --> %s'%(title,url))
            lib_page = self.get_library_page(url)
            time.sleep(2)
            paragraphs = self.html_parsing_paragraph(lib_page.get_body())
            doc = cmkb.CMKBDLibraryDocument(url=url,title=lib_page.get_title(),body=lib_page.get_body(),tags=tags,paragraphs=paragraphs)
            doc_jsons.append(doc.to_json())
        return doc_jsons
        
