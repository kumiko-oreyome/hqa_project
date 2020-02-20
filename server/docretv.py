from lxml import etree,html
from .webrequest import HealthArticleRequest,GoogleSearchRequest,HealthArticleRequest,Query,YahooAnswerQuestionRequest





def create_document_retriever(config):
    class_name,kwargs = config["class"],config["kwargs"]
    name2class = {'FakeRetriever':FakeRetriever,'CMKBElasticSearchRetriever': CMKBElasticSearchRetriever,'GoogleSearchRetriever': GoogleSearchRetriever}
    _cls = name2class[class_name]


class SimpleParagraphTransform():
    def __init__(self):
        pass
    def transform(self,sample_json):
        for doc in sample_json["documents"]:
            paragraphs = get_article_paragraphs_by_p_tags( doc)
            if len(paragraphs) == 0:
                paragraphs = get_article_paragraphs( doc)
            if len(paragraphs) == 0:
                print(doc['title'])
                print('%s paragraph length = 0\n url=%s'%(doc['title'],doc['url']))
                doc['paragraphs'] = [] 
                continue
            doc['paragraphs'] = paragraphs

def remove_whilte_space(data):
    if type(data) == str:
        data = [data]
    l = []
    for s in data:
        s = "".join(s.split())
        l.append(s)
    return l

##方法一: n個字就變成一段
def get_article_paragraphs(documnent):
    if 'yahoo' in documnent["url"]:
        return get_article_paragraphs_commonhealth(documnent['body'])
    elif 'commonhealth' in documnent["url"]:
        return get_article_paragraphs_commonhealth(documnent['body'])
    else:
        assert False

def get_article_paragraphs_by_p_tags(documnent):
    if 'yahoo' in documnent["url"]:
        return get_article_paragraphs_by_p_tags_commonhealth(documnent['body'])
    elif 'commonhealth' in documnent["url"]:
        return get_article_paragraphs_by_p_tags_commonhealth(documnent['body'])
    else:
        assert False

def get_article_paragraphs_commonhealth(html_content):
    max_char_num = 400
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.HTML(html_content,parser)
    for bad in tree.xpath("//div/script"):
        bad.getparent().remove(bad)
    text_content = tree.xpath("//text()")
    return _get_article_paragraphs(text_content,max_char_num)


def get_article_paragraphs_yahoo(html_content):
    max_char_num = 400
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.HTML(html_content,parser)
    for bad in tree.xpath("//div/script"):
        bad.getparent().remove(bad)
    text_content = tree.xpath("//text()")
    return _get_article_paragraphs(text_content,max_char_num)



def _get_article_paragraphs(text_content,max_char_num):
    text_content = list(map(lambda  x: x.rstrip().lstrip(),text_content))
    text_content = list(filter(lambda  x: len(x)>0,text_content))
    paragraphs = []
    current_paragraph  = ''
    current_len = 0
    for text in text_content:
        current_paragraph+=text
        current_len+=len(text)
        if current_len>max_char_num:
            paragraphs.append(current_paragraph)
            current_len = 0
            current_paragraph  = ''
    if len(current_paragraph)>0:
        paragraphs.append(current_paragraph)
    return paragraphs

def get_article_paragraphs_by_p_tags_commonhealth(html_content):
    max_char_num = 450
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.HTML(html_content,parser)
    for bad in tree.xpath("//div/script"):
        bad.getparent().remove(bad)
    pnodes = tree.xpath("//p")
    return _get_article_paragraphs_by_p_tags(pnodes,max_char_num)

def get_article_paragraphs_by_p_tags_yahoo(html_content):
    max_char_num = 450
    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.HTML(html_content,parser)
    for bad in tree.xpath("//div/script"):
        bad.getparent().remove(bad)
    pnodes = tree.xpath("//p")
    return _get_article_paragraphs_by_p_tags(pnodes,max_char_num)


def _get_article_paragraphs_by_p_tags(pnodes,max_char_num):
    paragraphs = []
    current_paragraph  = ''
    for p in pnodes:
        text = ''.join(p.itertext())
        text = text.rstrip().lstrip()
        if len(current_paragraph)+len(text)>max_char_num:
            paragraphs.append(current_paragraph)
            current_paragraph=''
        current_paragraph+=text
    if len(current_paragraph)>0:
        paragraphs.append(current_paragraph)
    return paragraphs



class FakeRetriever():
    def __init__(self):
      pass
    def retrieve_candidates(self,question):
       return {'question':'幫我發大決 冰鳥','documents':[{'body':"12345678",'title':'title1','url':'url1','paragraphs':['1234','5678']},\
              {'body':"abcdefghhjj",'title':'abde','url':'url2','paragraphs':['1234','5678']}]}


class CMKBElasticSearchRetriever():
    def __init__(self,es_db,k,word_dict):
        self.k = k
        self.es_db = es_db
        self.word_dict = word_dict
    def search_elk(self,question):
        query = Query(question,word_dict=self.word_dict)
        dic_keywords =query.extract_keywords_with_diciotnary()
        keywords  = list(set(dic_keywords+query.extract_keywords()))
        docs = self.es_db.retrieve_library_doc(keywords,size=self.k)
        # filter paragraphs without one of keywords
        for doc in docs:
            new_paragraphs = []
            for p in doc['paragraphs']:
                for dw in dic_keywords:
                    if dw in p:
                        new_paragraphs.append(p)
                        break
            doc['paragraphs'] = new_paragraphs
        return docs

    def retrieve_candidates(self,question):
        docs = self.search_elk(question)
        print('retrieve %d docs'%(len(docs)))
        sample_json = {"documents":[],'question':question}
        for i,x in enumerate(docs[0:self.k]):
            o = {'title':x['title'],'url':x['url'],'body':x['body'],'paragraphs':x['paragraphs']}
            sample_json['documents'].append(o)
        return sample_json
        

class GoogleSearchRetriever():
    def __init__(self,site_url,k,event_loop,expand_keywords=False):
        super().__init__()
        self.site_url = site_url
        self.k = k
        self.req_cls = None
        self.event_loop = event_loop
        if 'commonhealth' in site_url:
            self.req_cls = HealthArticleRequest
        elif 'answers.yahoo' in site_url:
            self.req_cls = YahooAnswerQuestionRequest
        else:
            assert False
    def retrieve_candidates(self,question):
        gs_request = GoogleSearchRequest(question,self.site_url,loop=self.event_loop )
        page = gs_request.async_send()
        site_urls = page.get_result_links()
        sample_json = {"documents":[],'question':question}
        for doc_i,link in enumerate(site_urls[0:self.k]):
            print('request : %s'%(link))
            if ('commonhealth' not in link) and ('answers.yahoo' not in link):
                continue
            req = self.req_cls(link,event_loop=self.event_loop)
            article_page =  req.async_send()
            try:
                article =   article_page.to_json()
            except :
                print('error while parsing [%s]'%(link))
                continue
            article.update({'url':link})
            sample_json["documents"].append(article)
        SimpleParagraphTransform().transform(sample_json)
        return  sample_json
