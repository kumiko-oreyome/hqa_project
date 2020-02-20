import asyncio
from .config import mrc_server_config,elastic_host,elastic_port,elastic_index_cmkb,elastic_doc_type_cmkb,cmkb_doc_file,mrc_server_config_mock,\
    mock_mrc_model,selector_reader_model,mrc_server_port,mrc_server_host,elk_config,keywords_path

from . import mrc
from .mrc_proxy import create_mrc_proxy
from .webrequest import HealthArticleRequest,GoogleSearchRequest,YahooAnswerQuestionRequest,CMKBRequest, CMKBKeywordSearchPage,CMKBLibraryPage
from .dao.cmkb import CMKBDLibraryDocument,CMKBElasticDB
from .docretv import  create_document_retriever
from multidoc.util import jsonl_writer

event_loop = asyncio.get_event_loop()

doc_1 = {'title':'c8763','url':'www.c8763.com','paragraphs':["c8763是某黑色劍士的技能","發動c8763需要隊友幫稱十秒","結束了嗎"]}
doc_2 = {'title':'sao','url':'www.sao.com','paragraphs':["因為太過中二所以c8763會被噓","c8763後來有很多被人惡搞的梗"]}
test_mrc_input = {"question":"c8763是什麼","documents":[doc_1,doc_2]}


def test_mrc_server():

    app = mrc.create_app(mrc_server_config)
    print('send data')
    with app.test_client() as c:
        rv = c.post('/qa', json={'mrc_input':test_mrc_input,'answer_num':2,'algo_version':0})
    json_data = rv.get_json()
    print(json_data)




def test_common_health_request():
    print('test common health request')
    r = HealthArticleRequest('https://www.commonhealth.com.tw/article/article.action?nid=80297',event_loop)
    page = r.async_send()
    title = page.get_title()
    print('title %s'%(title))
    assert r.article_id == 80297
    assert '排濕' == title[0:2]


def test_google_search_request():
    COMMON_HEALTH_SITE_URL = 'https://www.commonhealth.com.tw/article'
    print('test google search request')
    r =  GoogleSearchRequest(keywords=['糖尿病','胃繞道','飲食'],site_url=COMMON_HEALTH_SITE_URL,loop=event_loop)
    page = r.async_send()
    print(len(page.get_result_links()))
    print(page.get_result_links())
    print(page.get_next_page_link())

def test_yahoo_answer_request():
    print('test yahoo answer request')
    r = YahooAnswerQuestionRequest('https://tw.answers.yahoo.com/question/index?qid=20191025153345AAXp5e3')
    page = r.async_send()
    title = page.get_title()
    #print(page.get_body())
    assert title == "食鹽過了期還可以吃嗎? 有何不良影響?"
    

def test_cmkb_request():
    print('test cmkb request')
    page = CMKBRequest().get_library_page('https://kb.commonhealth.com.tw/library/30.html')
    with open('./test/cmkb_test.html','w',encoding='utf-8') as f:
        f.write(page.html_content)


def test_cmkb_keyword_search():
    print('test cmkb keyword search')
    print('create kw search page')
    page = CMKBKeywordSearchPage('糖尿病',event_loop)
    print('start script')
    async def  script():
        await page.expand_page_until(3)
        assert len(page.current_page_infos) == 48
        print(page.current_page_infos)
        

def test_cmkb_library_page():
    html = open('./test/cmkb_test.html','r',encoding='utf-8').read()
    page = CMKBLibraryPage(html)
    assert page.get_title() == '什麼是糖尿病？'
    paragraphs = CMKBRequest.html_parsing_paragraph(page.get_body())
    doc = CMKBDLibraryDocument(url='https://kb.commonhealth.com.tw/library/30.html',title=page.get_title(),body=page.get_body(),tags=['糖尿病'],paragraphs=paragraphs)
    jsonl_writer('./test/test_cmkb_library_page.jsonl',[doc.to_json()])


def test_crawl_cmkb_docs_by_keyword():
    req = CMKBRequest(event_loop)
    docs = req.crawl_keyword_search_page('高血壓',100)
    jsonl_writer('./test/高血壓.jsonl',docs)


def test_elastic():
    db = CMKBElasticDB(host=elastic_host,port=elastic_port,index=elastic_index_cmkb,doc_type=elastic_doc_type_cmkb)
    res = db.retrieve_library_doc("糖尿病 高血壓 感冒")
    print('reteieve %d results'%(len(res)))
    for r in res:
        print(r['title'],r['tags'])

def test_TestMrcProxy():
    proxy = create_mrc_proxy({'class':'TestProxy','kwargs':{"config":mrc_server_config_mock}})
    print(proxy.send_mrc_input(test_mrc_input))
    proxy = create_mrc_proxy({'class':'TestProxy','kwargs':{"config":mrc_server_config}})
    print(proxy.send_mrc_input(test_mrc_input))

def test_DirectAccessProxy():  
    proxy = create_mrc_proxy({'class':'DirectAccessProxy','kwargs':{"config":mock_mrc_model}})
    print(proxy.send_mrc_input(test_mrc_input))
    proxy = create_mrc_proxy({'class':'DirectAccessProxy','kwargs':{"config":selector_reader_model}})
    print(proxy.send_mrc_input(test_mrc_input))

def test_RedirectProxy():  
    proxy = create_mrc_proxy({'class':'RedirectProxy','kwargs':{"server_url":'http://localhost:%s/qa'%(mrc_server_port)}})
    print(proxy.send_mrc_input(test_mrc_input))

def test_FakeRetriever():  
    retv =  create_document_retriever({'class':'FakeRetriever','kwargs':{}},event_loop)
    print(retv.retrieve_candidates("今天幾月幾號?"))


def test_GoogleSearchRetriever():  
    #url = 'https://tw.answers.yahoo.com'
    url = 'https://www.commonhealth.com.tw/article'
    retv =  create_document_retriever({'class':'GoogleSearchRetriever','kwargs':{'site_url':url,'k':5,'expand_keywords':False}},event_loop)
    print(retv.retrieve_candidates("糖尿病的飲食"))


def test_ELKRetriever():  
    config = {'class':'CMKBElasticSearchRetriever','kwargs':{'config':{'elk_db':elk_config,'word_dict':keywords_path,'k':5}}}    
    retv =  create_document_retriever(config,event_loop)
    print(retv.retrieve_candidates("糖尿病的原因是什麼?"))

if __name__ == '__main__':
    #test_mrc_server()
    #test_common_health_request()
    #test_google_search_request()
    #test_yahoo_answer_request()
    #test_cmkb_request()
    #test_cmkb_keyword_search()
    #test_cmkb_library_page()
    #test_crawl_cmkb_docs_by_keyword()
    #test_elastic()
    #test_TestMrcProxy()
    #test_DirectAccessProxy()
    #test_RedirectProxy()
    #test_FakeRetriever()
    #test_GoogleSearchRetriever()
    test_ELKRetriever()