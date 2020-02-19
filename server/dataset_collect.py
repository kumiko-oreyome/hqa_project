from .google_search import GoogleSearchRequest
from .common_health import  HealthArticleRequest
from .yahoo_answer import YahooAnswerQuestionRequest
from .db import JsonlRCDatabase
from .util import Query
import time
import random


YAHOO_ANSWER_SITE_URL = 'https://tw.answers.yahoo.com'
COMMON_HEALTH_SITE_URL  = 'https://www.commonhealth.com.tw/article'


    




    



def follow_gs_link_to_json(gs_request,link_req_cls,docs,sleep_time=None,k=None):
    page = gs_request.async_send()
    if k is None:
        k = len(page.get_result_links())
    for doc_i,link in enumerate(page.get_result_links()[0:k]):
        print('request : %s'%(link))
        req = link_req_cls(link)
        if sleep_time is None:
            article_page = req.async_send()
        else:
            article_page = req.send()
            time.sleep(sleep_time)
        article =   article_page.to_json()
        if article["body"] is None:
            continue
        article.update({'doc_id':len(docs['documents']),'url':link})
        docs['documents'].append(article)
        docs['doc2_answers'].append([])



def collect_health_news_by_query(query,db,k=None,expand=False):
    query = Query(query)
    # get top 10 results
    docs = {"documents":[],'question':query.query,'answer_docs':[],'answers':[],'doc2_answers':[]}
    gs_request = GoogleSearchRequest(query.query,COMMON_HEALTH_SITE_URL)
    follow_gs_link_to_json(gs_request, HealthArticleRequest,docs,k=k)
    if expand:
        kws = query.extract_keywords()
        docs['extract_keywords'] = kws
        gs_request = GoogleSearchRequest(kws,COMMON_HEALTH_SITE_URL)
        time.sleep(random.randint(0,10))
        follow_gs_link_to_json(gs_request, HealthArticleRequest,docs,k=k)
        #delete same documents
        new_docs = []
        url_set = set()
        for doc in docs["documents"]:
            if doc['url'] not in url_set:
                url_set.add(doc['url'])
                new_docs.append(doc)
        docs["documents"] = new_docs
        for i,doc in enumerate(new_docs):
            doc['doc_id'] = i
    db.add_new_sample(docs)
    db.save()


def collect_yahoo_answer_by_query(query,db,k=None,expand=False):
    req_cls = YahooAnswerQuestionRequest
    query = Query(query)
    # get top 10 results
    docs = {"documents":[],'question':query.query,'answer_docs':[],'answers':[],'doc2_answers':[]}
    gs_request = GoogleSearchRequest(query.query,YAHOO_ANSWER_SITE_URL)
    follow_gs_link_to_json(gs_request,req_cls,docs,k=k)
    if expand:
        kws = query.extract_keywords()
        docs['extract_keywords'] = kws
        gs_request = GoogleSearchRequest(kws,YAHOO_ANSWER_SITE_URL)
        time.sleep(random.randint(0,10))
        follow_gs_link_to_json(gs_request,YahooAnswerQuestionRequest ,docs,k=k)
        #delete same documents
        new_docs = []
        url_set = set()
        for doc in docs["documents"]:
            if doc['url'] not in url_set:
                url_set.add(doc['url'])
                new_docs.append(doc)
        docs["documents"] = new_docs
        for i,doc in enumerate(new_docs):
            doc['doc_id'] = i
    db.add_new_sample(docs)
    db.save()    
        

def collect_dataset_by_question_file(question_file,out_file,site):
    db = JsonlRCDatabase(out_file,'qid')
    with open(question_file,'r',encoding='utf-8') as f:
        for l in f:
            q = l.rstrip()
            print('collect question %s'%(q))
            if site == 'common_health':
                collect_health_news_by_query(q,db,True)
                #如果不加會被google擋掉
                time.sleep(250+random.randint(0,60))
            elif site == 'yahoo_answer':
                collect_yahoo_answer_by_query(q,db,True)
            else:
                assert False


def default_data_collect_query(query_text,db,expand=False,k=5,loop=None):
    query = Query(query_text)
    # get top 10 results
    docs = {"documents":[],'question':query.query,'answer_docs':[],'answers':[],'doc2_answers':[]}
    gs_request = GoogleSearchRequest(query.query,COMMON_HEALTH_SITE_URL,loop=loop)
    follow_gs_link_to_json(gs_request, HealthArticleRequest,docs,k=k,sleep_time=1)
    req_cls = YahooAnswerQuestionRequest
    gs_request = GoogleSearchRequest(query.query,YAHOO_ANSWER_SITE_URL,loop=loop)
    follow_gs_link_to_json(gs_request,req_cls,docs,k=k,sleep_time=1)
    if expand:
        kws = query.extract_keywords()
        docs['extract_keywords'] = kws
        gs_request = GoogleSearchRequest(kws,COMMON_HEALTH_SITE_URL)
        time.sleep(random.randint(0,10))
        follow_gs_link_to_json(gs_request, HealthArticleRequest,docs,k=k,sleep_time=1)
        gs_request = GoogleSearchRequest(kws,YAHOO_ANSWER_SITE_URL)
        time.sleep(random.randint(0,10))
        follow_gs_link_to_json(gs_request,YahooAnswerQuestionRequest ,docs,k=k)
        #delete same documents
        new_docs = []
        url_set = set()
        for doc in docs["documents"]:
            if doc['url'] not in url_set:
                url_set.add(doc['url'])
                new_docs.append(doc)
        docs["documents"] = new_docs
        for i,doc in enumerate(new_docs):
            doc['doc_id'] = i
    db.add_new_sample(docs)
    db.save()




if __name__ == '__main__':
    #collect_health_news_by_query('我平常飲食都吃很多蔬菜，也很注意清淡少用油炸，為什麼還會得乳癌',DEBUG_JSONL_DB,True)
    #collect_dataset_by_question_file(CHIU_QUESTION_FILE,CHIU_QUESTION_YAHOO_JSONL_DB_FILE,'yahoo_answer')
    #collect_yahoo_answer_by_query('我平常飲食都吃很多蔬菜，也很注意清淡少用油炸，為什麼還會得乳癌',DEBUG_JSONL_DB_YAHOO,True)
    COMMON_HEALTH_DATA_DIR = './python_crawler/annotation_server/data/common_health'
    #DEBUG_JSONL_DB = JsonlRCDatabase('%s/%s'%(COMMON_HEALTH_DATA_DIR,'debug.jsonl'),'qid')
    #DEBUG_JSONL_DB_YAHOO = JsonlRCDatabase('%s/%s'%(COMMON_HEALTH_DATA_DIR,'debug_yahoo.jsonl'),'qid')
    CHIU_QUESTION_JSONL_DB_FILE = './python_crawler/annotation_server/data/common_health/chiu_question2.jsonl'
    CHIU_QUESTION_YAHOO_JSONL_DB_FILE = './python_crawler/annotation_server/data/common_health/chiu_question_yahoo.jsonl'
    CHIU_QUESTION_FILE = './python_crawler/chiu_question.txt'

    with open('./data/docs/fake_db.jsonl','w',encoding='utf-8') as f:
        TMP_JSONL_DB = JsonlRCDatabase('./data/docs/fake_db.jsonl','qid')
        collect_health_news_by_query('糖尿病',TMP_JSONL_DB,k=5)
        collect_health_news_by_query('高血壓',TMP_JSONL_DB,k=5)
        #default_data_collect_query('適合中風者的飲食?',TMP_JSONL_DB,k=5)
        #collect_health_news_by_query('適合中風者的飲食?',TMP_JSONL_DB,k=5)
        #collect_yahoo_answer_by_query('適合中風者的飲食?',TMP_JSONL_DB,k=5)
        #collect_yahoo_answer_by_query('吃什麼可以減緩高血壓？',TMP_JSONL_DB,True)
    #,乳癌發生原因是什麼?