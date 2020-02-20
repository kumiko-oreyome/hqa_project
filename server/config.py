def create_config_dict(class_name,**kwargs):
    return {'class':class_name,'kwargs':{k:v for k,v in kwargs.items()}}


#decoders
default_decoder = create_config_dict('LinearDecoder',k=1)
top2_decoder = create_config_dict('LinearDecoder',k=2)

#judgers
default_judger = create_config_dict('MaxAllJudger')


#rankers
word_match_ranker = create_config_dict('WordMatchRanker')
tfidf_ranker = create_config_dict('TfIdfRanker')
bert_ranker = create_config_dict('BertPointwiseRanker',config='./models/bert_ranker/config.json')

#readers
bert_reader = create_config_dict('BertReader',config='./models/bert_reader/config.json')


#paragraph selectors
bert_ranker_selector = create_config_dict('RankBasedSelector',ranker=bert_ranker,k=1)
tfidf_selector  = create_config_dict('RankBasedSelector',ranker=tfidf_ranker,k=1)
wordmatch_selector = create_config_dict('RankBasedSelector',ranker=word_match_ranker,k=1)


# data collection for cmkb
cmkb_keywords = ['糖尿病','高血壓']
cmkb_doc_file = './data/disease_cmkb.jsonl'
keywords_path = './data/keywords.txt'
stopwords_path = './data/stopwords.txt'


# mrc server models
mock_mrc_model = create_config_dict('mock')
selector_reader_model = create_config_dict('SelectorReaderModel',selector=bert_ranker_selector,reader=bert_reader)

mrc_server_port = 8787
mrc_server_host = 
mrc_server_config_mock = {"port":mrc_server_port,"model":mock_mrc_model}
mrc_server_config = {"port":mrc_server_port,"model":selector_reader_model}

#document retriever
fake_retriever = {'class':'FakeRetriever','kwargs':{}}
gs_retriever = {'class':'GoogleSearchRetriever','kwargs':{'site_url':'https://www.commonhealth.com.tw/article','k':5,'expand_keywords':False}}

#mrc proxy
direct_access_proxy = {'class':'DirectAccessProxy','kwargs':{"config":selector_reader_model}}
redirect_proxy = {'class':'RedirectProxy','kwargs':{"server_url":'http://localhost:%s/qa'%(mrc_server_port)}}



# elastic search
elastic_host =  
elastic_port =  9200
elastic_index_cmkb = 'cmkb'
elastic_doc_type_cmkb= 'library'
elk_config = {"host":elastic_host,"port":elastic_port ,"index":elastic_index_cmkb ,"doc_type":elastic_doc_type_cmkb}
elk_retriever_config = {'class':'CMKBElasticSearchRetriever','kwargs':{'config':{'elk_db':elk_config,'word_dict':keywords_path,'k':5}}}



MRC_SERVER_CONFIG = mrc_server_config

# config for create qa server
MRC_PROXY_CONFIG = redirect_proxy
WEB_RETRIEVER_CONFIG = gs_retriever
ELK_RETRIEVER_CONFIG = elk_retriever_config



