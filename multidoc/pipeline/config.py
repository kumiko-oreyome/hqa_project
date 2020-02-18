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
bert_ranker_old = create_config_dict('BertPointwiseRanker',config='./models/ranker/bert_ranker/old/config.json')

#readers
bert_reader_old = create_config_dict('BertReader',config='./models/reader/bert_reader/old/config.json')


#paragraph selectors
bert_ranker_selector_old = create_config_dict('RankBasedSelector',ranker=bert_ranker_old,k=1)
tfidf_selector  = create_config_dict('RankBasedSelector',ranker=tfidf_ranker,k=1)
wordmatch_selector = create_config_dict('RankBasedSelector',ranker=word_match_ranker,k=1)


