from collections import Counter
from multidoc.util import RecordGrouper,JsonConfig,Tokenizer,Tfidf
from multidoc.dataset.dureader import DureaderExample,load_dataset



class ComponentConfig():
    # config : dict of {'class':'class name here','kwargs':{'arg1':value,'arg2':value2}}
    def __init__(self,config):
        self.config = config
    def get_cls_and_kwargs(self):
        return self.config['class'],self.config['kwargs']

def create_decoder(config_dict=None):
    if config_dict is None:
        return LinearDecoder()
    cls_name,kwargs = ComponentConfig(config_dict).get_cls_and_kwargs()
    name2cls = {'LinearDecoder':LinearDecoder}
    _cls = name2cls[cls_name]
    return  _cls(**kwargs)

def create_judger(config_dict=None):
    if config_dict is None:
        return MaxAllJudger()
    cls_name,kwargs = ComponentConfig(config_dict).get_cls_and_kwargs()
    name2cls = {'MaxAllJudger':MaxAllJudger,'TopKJudger':TopKJudger,'LambdaJudger':LambdaJudger}
    _cls = name2cls[cls_name]
    return  _cls(**kwargs)

def create_ranker(config_dict):
    from  multidoc.util.bert import BertPointwiseRanker,create_bert_model
    cls_name,kwargs = ComponentConfig(config_dict).get_cls_and_kwargs()
    name2cls = {'BertPointwiseRanker':BertPointwiseRanker,'TfIdfRanker':TfIdfRanker,'WordMatchRanker':WordMatchRanker}
    ranker_based_on_bert = ['BertPointwiseRanker']
    _cls = name2cls[cls_name]
    if cls_name in ranker_based_on_bert:
        # create ranker by config path
        ranker = _cls.from_config(JsonConfig(kwargs["config"]))
    else:
        ranker = _cls(**kwargs)
    return ranker

def create_reader(config_dict):
    from  multidoc.util.bert import BertReader
    cls_name,kwargs = ComponentConfig(config_dict).get_cls_and_kwargs()
    name2cls = {'BertReader':BertReader}
    reader_based_on_bert = ['BertReader']
    _cls = name2cls[cls_name]
    if cls_name in reader_based_on_bert:
        # create bert model by config path
        reader = _cls.from_config(JsonConfig(kwargs["config"]))
    else:
        reader = _cls(**kwargs)
    return reader

def create_paragraph_selector(config_dict):
    cls_name,kwargs = ComponentConfig(config_dict).get_cls_and_kwargs()
    name2cls = {'RankBasedSelector':RankBasedSelector}
    ranker_based_selector_cls = ['RankBasedSelector']
    _cls = name2cls[cls_name]
    if cls_name in ranker_based_selector_cls:
        kwargs["ranker"] = create_ranker(kwargs["ranker"])
    ranker = _cls(**kwargs)
    return ranker


def find_most_related_paragraph(match_item,paragraphs,metric_fn):
    most_related_para = 0
    max_related_score = 0
    most_related_para_len = 0
    for p_idx, para_tokens in enumerate(paragraphs):
        related_score = metric_fn(match_item,para_tokens)
        if related_score > max_related_score \
                or (related_score == max_related_score \
                and len(para_tokens) < most_related_para_len):
            most_related_para = p_idx
            max_related_score = related_score
            most_related_para_len = len(para_tokens)
    return most_related_para,max_related_score

# start_probs/end_probs list :[prob1,prob2....]
def extract_answer_dp_linear(start_probs,end_probs):
    # max_start_pos[i] max_start_pos end in i to get max score
    N = len(start_probs)
    assert N>0
    max_start_pos = [0 for _ in range(N)]
    for i in range(1,N):
        prob1 = start_probs[max_start_pos[i-1]]
        prob2 = start_probs[i]
        if prob1 >= prob2:
            max_start_pos[i] = max_start_pos[i-1]
        else:
            max_start_pos[i] = i
    max_span = None
    max_score = -100000
    for i in range(N):
        score = start_probs[max_start_pos[i]]+end_probs[i]
        if score > max_score:
            max_span = (max_start_pos[i],i)
            max_score = score
    return  max_span,max_score


# start_probs/end_probs list :[prob1,prob2....]
def extract_answer_brute_force(start_probs,end_probs,k=1):
    passage_len = len(start_probs)
    best_start, best_end, max_prob = -1, -1, 0
    l = []
    for start_idx in range(passage_len):
        for ans_len in range(passage_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_probs[start_idx]+end_probs[end_idx]
            l.append((start_idx,end_idx,prob))
            l = list(sorted(l,key=lambda x:x[2],reverse=True))[0:k]
    return  list(map(lambda x:(x[0],x[1]),l)), list(map(lambda x:x[2],l))






class LinearDecoder():
    def __init__(self,k=1):
        self.k = k
        
    def decode(self,start_probs,end_probs,text):
        span,score = self.decode_span(start_probs,end_probs)
        answer,max_score = self.decode_answer(span,score,text)
        return answer,max_score,span
    
    def decode_span(self,start_probs,end_probs):
        top_k_spans = []
        N = len(start_probs)
        assert N>0
        top_k_spans.append((0,0,start_probs[0]+end_probs[0]))
        new_topk_span_candidates = [(0,start_probs[0])]
        for i in range(1,N):   
            new_topk_span_candidates.append((i,start_probs[i]))
            new_topk_span_candidates =  list(sorted(new_topk_span_candidates,key=lambda x: x[1],reverse=True))[0:self.k]
            top_k_spans.extend([ (si,i,start_probs[si]+end_probs[i]) for si,_ in new_topk_span_candidates])
            top_k_spans = list(sorted(top_k_spans,key=lambda x: x[2],reverse=True))[0:self.k]
        return list(map(lambda x: (x[0],x[1]),top_k_spans)),list(map(lambda x: x[2],top_k_spans))

    def decode_answer(self,span,score,text):
        intervals = []
        for start,end in span:
            intv_flag = False
            for i,(intv_start,intv_end) in enumerate(intervals):
                if max(intv_start,start) <= min(intv_end,end):
                    intv_flag = True
                    intervals[i] = ( min(intv_start,start),max(intv_end,end))
                    break
            if not intv_flag:
                intervals.append((start,end))
        # concat spans
        ret = ''
        for intv in intervals:
            ret+= text[intv[0]:intv[1]+1]

        return ret,max(score)

## answer selection
    
class MaxAllJudger():
    def __init__(self):
        pass
    def judge(self,documents):
        ret = {}
        for q,v in documents.items():
            ret[q] = []
            max_score = -10000000
            for d in v :
                if d['span_score'] > max_score:
                    max_score = d['span_score']
                    ret[q] = [d]
        return ret


class TopKJudger():
    def __init__(self,k=1):
        self.k = k
    def judge(self,documents):
        ret = {}
        for q,v in documents.items():
            l= sorted(v,key=lambda x: -1*x['span_score'])
            ret[q] = [ x for x in l[:self.k]]
        return ret



class LambdaJudger():
    def __init__(self,k,score_func):
        self.k = k
        self.score_func = score_func
    def judge(self,documents):
        ret = {}
        for q,v in documents.items():
            ret[q] = []
            for d in v :
                d['judger_score'] = self.score_func(d)
            l= sorted(v,key=lambda x: -1*x['judger_score'])
            ret[q] = [ x for x in l[:self.k]]
               
        return ret

class MultiplyJudger():
    def __init__(self):
        pass
    def judge(self,documents):
        ret = {}
        for q,v in documents.items():
            ret[q] = []
            max_score = -100000
            for d in v :
                score = d['span_score']*d['rank_score']
                if score > max_score:
                    max_score =score
                    ret[q] = [d]
        return ret
    
    
    

class RankBasedSelector():
    @classmethod
    def select_top_k_item_in_records(cls,records,group_field,score_field,k):
        group_dict = RecordGrouper(records).group(group_field)
        return cls.select_top_k_item_in_group( group_dict,score_field,k)
    @classmethod
    def select_top_k_item_in_group(cls,group_dict,score_field,k):
        ret = []
        for _,items in group_dict.items():
            l = list(sorted(items,key=lambda x: -1*x[score_field]))
            ret.extend(l[0:k])
        return ret
            
    def __init__(self,ranker,k=1):
        self.ranker = ranker
        self.k = k
    
    def select_paragraph(self,paragraph):
        pass
    
    def paragraph_selection(self,sample_list):
        samples_with_rankscore = self.ranker.evaluate_on_records(sample_list)
        return self.select_top_k_each_doc(samples_with_rankscore)


    def _group_record_by_question(self,records_with_rankscore):
        grouper = RecordGrouper(records_with_rankscore)
        if 'question_id' in records_with_rankscore[0]:  
            group_dict = grouper.group('question_id')
        else:
            group_dict = grouper.group('question')
        return group_dict
    
    def select_top_k_each_doc(self,records_with_rankscore):
        group_dict = self._group_record_by_question(records_with_rankscore)
        selected_samples = []
        for _,values in group_dict.items():
            l = self.select_top_k_item_in_records(values,'doc_id','rank_score',self.k)
            selected_samples.extend(l) 
        return selected_samples
    
# rankers
class TfIdfRanker():
    def __init__(self,corpus_path_list=['./data/zhidao.train.json','./data/search.train.json']):
        self.corpus_path_list = corpus_path_list
        #load examples
        samples = load_dureader_examples(corpus_path_list,'gold_paragraph')
        passages = list(map(lambda x:x['passage'],samples))
        self.tokenizer =  Tokenizer()
        self.tfidf = Tfidf(passages,self.tokenizer.tokenize,corpus_path='./data/tfidf_ranker_corpus')

    def rank(self,example_dict):
        pass

    def evaluate_on_records(self,record_list):
        for record in record_list:
            question = record['question']
            passage =  record['passage']
            score = self.tfidf.cosine_similarity(question,passage)
            record['rank_score'] = score
        return record_list

    def match_score(self,question_tokens,passage_tokens):
        common_with_question = Counter(passage_tokens) & Counter(question_tokens)
        correct_preds = sum(common_with_question.values())
        if correct_preds == 0:
            recall_wrt_question = 0
        else:
            recall_wrt_question = float(correct_preds) / len(question_tokens)
        return recall_wrt_question


class WordMatchRanker():
    def __init__(self):
        self.tokenizer =  Tokenizer()
    def rank(self,example_dict):
        pass
    def evaluate_on_records(self,record_list):
        for record in record_list:
            question = record['question']
            passage =  record['passage']
            question_tokens = self.tokenizer.tokenize(question)
            passage_tokens = self.tokenizer.tokenize(passage)
            score = self.match_score(question_tokens,passage_tokens)
            record['rank_score'] = score
        return record_list

    def match_score(self,question_tokens,passage_tokens):
        common_with_question = Counter(passage_tokens) & Counter(question_tokens)
        correct_preds = sum(common_with_question.values())
        if correct_preds == 0:
            recall_wrt_question = 0
        else:
            recall_wrt_question = float(correct_preds) / len(question_tokens)
        return recall_wrt_question
    

