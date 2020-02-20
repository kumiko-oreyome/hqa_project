from .metric import compute_bleu_rouge,normalize,RankingMetric,SQuadMetric
from multidoc.util import group_dict_list,RecordGrouper
from multidoc.dataset import  dureader
from tqdm import tqdm
class DureaderMultiDocMrcEvaluator():
    def __init__(self,metric_name='dureader'):
        self.metric_name = metric_name
        assert self.metric_name in ['dureader','squad']
    
    def get_defalut_judger(self):
        from multidoc.core.op import MaxAllJudger
        return MaxAllJudger()
    
    # TODO: refactoring this part to another class
    def evaluate_multidoc_mrc_on_path(self,paths,ranker,reader,judger=None):
        from multidoc.core.op import  RankBasedSelector
        selector = RankBasedSelector(ranker,k=1)
        selected_examples = []
        for example in tqdm(dureader._load_dataset(paths,[])):
            example =  dureader.DureaderExample(example)
            if example.illegal_answer_doc():
                continue
            l = example.flatten(['question_id','question','answers'],[])
            res = selector.paragraph_selection(l)
            selected_examples.extend(res)
        return self.evaluate_reader_on_examples(selected_examples,reader,judger)
    
    def evaluate_multidoc_mrc_on_examples(self,examples,ranker,reader,judger=None):
        pass
          
    def evaluate_reader_on_path(self,path,evaluate_mode,reader,judger=None):
        assert evaluate_mode in ['gold_paragraph','answer_doc',None]
        examples = dureader.load_dureader_examples(path,evaluate_mode,False)
        examples = dureader.filter_no_answer_doc_examples(examples)
        examples = dureader.flatten_examples(examples,['question_id','question','answers'],[])
        print('load %d evaluate examples'%(len(examples)))
        return self.evaluate_reader_on_examples(examples,reader,judger)
         
    def evaluate_reader_on_examples(self,examples,reader,judger=None):
        if judger is None:
            judger = self.get_defalut_judger() 
        _preds = reader.evaluate_on_records(examples)
        _preds = group_dict_list(_preds,'question_id')
        pred_answers  = judger.judge(_preds)
        print('bidaf evaluation')
        if self.metric_name == 'dureader':
            return self.evaluate_dureader_metrics(pred_answers)
        elif self.metric_name == 'squad':
            return self.evaluate_squad_metrics(pred_answers)
        return None
    def evaluate_squad_metrics(self,pred_answers):
        pred_for_eval = {}
        ref_dict = {}
        for qid,v in pred_answers.items():
            best_pred = v[0]
            if len(best_pred['answers']) == 0:
                continue
            pred_for_eval[qid] = best_pred['span']
            ref_dict[qid]  = best_pred['answers'][0]
        metric = SQuadMetric(pred_for_eval,ref_dict)
        em,f1_score = metric.exactly_match(),metric.f1_score()
        return {'EM':em,'F1':f1_score}
    
    def evaluate_dureader_metrics(self,pred_answers):
        pred_for_bidaf_eval = {}
        ref_dict = {}
        for qid,v in pred_answers.items():
            best_pred = v[0]
            if len(best_pred['answers']) == 0:
                continue
            pred_for_bidaf_eval[qid] = normalize([ best_pred['span']])
            ref_dict[qid]  = normalize(best_pred['answers'])
        result = compute_bleu_rouge(pred_for_bidaf_eval,ref_dict)
        return result
    
class DureaderRankingEvaluator():
    def __init__(self,**sample_args):
        self.sample_args = sample_args
    
    def evaluate_on_path(self,path,ranker):
        examples = dureader.load_pointwise_examples(path,**sample_args)
        print('load %d evaluate examples'%(len(examples)))
        return self.evaluate_on_examples(examples,ranker)
         
    def evaluate_on_examples(self,examples,ranker):
        print('ranker evaluation')
        _preds = ranker.evaluate_on_records(examples)
        sorted_results = RecordGrouper(_preds).group_sort('question_id','rank_score',50)
        metric = RankingMetric(sorted_results)
        accuracy,precision = metric.accuracy(),metric.precision(1)
        return {'accuracy':accuracy,'precision':precision}
        
