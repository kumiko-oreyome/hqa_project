from collections import Counter
from .bleu import Bleu
from .rouge import Rouge

class RankingMetric():
    def __init__(self,rank_dict):
        self.rank_dict = rank_dict
        
    def accuracy(self):
        tot = 0
        cnt = 0
        for q,v in self.rank_dict.items():
            if len(v) == 0:
                 continue
            acc =0 
            for pred in v:
                if pred['rank_score']>=0.5 and pred['label']==1:
                    acc+=1
                elif  pred['rank_score']<0.5 and pred['label']==0:
                    acc+=1
            tot+= acc /len(v)
            cnt+=1
        return tot/cnt
    
    def recall(self,k=1):
        tot = 0
        cnt = 0
        for q,v in self.rank_dict.items():
            l = self.rank_dict[q][0:k]
            acc = sum([ record['label'] for record in l ])
            tot+= acc/sum([ record['label'] for record in v ])
            cnt+=1
        return tot/cnt
    
    def precision(self,k=1):
        tot = 0
        cnt = 0
        for q,v in self.rank_dict.items():
            l = self.rank_dict[q][0:k]
            acc = sum([ record['label'] for record in l ])
            tot+= acc/len(l)
            cnt+=1
        return tot/cnt

#This is character based metric for Chinese  
class SQuadMetric():
    # pred_dict {question_id(str):pred(str)}
    # answer_dict {question_id(str):answer(str)}
    def __init__(self,pred_dict,answer_dict):
        self.pred_dict = pred_dict
        self.answer_dict = answer_dict
        assert set(pred_dict.keys()) == set(answer_dict.keys())
    def exactly_match(self):
        tot = 0
        cnt = 0
        for q,ans in self.answer_dict.items():
            pred = self.pred_dict[q]
            if ans == pred:
                tot+=1
            cnt+=1
        return tot/cnt
    
    def f1_score(self):
        tot = 0
        cnt = 0
        for q,ans in self.answer_dict.items():
            pred = self.pred_dict[q]
            precision,recall,f1 = precision_recall_f1(list(pred),list(ans))
            tot+= f1
            cnt+=1
        return tot/cnt 

def max_metric_over_multiple_refs(candidate,refs,metric_fn):
    return max([metric_fn(candidate,ref) for ref in refs])


def precision_recall_f1(sent_toks,ref_toks):
    common = Counter(sent_toks) & Counter(ref_toks )
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(sent_toks)
    r = 1.0 * num_same / len(ref_toks)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1

def f1_score(sentence,ref):
    return precision_recall_f1(sentence,ref)[2]

def recall(sentence,ref):
    return precision_recall_f1(sentence,ref)[1]

def precision(sentence,ref):
    return precision_recall_f1(sentence,ref)[0]

def normalize(s):
    """
    Normalize strings to space joined chars.
    Args:
        s: a list of strings.
    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(' '.join(tokens))
    return normalized

def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['Rouge-L'] = rouge_score
    return scores