from . import ranker  as qa_ranker
from common.util  import RecordGrouper,Factory




class ParagraphSelector():
    def __init__(self,k):
        self.k = k 
    
    def paragraph_selection(self,sample_list):
        samples_with_rankscore = self.evaluate_scores(sample_list)
        return self.select_top_k_each_doc(samples_with_rankscore)

    def evaluate_scores(self,sample_list):
        pass

    def select_top_k_each_doc(self,samples_with_rankscore):
        grouper = RecordGrouper(samples_with_rankscore)
        group_dict = grouper.group('question')
        selected_samples = []
        for _,values in group_dict.items():
            doc = RecordGrouper(values).group('doc_id')
            for _,paragraphs in doc.items():
               l = list(sorted(paragraphs,key=lambda x: -1*x['rank_score']))
               selected_samples.extend(l[0:self.k]) 
        return selected_samples

class TfIdfSelector(ParagraphSelector): 
    def __init__(self,ranker=None,k=1):
        super().__init__(k)
        print('tfidf selector')
        self.ranker =ranker
        if self.ranker is None:
            self.ranker = qa_ranker.TfIdfRanker() ###....

    def evaluate_scores(self,sample_list):
        return self.ranker.evaluate_on_records(sample_list)


class WordMatchSelector(ParagraphSelector):
    def __init__(self,k=1):
        super().__init__(k)
        self.ranker = qa_ranker.WordMatchRanker()

    def evaluate_scores(self,sample_list):
        return self.ranker.evaluate_on_records(sample_list)




class BertRankerSelector(ParagraphSelector):
    def __init__(self,ranker,k=1):
        super().__init__(k)
        if type(ranker)==str:
            try:
                self.ranker = qa_ranker.RankerFactory.from_exp_name(ranker)
            except:
                self.ranker = qa_ranker.RankerFactory.from_config_path(ranker)
        else:
            self.ranker = ranker
            
    def evaluate_scores(self,sample_list):
        return self.ranker.evaluate_on_records(sample_list)



class ParagraphSelectorFactory(Factory):
    NAME2CLS = {"word_match":WordMatchSelector,"tfidf":TfIdfSelector,'bert_ranker':BertRankerSelector}
    def __init__(self):
        pass

    @classmethod
    def create_selector(cls,information,**kwargs):
        if isinstance(information,str):
            return cls.from_name(information,**kwargs)
        elif isinstance(information,dict):
            return cls.from_dict(information)
        else:
            assert False

    @classmethod
    def from_name(cls,name,**kwrags):
        if name not in cls.NAME2CLS:
            return None
        return cls.NAME2CLS[name](**kwrags)


