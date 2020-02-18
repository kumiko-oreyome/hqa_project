import itertools,random
from multidoc.core.entity import Document,MultiDocExample,Paragraph
from multidoc.eval.metric import f1_score,recall,max_metric_over_multiple_refs
from multidoc.util import jsonl_reader


def load_dureader_examples(paths,mode):
    l = []
    for example in _load_dataset(paths,[]):
        if mode == 'answer_doc' or mode== 'gold_paragraph':
            drex = DureaderExample(example)
            if drex.illegal_answer_doc():
                continue
            if mode == 'gold_paragraph':
                example = drex.select_by_indexs('all','gold_paragraph',['paragraphs','segmented_paragraphs'])
            elif mode == 'answer_doc':
                example = drex.select_by_indexs('answer_doc','gold_paragraph',['paragraphs','segmented_paragraphs'])
        l.append(example)
    return l

def filter_no_answer_doc_examples(examples):
    l = []
    for example in examples:
        if DureaderExample(example).illegal_answer_doc():
            continue
        l.append(example)
    return l

def flatten_examples(examples,sample_fields,doc_fields,para_fields=[]):
    l = []
    for example in examples:
        l.extend(flatten_example(example,sample_fields,doc_fields,para_fields))
    return l



def load_dataset(path_list,pipelie_funcs,as_list=True):
    if as_list:
        return list(itertools.chain(*list(_load_dataset(path_list, pipelie_funcs))))
    else:
        return _load_dataset(path_list,pipelie_funcs)

def _load_dataset(path_list,pipelie_funcs):
    if isinstance(path_list,str):
        path_list = [path_list]
    for path in path_list:  
        for raw_sample in jsonl_reader(path):
            result = raw_sample
            for func in pipelie_funcs:
                result = func(result)
                if result is None:
                    break
            if result is None:
                continue
            yield result
            
def flatten_example(json_obj,sample_fields,doc_fields,para_fields=[]):
    return DureaderExample(json_obj).flatten(sample_fields,doc_fields,para_fields)

def flatten_doc(json_obj,doc_fields,para_fields=[]):
    return  DureaderDocument(json_obj).flatten(doc_fields,para_fields)






def load_pointwise_examples(path,sample_method,neg_num,attach_label=True):
    def _sample_from_example(example):
        obj = example.sample_obj
        question,question_id = obj["question"],obj["question_id"]
        if sample_method == 'answer_doc':
            pos_paragraphs,neg_paragraphs = ParagraphSampler(obj).sample_gold_para_as_postive(k=neg_num)
        else:
            pass
        ret = []
        for p in pos_paragraphs:
            d  = {'question':question,'question_id':question_id,'passage':p}
            if attach_label:
                d["label"] = 1
            ret.append(d)
        for p in neg_paragraphs:
            d  = {'question':question,'question_id':question_id,'passage':p}
            if attach_label:
                d["label"] = 0
            ret.append(d)
        return ret
            
    check_answer_doc = lambda x:None if DureaderExample(x).illegal_answer_doc() else DureaderExample(x)
    check_length = lambda x: None if x.get_answer_span()[1]+3+min(len(x.sample_obj["question"]),max_q_len) >max_seq_len else x
    sample_func = lambda x: _sample_from_example(x)
    preprocess_funcs = [ check_answer_doc,sample_func]
    examples =  load_dataset(path,preprocess_funcs)
    return examples
        

class ParagraphSampler():
    @classmethod
    def sample_paragraphs_without_positive(cls,paras,pos_idx,k=1):
        neg_idx_list = [ idx for idx in range(len(paras)) if idx!=pos_idx]
        neg_num = min(k,len(neg_idx_list))
        neg_examples = []
        if neg_num > 0:
            neg_idxs = random.sample(neg_idx_list,neg_num)
            neg_examples= [ paras[ni] for ni in neg_idxs]
        return neg_examples
    
    def __init__(self,example):
        self.example = example
        
    def sample_gold_para_as_postive(self,k=1):
        pos_examples = []
        neg_examples = []
        drex = DureaderExample(self.example)
        if drex.illegal_answer_doc():
            return pos_examples,neg_examples
        answer_doc,answer_doc_idx = drex.get_answer_doc()
        paras = answer_doc['paragraphs']
        pos_idx = answer_doc['most_related_para']
        pos_examples = [paras[pos_idx]]
        neg_examples = self.sample_paragraphs_without_positive(paras,pos_idx,k)
        return pos_examples,neg_examples   

    # need to keep the information of which positve example come from which document??? (use yield instead??)
    def sample_most_realted_paras_as_postive(self,k=1):
        pos_examples = []
        neg_examples = []
        for doc in self.example["documents"]:
            if 'most_related_para' not in doc:
                continue        
            paras = doc['paragraphs']
            pos_idx = doc['most_related_para']
            pos_examples.append(paras[pos_idx])
            neg_examples.extend(self.sample_paragraphs_without_positive(paras,pos_idx,k))
        return pos_examples,neg_examples  


def find_best_question_match(doc, question, with_score=False):
    if len(question) == 0:
        most_related_para,max_related_score = 0,0
    else:
        most_related_para,max_related_score = find_most_related_paragraph(question,doc['segmented_paragraphs'],lambda item,para_tokens:recall(item,para_tokens))
    if with_score:
        return most_related_para, max_related_score
    return most_related_para

def find_fake_answer(sample):    
    for doc in sample['documents']:
        most_related_para = 0
        if len(sample['segmented_answers']) > 0:
            most_related_para, _ = find_most_related_paragraph(sample['segmented_answers'],doc['segmented_paragraphs'],lambda answers,para_tokens: max_metric_over_multiple_refs(para_tokens,answers,recall))
        doc['most_related_para'] = most_related_para

    sample['answer_docs'] = []
    sample['answer_spans'] = []
    sample['fake_answers'] = []
    sample['match_scores'] = []

    if len(sample['segmented_answers']) == 0 :
        return
    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None
    answer_tokens = set()
    for segmented_answer in sample['segmented_answers']:
        answer_tokens = answer_tokens | set([token for token in segmented_answer])
    for d_idx, doc in enumerate(sample['documents']):
        if not doc['is_selected']:
            continue
        if doc['most_related_para'] == -1:
            doc['most_related_para'] = 0
        most_related_para_tokens = doc['segmented_paragraphs'][doc['most_related_para']][:1000]
        paragraph = Paragraph(most_related_para_tokens)
        for (start_idx,end_idx),span_tokens in  paragraph.enumerate_spans():
            if span_tokens[start_idx] not in answer_tokens:
                continue
            match_score = max_metric_over_multiple_refs(span_tokens,sample['segmented_answers'],f1_score)
            if match_score > best_match_score:
                best_match_d_idx = d_idx
                best_match_span = [start_idx, end_idx]
                best_match_score = match_score
                best_fake_answer = ''.join(span_tokens)
    if best_match_score > 0:
        sample['answer_docs'].append(best_match_d_idx)
        sample['answer_spans'].append(best_match_span)
        sample['fake_answers'].append(best_fake_answer)
        sample['match_scores'].append(best_match_score)


        




def get_fields_by_index(json_obj,indexs,indexed_fields):
    if type(indexs) == int:
        indexs = [indexs]
    if type(indexed_fields)==str:
        indexed_fields = [indexed_fields]
    ret = {}
    for field in indexed_fields:
        ret[field] = [ json_obj[field][i] for i in indexs]
    return ret

def copy_by_fields(json_obj,white_fields=[],black_fields=[]):
    if type(white_fields)==str:
        white_fields = [white_fields]
    if type(black_fields)==str:
        black_fields = [black_fields]
    
    if len(white_fields)==0:
        all_fields = list(json_obj.keys())
    else:
        all_fields = white_fields 
    all_fields = list(set(all_fields)-set(black_fields))
    return {f:json_obj[f] for f in all_fields}

class DureaderExample():
    ALL_FIELDS = ['answer_spans','answer_docs','fake_answers','question','answers','question_id','question_type']

    def __init__(self,sample_obj):
        self.sample_obj = sample_obj
    
    def update(self,field,value):
        self.sample_obj[field] = value
        return self
    
    def select_fields(self,sample_fields,doc_fields,para_fields):
        r = copy_by_fields(self.sample_obj,sample_fields)
        r['documents'] = [DureaderDocument(doc).select_fields(doc_fields,para_fields) for doc in self.sample_obj['documents']]
        return r
    
    def select_by_indexs(self,doc_indexs,para_indexs,para_fields): 
        doc_indexs = self.get_indexs(doc_indexs)
        if para_indexs == 'gold_paragraph' or para_indexs =='most_related_para':
            if type(doc_indexs)==list:
                para_indexs = ['gold_paragraph' for _ in range(len(doc_indexs)) ]
            else:
                para_indexs = ['gold_paragraph']
        l = []
        for doc_i,pidxs in zip(doc_indexs,para_indexs):
            d =  DureaderDocument(self.get_docs(doc_i))
            l.append(d.select_by_indexs(pidxs,para_fields))           
        self.sample_obj['documents'] = l
        return self.sample_obj
    
    def copy_content(self,white_fields=[],black_fields=[]):
        o = copy_by_fields(self.sample_obj ,white_fields,black_fields)
        return o

    def copy_subset_documents(self,doc_indexs):
        o = self.copy_content()
        o['documents'] = self.get_docs(doc_indexs)
        return o

    def get_documents(self):
        return self.sample_obj['documents']
    
    def get_document(self,index):
        return self.sample_obj['documents'][index]
      
    def illegal_answer_doc(self):
        return  'answer_docs' not in self.sample_obj or len(self.sample_obj['answer_docs'])==0  \
    or self.sample_obj['answer_docs'][0]>=len(self.sample_obj['documents']) or self.sample_obj['answer_docs'][0]<0

    def get_most_related_paras(self,wrap_list=False):
        l = [DureaderDocument(x).get_most_related_para() for x in self.get_documents()] 
        if wrap_list:
            l = [ [x] for x in l]
        return l      


    def get_docs(self,doc_idxs):
        docs =  get_fields_by_index(self.sample_obj,doc_idxs,'documents')['documents']
        if type(doc_idxs) == int:
            return docs[0]
        return docs
    
    def get_indexs(self,doc_index):
        if doc_index=='all':
            doc_indexs = list(range(len(self.get_documents())))
        elif doc_index == 'answer_doc':
            doc_indexs = [self.get_answer_doc()[1]]
        return doc_indexs
    
    def get_paragraph_fields(self,doc_indexs,paragraph_indexs,para_fields):
        doc_indexs = self.get_indexs(doc_indexs)
        l = []
        for doc_i,pidxs in zip(doc_indexs,paragraph_indexs):
            d =  DureaderDocument(self.get_docs(doc_i))
            l.append(d.get_paragraph_fields(pidxs,para_fields))
        return l

    def get_answer_doc(self):
        if self.illegal_answer_doc():
            return None,None
        doc_id = self.sample_obj['answer_docs'][0]
        return self.get_document(doc_id),doc_id

    def get_answer_span(self):
        return  self.sample_obj['answer_spans'][0]

    def get_answer(self):
        return self.sample_obj['answers'][0]

    def flatten(self,sample_fields,doc_fields,para_fields=[]):
        ret = []
        for doc_id,doc in  enumerate(self.get_documents()):
            passage_list  =  DureaderDocument(doc).flatten(doc_fields,para_fields)   
            for obj in passage_list:
                obj.update({ 'doc_id':doc_id}) 
                obj.update({ k:self.sample_obj[k] for k in sample_fields}) 
            ret.extend(passage_list)
        return ret


    def get_gold_paragraph(self,para_fields):
        answer_doc  = self.get_answer_doc()[0]
        d =  DureaderDocument(answer_doc)
        return  d.get_paragraph_fields(d.get_most_related_para(),para_fields)

    def get_gold_span(self):
        answer_doc = self.get_answer_doc()[0]
        if 'segmented_paragraphs' in answer_doc:
            p = Paragraph(self.get_gold_paragraph(['segmented_paragraphs'])['segmented_paragraphs'])
            span = self.get_answer_span()
            cstart,cend = p.find_char_span(p.get_span(span))
        else:
            if 'fake_answers' in answer_doc:
                answer = answer_doc['fake_answers'][0]
            else:
                answer = self.get_answer()
            paragraph = self.get_gold_paragraph(['paragraphs'])['paragraphs']
            cstart = paragraph.find(answer)
            cend = cstart+len(answer)-1    
        assert cstart >= 0
        return cstart,cend

    def charspan_preprocessing(self,span_field_name):
        example = self
        empty_doc_example = DureaderExample(example.copy_subset_documents([]))

        if 'segmented_paragraphs' in example.get_answer_doc()[0]:
            empty_doc_example.update('documents',\
                [{'paragraphs':[''.join(example.get_gold_paragraph(['segmented_paragraphs'])['segmented_paragraphs'])]}]).update(span_field_name,example.get_gold_span())
        else:   
            # drcd preprocessing dont generate segmented_paragraphs for saving disk place
            empty_doc_example.update('documents',[{'paragraphs':[example.get_gold_paragraph(['paragraphs'])['paragraphs']]}]).update(span_field_name,example.get_gold_span())

        return empty_doc_example 

class DureaderDocument():
    ALL_FIELDS = ['title','most_related_para']

    def __init__(self,json_obj):
        self.json_obj = json_obj
    
    def select_fields(self,doc_fields,para_fields):
        r = copy_by_fields(self.json_obj,doc_fields)
        r.update(self.get_paragraph_fields('all',para_fields))
        return r
    
    def select_by_indexs(self,para_indexs,para_fields): 
        self.json_obj.update(self.get_paragraph_fields(para_indexs,para_fields))
        return self.json_obj
        
    def update(self,field,value):
        self.json_obj[field] = value
        return self

    def get_most_related_para(self):
        return self.json_obj['most_related_para']
        

    def copy_content(self,white_fields=[],black_fields=[]):
        o = copy_by_fields(self.json_obj ,white_fields,black_fields)
        return o

    def copy_subset_paragraphs(self,para_indexs,para_fields):
        o = self.copy_content()
        o.update(get_fields_by_index(self.json_obj,para_indexs,para_fields))
        return o

    
    def get_indexs(self,paragraph_indexs):
        if paragraph_indexs == 'all':
            paragraph_indexs =   list(range(len(self.json_obj['paragraphs'])))
        elif paragraph_indexs == 'most_related_para' or paragraph_indexs == 'gold_paragraph':
            paragraph_indexs = [self.get_most_related_para()]
        return paragraph_indexs 
    
    def get_paragraph_fields(self,paragraph_indexs,fields):
        if type(fields) == str:
            fields = [fields]
        
        paragraph_indexs = self.get_indexs(paragraph_indexs)
            
        x = get_fields_by_index(self.json_obj,paragraph_indexs,fields)
        if type(paragraph_indexs)==int:
            for f in fields:
                x[f] = x[f][0]
        return x


    def flatten(self,doc_fields,para_fields):
        ret = []
        for para_id,passage in enumerate(self.json_obj['paragraphs']):
            obj = {'passage':passage,'passage_id':para_id}
            obj.update({ k:self.json_obj[k] for k in doc_fields})
            obj.update(self.get_paragraph_fields(para_id,para_fields))
            ret.append(obj)
        return ret
        