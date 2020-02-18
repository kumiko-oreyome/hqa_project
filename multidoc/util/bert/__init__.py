import functools,torch
from .. import get_default_device,torchtext_batch_to_dictlist
from torchtext.data import Dataset,Example,RawField,Iterator,Field,BucketIterator
from .modeling import BertForQuestionAnswering,BertForSequenceClassification,BertConfig
from .tokenization import BertTokenizer
import os
from  multidoc.core.op import create_decoder



def create_bert_model(bert_config_dir,model_type,weight_path=None,device=None):
    config_path,bert_pretrained_path,vocab_path = '%s/config.json'%(bert_config_dir),'%s/model.bin'%(bert_config_dir),'%s/vocab.txt'%(bert_config_dir)
    config = BertConfig(config_path)
    if model_type == 'reader':
        model = BertForQuestionAnswering(config)
    elif  model_type == 'ranker':
        model = BertForSequenceClassification.from_pretrained(bert_config_dir,num_labels=2)
    
    if weight_path is not None:
        print('load weight from %s'%(weight_path))
        model.load_state_dict(torch.load(weight_path,map_location=device))

    tokenizer = BertTokenizer(vocab_path, do_lower_case=True)

    if device is None:
        model = model.cpu()
    else:
        model = model.to(device)
    
    return model,tokenizer



RAW_FIELD = RawField()

class BertInput():
    def __init__(self,question,passage,inp,seg,attn_mask,pos_map):
        self.question = question
        self.passage = passage
        self.inp = inp
        self.seg = seg
        self.attn_mask = attn_mask
        self.pos_map = pos_map
    def apply_bert_fields(self,func):
        self.inp,self.seg,self.attn_mask = func(self.inp),func(self.seg),func(attn_mask)
    #def to_tensor(self):
    #self.apply_bert_fields()
        
    def get_input_position(self,*pos):
        return [self.pos_map[p] for p in pos]
    def to_dict(self):
        return {'question':self.question,'passage':self.passage,'input_ids':self.inp,'segment_ids':self.seg,'input_mask':self.attn_mask}

class BertInputConverter(): 
    def __init__(self,tokenizer,max_q_len,max_seq_len):
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_seq_len = max_seq_len
    # return dict
    def convert(self,question,passage):
        query_tokens = list(question)
        if len(query_tokens) > self.max_q_len:
            query_tokens = query_tokens[0:self.max_q_len]
        tokens, segment_ids = [], []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        # for pos map
        inc_n = len(tokens)
        pos_map = []
        for i,token in enumerate(list(passage)):
            tokens.append(token)
            segment_ids.append(1)
            pos_map.append(inc_n+i)
        tokens.append("[SEP]")
        segment_ids.append(1)
        # when lenght passage+question > max_seq_len , transform the postions which > max_seq_len 
        # to the last position of truncated passage (max_seq_len-2)
        if len(tokens) > self.max_seq_len:
            tokens[self.max_seq_len-1] = "[SEP]"
            pos_map = pos_map[0:self.max_seq_len-1-inc_n]+[self.max_seq_len-2  for i in range( self.max_seq_len-1-inc_n,len(pos_map))]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens[:self.max_seq_len])      ## !!! SEP
            segment_ids = segment_ids[:self.max_seq_len]
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) == len(segment_ids)
        return BertInput(question,passage,input_ids,segment_ids,input_mask,pos_map)
    
    def convert_examples(self,examples,span_field=None):
        for example in examples:
            res =  self.convert(example['question'],example['passage'])
            example.update(res.to_dict())
            if span_field is not None:
                example[span_field] = res.get_input_position(* example[span_field])

        return examples


class RecordDataset(object):
    #FIELD_MAP = {'passage':RAW_FIELD,'question':RAW_FIELD,'question_id':RAW_FIELD,'answers':RAW_FIELD,'question_type':RAW_FIELD}
    def __init__(self,sample_list,device=None):
        self.sample_list = sample_list
        self.device = device if device is not None else get_default_device()
        self.fields = self.get_fields()

    def get_fields(self):
        assert len(self.sample_list[0].keys())>0
        fields = [(k,RAW_FIELD) for k in self.sample_list[0]]
        return fields

class BertDataset( RecordDataset): 
    bert_field = Field(batch_first=True, sequential=True, tokenize=lambda ids:[int(i) for i in ids],use_vocab=False, pad_token=0) 
    def __init__(self,sample_list,converter,bert_field_names=[],device=None):
        super(BertDataset,self).__init__(sample_list,device)
        self.cvt = converter
        # fix in future
        if 'input_ids' not in sample_list[0]:
            self.sample_list = converter.convert_examples(sample_list)
        self._add_bert_fields(['input_ids','input_mask','segment_ids']+bert_field_names)
                
    def _add_bert_fields(self,bert_field_names):
        self.fields+= [ (name,self.bert_field) for name in bert_field_names]
    def make_dataset(self):
        l = []
        for sample in  self.sample_list:
            l.append(Example.fromdict(sample,{t[0]:t for t in self.fields}))
        dataset = Dataset(l,self.fields)
        return dataset
    def make_batchiter(self,batch_size=32):
        dataset  = self.make_dataset()
        return BucketIterator(dataset,batch_size=batch_size,sort_key=lambda x: len(x.input_ids),sort_within_batch=True,device=self.device)


class BertPointwiseRanker():
    @classmethod
    def from_config(cls,config):
        device = get_default_device()
        model,tokenizer = create_bert_model(config.get_values('pretrained_bert_path'),'ranker','%s/model.bin'%(config.get_values('save_dir')),device)
        bert_input_converter = BertInputConverter(tokenizer,config.get_values('max_q_len'),config.get_values('max_seq_len'))
        model.eval() 
        reader = BertPointwiseRanker(model,bert_input_converter,device)
        return reader
    
    def __init__(self,model,input_converter,device=None):
        if device is None:
            device = get_default_device()
        self.device = device
        self.model = model
        self.model = self.model.to( self.device)
        self.input_converter = input_converter


    def rank(self,example_dict,batch_size=16):
        examples = []
        for q,passage_list in example_dict.items():
            examples.extend( [(q,d['passage']) for d in passage_list] )
        batch_iter = BatchIter(examples,batch_size,self.numeralize_fn)
        predictions = predict_on_batch(self.model,batch_iter,sigmoid=True)
        result = aggregate_prediction(examples,predictions,labels=None,sort=True)
        sorted_dict = sort_preidction_by_score(result,attach_score=True)
        ret = {}
        for q, tuple_list in sorted_dict.items():
            pred_dict = tuple2dict(tuple_list,['question','passage','rank_score'])
            ret[q] = pred_dict
        return ret


    def evaluate_on_records(self,record_list,batch_size=128):
        iterator = self.get_batchiter(record_list,batch_size)
        return self.evaluate_on_batch(iterator)

    def get_batchiter(self,record_list,batch_size=64):
        dataset  = BertDataset(record_list,self.input_converter,device=self.device)
        iterator = dataset.make_batchiter(batch_size=batch_size)
        return iterator

    def evaluate_on_batch(self,iterator):
        with torch.no_grad():
            preds = []
            for  i,batch in enumerate(iterator):
                if (i+1) % 1000 == 0:
                    print('evaluate ranker on %d batch'%(i))
                match_scores = self.predict_score_one_batch(batch)
                if match_scores.is_cuda:
                    match_scores = match_scores.cpu()
                match_scores =  match_scores.numpy().tolist()
                batch_dct_list =  torchtext_batch_to_dictlist(batch)
                for j,item_dict in enumerate(batch_dct_list):
                    item_dict.update({'rank_score':match_scores[j]})
                    preds.append(item_dict)
        return  preds

    def predict_score_one_batch(self,batch):
        match_scores = self.model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask)
        match_scores  = torch.nn.Sigmoid()(match_scores) # N,2
        match_scores  =  match_scores[:,1]+0.0000000001 #  N  ,get positve socre
        return match_scores

class BertReader():
    @classmethod
    def from_config(cls,config):
        device = get_default_device()
        model,tokenizer = create_bert_model(config.get_values('pretrained_bert_path'),'reader','%s/model.bin'%(config.get_values('save_dir')),device)
        bert_input_converter = BertInputConverter(tokenizer,config.get_values('max_q_len'),config.get_values('max_seq_len'))
        
        if  'decoder' not in config.json_obj:
            decoder = create_decoder(None)
        else:
            decoder = create_decoder(config.json_obj["decoder"])
            
        model.eval() 
        reader = BertReader(model,bert_input_converter,decoder,device)
        return reader
    
    def __init__(self,model,input_converter,decoder,device=None):
        if device is None:
            device = get_default_device()
        self.device = device
        self.model = model
        self.model = self.model.to( self.device)
        self.decoder = decoder
        #bert-base-chinese
        self.input_converter = input_converter
    
    # record : list of dict  [ {field1:value1,field2:value2...}}]
    def evaluate_on_records(self,records,batch_size=64):
        iterator = self.get_batchiter(records,batch_size=batch_size)
        return  self.evaluate_on_batch(iterator)
    
    def get_batchiter(self,records,batch_size=64):
        dataset  = BertDataset(records,self.input_converter,device=self.device)
        iterator = dataset.make_batchiter(batch_size=batch_size)
        return iterator

    def evaluate_on_batch(self,iterator):
        preds = []
        with torch.no_grad():
            for  i,batch in enumerate(iterator):
                if (i+1) % 1000 == 0:
                    print('evaluate reader on %d batch'%(i))
                preds.extend(self.predict_one_batch(batch))
        return  preds

    def predict_one_batch(self,batch):
        start_probs, end_probs = self.model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask)
        return self.decode_batch(start_probs, end_probs,batch)

    def decode_batch(self,start_probs,end_probs,batch):
        batch_dct_list =  torchtext_batch_to_dictlist(batch)
        preds = []
        for j in range(len(start_probs)):
            sb,eb = start_probs[j], end_probs[j]
            sb ,eb  = sb.cpu().numpy(),eb.cpu().numpy()
            text = "$" + batch.question[j] + "\n" + batch.passage[j]
            answer,score,_ = self.decoder.decode(sb,eb,text)
            #score = score.item() #輸出的score不是機率 所以不會介於0~1之間
            batch_dct_list[j].update({'span':answer,'span_score':score})
            preds.append(batch_dct_list[j]) 
        return preds