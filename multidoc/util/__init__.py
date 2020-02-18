import json
import torch
import pandas as pd
from types import SimpleNamespace
from .textutil import Tokenizer,Tfidf,to_simplified_sentences

class JsonConfig():
    @classmethod
    def save(cls,path,obj):
        write_json(path,obj)
    
    def __init__(self,path):
        self.path = path
        self.json_obj = self._load()
    
    def _load(self):
        return read_json(self.path)
    
    def get_values(self,*names):
        if len(names)==1 and type(names[0])==str:
            return self._get_value(names[0])
        return [ self.json_obj[name] for name in names]
    
    def _get_value(self,name):
        return self.json_obj[name]
    # args : parse_arg
    def combine_args(self,args):
        pass
    

class MetricTracer():
    def __init__(self):
        self.tot = 0
        self.n = 0
    def zero(self):
        self.tot = 0
        self.n = 0
    def add_record(self,record):
        self.tot+=record
        self.n+=1
    def avg(self):
        if self.tot == 0:
            return 0
        return self.tot/self.n
    def print(self,prefix=''):
        print('%s%.3f'%(prefix,self.avg()))

class RecordGrouper():
    def __init__(self,records):
        self.records = records

    @classmethod
    def from_group_dict(cls,group_key,structured):
        records = []
        for k,l in structured.items():
            for v in l:
                v.update({group_key:k})
                records.append(v)
        return cls(records)
    
    def group(self,field_name):
        df = pd.DataFrame.from_records(self.records)
        gp = df.groupby(field_name).apply(lambda x:x.to_dict('records')).to_dict()
        return gp

    def group_sort(self,group_key,sort_key,k=None):
        df = pd.DataFrame.from_records(self.records)
        gp = df.groupby(group_key).apply(lambda x:x.sort_values(by=sort_key, ascending=False).head(k if k is not None else len(x) ).to_dict('records')).to_dict()
        return gp
            
    def to_records(self):
        l = pd.DataFrame.from_records(self.records).to_dict('records')
        return l


def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

def write_json(path,json_obj):
    with open(path,'w',encoding='utf-8') as f:
        f.write(json.dumps(json_obj,ensure_ascii=False))   

def jsonl_reader(path):
    with open(path,'r',encoding='utf-8') as f:
        for line in f :
            json_obj = json.loads(line.strip(),encoding='utf-8')
            yield json_obj

def jsonl_writer(path,l):
    with open(path,'w',encoding='utf-8') as f:
        for o in l :
            f.write(json.dumps(o,ensure_ascii=False)+'\n')  

def torchtext_batch_to_dictlist(batch):
    d = {}
    fields_names = list(batch.fields)
    d = { k:getattr(batch,k) for k in fields_names if not isinstance(getattr(batch,k),torch.Tensor)}
    l = pd.DataFrame(d).to_dict('records')
    return l
    



def group_dict_list(dictlist,key,apply_fn=None):
    ret = {}
    for obj in dictlist:
        value = obj[key]
        if value not in ret:
            ret[value] = []
        ret[value].append(obj)
    if apply_fn is None:
        return ret
    for key,v in ret.items():
        ret[key] = apply_fn(v)
    return ret


def group_tuples(tuple_list,item_index,contains_key=False):
    table = {}
    for t in tuple_list:
        key = t[item_index]
        if key not in table:
            table[key] = []
        if contains_key:
            item = t
        else:
            item = tuple(( x for i,x in enumerate(t) if i!=item_index))
        table[key].append(item)
    return table

def tuple2dict(tuples,key_names):
    convert_fn = lambda t: { key:item   for item,key in zip(t,key_names)} 
    if isinstance(tuples,tuple):
        assert len(tuples) == len(key_names)
        return convert_fn(tuples)
    else:
        assert len(tuples[0]) == len(key_names)
    return [  convert_fn(tup)  for tup in tuples]

def load_json_config(path,to_attr=True,**extra_attr):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    data.update(extra_attr)
    ret  = data
    if to_attr:
        ret = SimpleNamespace(**data)
    return ret

def get_default_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device