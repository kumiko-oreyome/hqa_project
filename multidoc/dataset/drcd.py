import json
from multidoc.util import read_json,write_json,jsonl_reader,jsonl_writer


class DrcdPreprocessing():
    def __init__(self):
        pass

    def save_first_n_data(self,path,save_path,n):
        ds = DrcdDataset.from_path(path)
        ds.examples = ds.examples[:n]
        ds.save(save_path)


    def dureader_transform_on_file(self,path):
        ds = DrcdDataset.from_path(path)
        print('total %d drcd examples'%(len(ds.examples)))
        for i,example in enumerate(ds.examples):
            if (i+1)%100 == 0:
                print('process %d-th example'%(i))
            for dureader_example in self.dureader_transform_on_example(example):
                yield dureader_example

    def dureader_transform_on_example(self,example_obj):
        qas = []
        paragpraphs = []
        for pi,paragraph_obj in enumerate(example_obj["paragraphs"]):
            pargraph_text = paragraph_obj["context"]
            paragpraphs.append(pargraph_text)
            for o in paragraph_obj["qas"]:
                if "answers" not in o or len(o["answers"])==0:
                    continue
                answer = o["answers"][0]["text"]
                start_pos = pargraph_text.find(answer)
                assert start_pos>=0
                end_pos = start_pos+len(answer)-1
                qas.append({"question_id":o["id"],"question":o["question"],"answers":[answer],"answer_docs":[0],"answer_spans":[[start_pos,end_pos]],"pi":pi})
        for x in qas:
            x["documents"] = [{"paragraphs":paragpraphs,"most_related_para":x["pi"]}]
            del x["pi"]
        return qas

        

class DrcdDataset():
    @classmethod
    def from_path(cls,path):
        if path.endswith('.json'):
            o = read_json(path)
            examples = cls.parse_json(o)
        elif path.endswith('.jsonl'):
            examples = list(jsonl_reader(path))
        return DrcdDataset(examples)
    @classmethod
    def parse_json(cls,obj):
        l = []
        for data in obj["data"]:
            l.append(data)
        return l

    def __init__(self,examples):
        self.examples = examples
    
    def get_example_by_id(self):
        pass

    def get_examples(self,start=0,end=None):
        if end is None:
            end = len(self.examples)
        return self.examples[start:end]

    def to_json(self):
        return {"data":self.examples}

    def save(self,path,form='jsonl'):
        if form=='jsonl':
            jsonl_writer(path,self.examples)
        elif form=='json':
            write_json(path,self.to_json())
        else:
            assert False

def dump_first_n_data_drcd(path,save_path,n):
    DrcdPreprocessing().save_first_n_data(path,save_path,n)

def dureader_transform(path,save_path):
    with open(save_path,'w',encoding='utf-8') as f:
        for example in DrcdPreprocessing().dureader_transform_on_file(path):
            f.write(json.dumps(example,ensure_ascii=False)+'\n')
if __name__ == '__main__':
    #N = 10
    #dump_first_n_data_drcd('./data/DRCD_training.json','./data/DRCD_training.%d.jsonl'%(N),N)
    dureader_transform('./data/DRCD_dev.json','./data/DRCD_dev.dureader.json')
    #dureader_transform('./data/DRCD_training.%d.jsonl'%(N),'./data/DRCD_training.%d.dureader.json'%(N))
    dureader_transform('./data/DRCD_training.json','./data/DRCD_training.dureader.json')