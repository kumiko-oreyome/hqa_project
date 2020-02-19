import os
from collections import OrderedDict
from multidoc.util import jsonl_reader,jsonl_writer 



class JsonlRCDatabase():
    CRAWLER_SCHEMA = {
        'sample_fields':['documents','question','question_id'],
        'document_fields':['doc_id','body','url'],
        'collection_key':'documents'
    }

    def __init__(self,path,sample_id_field):
        self.path = path
        self.sample_id_field = sample_id_field
        if  not os.path.exists(self.path):
            assert False
        self.load_samples()
    def load_samples(self):
        self.samples = OrderedDict()
        for data in jsonl_reader(self.path):
            sample_id = data[self.sample_id_field]
            if sample_id in self.samples:
                print('duplidate id %s'%(str(sample_id)))
                assert False
            self.samples[sample_id] = data
        print('total %d samples'%(len(self.samples)))

    def add_new_sample(self,data):
        assert self.sample_id_field not in data
        sample_id = self.next_sample_id()
        data[self.sample_id_field] = sample_id
        self.samples[sample_id] = data

    def update_sample(self,new_data):
        assert self.sample_id_field in new_data
        sample_id = new_data[self.sample_id_field]
        assert sample_id in self.samples
        self.samples[sample_id] = new_data


    def next_sample_id(self):
        if len(self.samples.keys()) == 0:
            return 0
        return max(self.samples.keys())+1

    def save(self):
        jsonl_writer(self.path,self.samples.values())

    def get_sample_by_id(self,id_value):
        if id_value not in self.samples:
            return None
        return self.samples[id_value]

    def del_sample_by_id(self,id_value):
        del self.samples[id_value]
    

    def reset(self):
        self.load_samples()