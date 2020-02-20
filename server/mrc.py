import httpx
from .config import MRC_SERVER_CONFIG 
from flask import Flask,request,jsonify
from multidoc.dataset.dureader import DureaderExample,flatten_examples
from multidoc.util import RecordGrouper,group_dict_list,get_default_device
from multidoc.core.op import create_reader,create_ranker,create_paragraph_selector,TopKJudger


def create_app(config):
    print('create app')
    app = Flask(__name__)
    #app.config['DEBUG'] = True 
    app.config.from_mapping(
        SECRET_KEY='key'
    )
    print('create model')
    model = create_mrc_model_for_server(config["model"])
    @app.route('/qa',methods=['POST'])
    def qa_by_multi_mrc():
        print('qa_by_multi_mrc')
        if request.json is None:
            return jsonify({'result':'failed','message':'request is not json'})
        if not check_request_valid(request.json ):
            return jsonify({'result':'failed','message':'request fields is not valid'})
        if not check_mrc_input_valid(request.json["mrc_input"]):
            return jsonify({'result':'failed','message':'invalid mrc input format'})
        answer_list = model.get_answer_list(request.json["mrc_input"],request.json["answer_num"])
        return jsonify({'result':'success','message':'mrc success','answers':answer_list})
    return app


def check_request_valid(req_json):
    if 'mrc_input' not in req_json:
        print('mrc_input field not in request')
        return False
    if 'answer_num' not in req_json:
        print('answer_num field not in request')
        return False
    if 'algo_version' not in req_json:
        print('algo_version field not in request')
        return False
    return True

def check_mrc_input_valid(json_obj):
    if 'question' not in json_obj:
        print('question is not provided')
        return False
    if 'documents' not in json_obj or len(json_obj['documents'])==0:
        print('documents is not provided')
        return False
    for doc in json_obj['documents']:
        if 'url' not in doc or 'title' not in doc:
            print('url or title not in doc ')
            print(doc)
            return False
        if 'paragraphs' not in doc:
            print('paragraph not in doc')
            print(doc)
            return False
    return True




def create_mrc_model_for_server(config):
    cls_name,kwargs = config['class'],config['kwargs']
    if cls_name == 'MockMRCModel' or cls_name == 'mock':
        model = MockMRCModel()
    elif cls_name in ['SelectorReaderModel']:
        name2cls = {'RankerReaderModel':None,'SelectorReaderModel':SelectorReaderModel}
        _cls = name2cls[cls_name]
        model = _cls.from_config(kwargs)
    else:
        print('cannot find class %s please set correct config class!'%(_cls))
        assert False
    return model



class MockMRCModel():
    def __init__(self):
        pass
    def get_answer_list(self,mrc_input,k=3):
        x = DureaderExample(mrc_input)
        l = x.flatten(['question'],['url','title'])
        l = l [0:k]
        ret = []
        for x in l:
            if len( x['passage'])>10:
               answer = x['passage'][10:20]
            else:
               answer  = x['passage'][0:10]
            ret.append({'paragraph':x['passage'],'answer':answer,'title':x['title'],'url':x['url']})
        return ret   
    

class SelectorReaderModel():
    @classmethod
    def from_config(cls,config):
        selector = create_paragraph_selector(config['selector']) 
        reader =  create_reader(config['reader'])
        return SelectorReaderModel(selector,reader)
    
    def __init__(self,selector,reader):
        self.selector = selector
        self.reader = reader
    
    def get_answer_list(self,mrc_input,k=3):
        mrc_input['id'] = 0
        x = DureaderExample(mrc_input)
        records = x.flatten(['question','id'],['url','title'])
        selected_records  = self.selector.paragraph_selection(records) 
        reader_results = self.reader.evaluate_on_records(selected_records,batch_size=128)
        reader_results = group_dict_list(reader_results,'id')
        ret_list= TopKJudger(k=k).judge(reader_results)[0]
        ret = []
        for x in ret_list:
            ret.append({'paragraph':x['passage'],'answer':x['span'],'title':x['title'],'url':x['url']})
        return ret


if __name__ == '__main__':
    app_config = MRC_SERVER_CONFIG 
    app = create_app( app_config)
    app.run(debug=False,port=app_config["port"],threaded=True)