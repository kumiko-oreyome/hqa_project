import asyncio
import random
from flask import (
    Flask,Blueprint, flash, g, redirect, render_template, request, session, url_for,Markup,jsonify
)
from . import config 
from .config import MRC_PROXY_CONFIG,WEB_RETRIEVER_CONFIG,ELK_RETRIEVER_CONFIG
from .mrc_proxy import create_mrc_proxy, DirectAccessProxy
from .docretv import  create_document_retriever
from .mrc import SelectorReaderModel
from multidoc.core.op import create_paragraph_selector

event_loop = asyncio.get_event_loop()


class RequestQA():
    @classmethod
    def is_valid(cls,json_obj):
        if 'question' not in json_obj:
            print('question not in request json')
            return False
        if 'answer_num' not in json_obj:
            print('answer num not in request json')
            return False
        if 'answer_num' in json_obj and  type(json_obj['answer_num'])!=int:
            print('answer num must be int')
            return False
        if 'algo_version' not in json_obj:
            print('algo_version not in request json')
            return False
        return True
        
    def __init__(self,json_req):
        self.json_req = json_req
        self.parse()
    def parse(self):
        self.question =self.json_req['question']
        self.answer_num = int(self.json_req['answer_num'])
        self.algo_version = int(self.json_req['algo_version'])


class FakeReader():
    def __init__(self):
        pass
    def evaluate_on_records(self,records,batch_size=128):
        results = random.sample(records,k=2)
        for x in results:
            x['span_score'] =1.0
            x['span'] = ''
        return results




def wrap_response(json):
    if 'result' not in json and 'answers' in json:
        json['result'] = 'success'
    return json


def handle_mrc_result(req,response):
    if response['result']!='success':
        return  jsonify(response)
    for answer in response['answers']:
        start_pos = answer['paragraph'].find(answer['answer'] )
        answer['answer_pos'] = [start_pos,start_pos+len(answer['answer'])-1]
    response.update({'question':req.question,'algo_version':req.algo_version})
    return jsonify(wrap_response(response))

def create_app():
    app = Flask(__name__)
    #app.config['DEBUG'] = True 
    app.config.from_mapping(
        SECRET_KEY='key'
    )
    print('initialze server')
    web_retriever = create_document_retriever(WEB_RETRIEVER_CONFIG,event_loop)
    elk_retriever = create_document_retriever(ELK_RETRIEVER_CONFIG,event_loop)
    mrc_proxy = create_mrc_proxy(MRC_PROXY_CONFIG)
    print('complete initialze')
    bp = create_bp(app,web_retriever,elk_retriever,mrc_proxy)
    app.register_blueprint(bp)
    return app

def create_bp(app,web_retriever,elk_retriever,mrc_proxy):
    bp = Blueprint('api', __name__, url_prefix='/api')
    import jieba  
    jieba.analyse.set_stop_words(config.stopwords_path)

    @bp.before_request
    def check_request_json():
        if request.json is None:
            return jsonify({'result':'failed','message':'request is not json'})
        if  not RequestQA.is_valid(request.json):
            return  jsonify({'result':'false','message':'json is not valid for qa'})

    @bp.route('/fakeqa',methods=['POST'])
    def fake_qa():
        req = RequestQA(request.json)
        paragraphs = []
        response = {'question':req.question,'algo_version':req.algo_version}
        fake_paragraphs = ['你怎麼不問神奇海螺','發大財','國家機器動得很勤勞']
        for i in range(req.answer_num):
            if i >= 3:
                paragraph = '段落%d:\n 你要的還真多,貪心的人,我沒梗了'%(i+1)
            else:
                paragraph = fake_paragraphs[i]
            paragraphs.append({'paragraph':paragraph,'answer':paragraph[0:2],'title':'這是標題[%d]拉哈哈'%(i+1),'url':'https://www.google.com.tw'})
        response['answers'] = paragraphs
        return jsonify(wrap_response(response))

    
    @bp.route('/webqa',methods=['POST'])
    def qa_by_websearch():
        print('qa_by_websearch')
        req = RequestQA(request.json)
        mrc_input = web_retriever.retrieve_candidates(req.question)
        response = mrc_proxy.send_mrc_input(mrc_input)
        return handle_mrc_result(req,response)


    
    @bp.route('/kbqa',methods=['POST'])
    def qa_by_cmkb():
        print('qa_by_cmkb')
        req = RequestQA(request.json)
        mrc_input = elk_retriever.retrieve_candidates(req.question)
        response = mrc_proxy.send_mrc_input(mrc_input)
        return handle_mrc_result(req,response)

    
    @bp.route('/fastqa',methods=['POST'])
    def fast_qa_by_cmkb():
        print('fast qa by cmkb')
        req = RequestQA(request.json)
        mrc_input = elk_retriever.retrieve_candidates(req.question)
        ranker = create_paragraph_selector({'class':'RankBasedSelector','kwargs':{'ranker':config.word_match_ranker,'k':1}})
        reader = FakeReader()
        model = SelectorReaderModel(ranker,reader)
        proxy = DirectAccessProxy(model)
        response = proxy.send_mrc_input(mrc_input)
        return handle_mrc_result(req,response)
    
    return bp





if __name__ == '__main__':
    app = create_app()
    app.run(debug=False,host='0.0.0.0',port=5000,threaded=True)