from .config import mrc_server_config
from . import mrc 

class TestMrcApp():
    def __init__(self,config):
        self.app = mrc.create_app(config)
    def multi_mrc(self,mrc_input,answer_num=3,algo_version=0):
        with self.app.test_client() as c:
            rv = c.post('/qa', json={'mrc_input':mrc_input,'answer_num':answer_num,'algo_version':algo_version})
        return rv.get_json()




def test_mrc_server():
    doc_1 = {'title':'c8763','url':'www.c8763.com','paragraphs':["c8763是某黑色劍士的技能","發動c8763需要隊友幫稱十秒","結束了嗎"]}
    doc_2 = {'title':'sao','url':'www.sao.com','paragraphs':["因為太過中二所以c8763會被噓","c8763後來有很多被人惡搞的梗"]}
    test_mrc_input = {"question":"c8763是什麼","documents":[doc_1,doc_2]}
    app = mrc.create_app(mrc_server_config)
    print('send data')
    with app.test_client() as c:
        rv = c.post('/qa', json={'mrc_input':test_mrc_input,'answer_num':2,'algo_version':0})
    json_data = rv.get_json()
    print(json_data)

if __name__ == '__main__':
    test_mrc_server()