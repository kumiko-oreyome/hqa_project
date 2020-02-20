import traceback
import httpx
from .mrc  import  create_app,create_mrc_model_for_server



def create_mrc_proxy(config):
    class_name,kwargs = config["class"],config["kwargs"]
    name2class = {'TestProxy':TestProxy,'RedirectProxy': RedirectProxy,'DirectAccessProxy':DirectAccessProxy}
    _cls = name2class[class_name]   
    return _cls(**kwargs)



class TestProxy():
    def __init__(self,config):
        self.app = create_app(config)
    def send_mrc_input(self,mrc_input,answer_num=3,algo_version=0):
        with self.app.test_client() as c:
            rv = c.post('/qa', json={'mrc_input':mrc_input,'answer_num':answer_num,'algo_version':algo_version})
        return rv.get_json()


class RedirectProxy():
    def __init__(self,server_url):
        self.server_url = server_url
    def send_mrc_input(self,mrc_input,answer_num=3,algo_version=0):
        try:
            r = httpx.post(self.server_url,json={'mrc_input':mrc_input,'answer_num':answer_num,'algo_version':algo_version},timeout=120)
        except Exception as e:
            print('##########################')
            traceback.print_exc()
            print('##########################')
            print(str(e))
            print('some error occur while redirect to mrc server %s'%(self.server_url))
            return {'result':'failed','message':'some error occur while redirect'}
        try:
            return r.json()
        except:
            return {'result':'failed','message':str(r)}



# mrc model in the same process of qa server
class DirectAccessProxy():
    def __init__(self,config):
        self.model =  create_mrc_model_for_server(config)
    def send_mrc_input(self,mrc_input,answer_num=3,algo_version=0):
        return self.model.get_answer_list(mrc_input,answer_num)