from lxml import etree,html
import re
from multidoc.util import  jsonl_writer,jsonl_reader
from ..webrequest import Downloader
from urllib.parse import quote
import asyncio,time
from pyppeteer import launch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Search
from elasticsearch_dsl import Q
import random

##  pyppeteer.errors.NetworkError: Protocol Error (Runtime.callFunctionOn): Session closed. Most likely the page has been closed.   
##  SOLVE:https://github.com/miyakogi/pyppeteer/issues/178

class CMKBDLibraryDocument():
    def __init__(self,url,title,body,tags,paragraphs):
        self.url = url 
        self.title = title
        self.body = body
        self.tags = tags
        self.paragraphs = paragraphs

    def to_json(self):
        return {'url':self.url,'title':self.title,'tags':self.tags,'paragraphs':self.paragraphs,'body':self.body}



class CMKBElasticDB():
    def __init__(self,host,port,index,doc_type):
        self.es = Elasticsearch([{'host':host,'port':port}])
        self.index= index
        self.doc_type= doc_type

    def create_index(self):
        pass

    def create_library_docs(self):
        pass

    def insert_library_docs_from_file(self,filepath):
        insert_dict = []
        for json_obj in jsonl_reader(filepath):
            d =  CMKBDLibraryDocument(**json_obj).to_json()
            d.update({ "_index": self.index,"_type": self.doc_type})
            insert_dict.append(d)

        status,_ = bulk(self.es,insert_dict,index=self.index)
        print(status)

    def insert_doc(self,doc):
        pass

    def delete_all_docs(self):
        s = Search(index=self.index).using(self.es)
        s.update_from_dict({"query":{"match_all":{}},"size":1000})
        response = s.execute()
        actions = []
        for h in response.hits:
            actions.append({ '_op_type': 'delete',"_index" : self.index, "_id" : h.meta.id,'_type': self.doc_type })
        status,_ = bulk(self.es,actions)
        print("delete")
        print(status)
    #TODO
    # size bug in update_from_dict... size not working
    def retrieve_library_doc(self,keywords,size=10,search_fields=["title^3","tags^3","paragraphs"]):
        if type(keywords) == str:
            keywords = [keywords]
        query = " ".join(keywords)
        s = Search(index=self.index).using(self.es)
        s.update_from_dict({"query": {"simple_query_string" : {"fields" : search_fields,"query" :query}}})
        res = s.execute()
        l = []
        for d in res.hits:
            l.append(CMKBDLibraryDocument(url=d.url,title=d.title,body=d.body,tags=d.tags,paragraphs=[ s for s in d.paragraphs]).to_json())
        return l
        
    def get_results(self,res):
        l = []
        for hit in res['hits']['hits']:
            l.append(hit['_source'])
        return l