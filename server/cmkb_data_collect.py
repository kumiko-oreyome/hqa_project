import asyncio
from multidoc.util import jsonl_writer
from .config import cmkb_keywords,cmkb_doc_file
from .webrequest import  CMKBRequest

event_loop = asyncio.get_event_loop()
def collect_data_by_keywords():
    keywords =  cmkb_keywords
    l = []
    req = CMKBRequest(event_loop)
    for keyword in keywords:  
        docs = req.crawl_keyword_search_page(keyword,100)
        l.extend(docs)
    jsonl_writer(cmkb_doc_file,l)

collect_data_by_keywords()