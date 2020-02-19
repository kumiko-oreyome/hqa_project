from .config import elastic_host,elastic_port,elastic_index_cmkb,elastic_doc_type_cmkb,cmkb_doc_file 
from multidoc.util import jsonl_reader,jsonl_writer
from .dao.cmkb import CMKBElasticDB
def build_elastic_cmkb_lib_from_file(host,port,index,doc_type,filepath,delete_previous=False):
    db = CMKBElasticDB(host,port,index,doc_type)
    if delete_previous :
        db.delete_all_docs()
    db.insert_library_docs_from_file(filepath)

def clear_duplicate_doc(filepath):
    l = []
    title_set = set()
    for sample in jsonl_reader(filepath):
        if sample['title'] not in title_set: 
            l.append(sample)
            title_set.add(sample['title'])
    print('total  samples after remove dup docs %d'%(len(l)))
    jsonl_writer(filepath,l)


if __name__ == '__main__':
    print('dump document to elk search from file %s'%(cmkb_doc_file))
    print('remove dup docs')
    clear_duplicate_doc(cmkb_doc_file)
    build_elastic_cmkb_lib_from_file(elastic_host,elastic_port,elastic_index_cmkb,elastic_doc_type_cmkb,cmkb_doc_file)