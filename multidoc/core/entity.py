class MatchResult():
    def __init__(self,results=[]):
        self.results = results
    def add_result(self,result):
        self.results.append(result)
    def rank(self,rank_fn,reverse=True):
        return list(sorted(self.results,key=rank_fn,reverse=reverse))
    def select(self,select_fn):
        return select_fn(self.results)
    def select_by_rank(self,rank_fn,reverse,top_k=1):
        return self.select(lambda results:self.rank(rank_fn,reverse=reverse)[0:top_k])

class MultiDocExample():
    @classmethod 
    def from_question_and_paragraphs(cls,question,paragraphs):
        if len(paragraphs) == 0:
            return  MultiDocExample(question,paragraphs)
        if isinstance(paragraphs[0],Paragraph):
            return  MultiDocExample(question,paragraphs)
        return MultiDocExample(question,[Document.from_string_or_tokens(p) for p in paragraphs])

    def __init__(self,question,documents):
        self.question = question
        self.documents = documents

    def match_all_paragraphs(self,score_fn):
        r = MatchResult()
        for doc in self.get_all_documents():
            for p in doc.get_all_paragraphs():
                r.add_result((p,score_fn(p)))
        return r

    def match_paragraphs_every_document(self,score_fn):
        for doc in self.get_all_documents():
            r = MatchResult()
            for p in doc.get_all_paragraphs():
                r.add_result((p,score_fn(p)))
            yield r
        
    def get_document(self,idx):
        return self.documents[idx]

    def get_all_documents(self):
        for di in range(len(self.documents)):
            yield self.get_document(di)


class Document():
    @classmethod 
    def from_string_or_tokens(cls,str_or_tokens):
        if len(str_or_tokens) == 0:
            return  Document(str_or_tokens)
        return [Paragraph(item) for item in str_or_tokens]
    def __init__(self,paragraphs):
        self.paragraphs = paragraphs
    def match_over_paragraphs(self,score_fn):
        r = MatchResult()
        for p in self.get_all_paragraphs():
             r.add_result((p,score_fn(p)))
        return r
    def get_all_paragraphs(self):
        for pi in range(len(self.paragraphs)):
            yield self.get_paragraph(pi)
    def get_paragraph(self,idx):
        return self.paragraphs[idx]


class Paragraph():
    def __init__(self,tokens):
        self.tokens = tokens
    def enumerate_spans(self):
        for i in range(len(self.tokens)):
            for j in range(i,len(self.tokens)):
                span = (i,j)
                yield span,self.tokens[i:j+1]

    def find_char_span(self,tokens):
        s_self = self.stringfy()
        s_tokens = Paragraph(tokens).stringfy()
        cstart =  s_self.find(s_tokens)
        if cstart == -1:
            return -1,-1
        return cstart,cstart + len(s_tokens)-1

        
    def get_span(self,span):
        return self.tokens[span[0]:span[1]+1]
    def stringfy(self):
        if type(self.tokens) == str:
            return self.tokens 
        return ''.join(self.tokens)




