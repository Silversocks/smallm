import re

with open("practicefiles/theverdict.txt","r") as f:
    raw_text=f.read()

class vocabmaker:
    def __init__(self,raw_text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        result = [i for i in result if i.strip()]
        tokens=sorted(set(result))
        tokens.extend("|EoF|","<|unk|>")
        self.vocab={token:integer for integer,token in enumerate(tokens)}
    def getvocab(self):
        return self.vocab

class mytokeniser:
    def __init__(self,vocab:dict):
        self.strtoint=vocab
        self.inttostr={integer:token for token,integer in vocab.items()}
    def encode(self,text:str):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.strtoint else "<|unk|>" for item in preprocessed]
        ids=[self.strtoint[i] for i in preprocessed]
        return ids
    def decode(self, ids:list):
        text = " ".join([self.inttostr[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
