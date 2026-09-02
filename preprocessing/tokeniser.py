import re
import tiktoken
import vocabmaker

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

class bpeencoder:
    def __init__(self) -> None:
        self.tokeniser=tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    with open("practicefiles/theverdict.txt","r") as f:
        buffer=f.read()
    tokeniser=mytokeniser(vocabmaker.vocabmaker(buffer).getvocab())
