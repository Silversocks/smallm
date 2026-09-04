import re

class vocabmaker:
    def __init__(self,raw_text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        result = [i for i in result if i.strip()]
        tokens=sorted(set(result))
        tokens.extend(["|EoF|","<|unk|>"])
        self.vocab={token:integer for integer,token in enumerate(tokens)}
    def getvocab(self):
        return self.vocab

if __name__ == "__main__":
    with open("practicefiles/theverdict.txt","r") as f:
        buffer=f.read()
    print(vocabmaker(buffer).getvocab()[-10])
