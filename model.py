from decoder import *
from encoder import *
from utils import *

class VectorConstructor(nn.Module):
    def __init__(self,word_dim,corpus,max_vocab=99999):
        super().__init__()
        # construct vector embeddings and other utils for given corpus
        self.corpus = corpus
        self.key_words = []
        self.token_to_id = build_vocab(corpus)
        self.id_to_token = reverse_diction(self.token_to_id)
        self.word_vectors = nn.Embedding(max_vocab,word_dim)

    def forward(self,sentence):
        code = encode(tokenize(sentence),self.token_to_id)
        word_vectors = self.word_vectors(torch.tensor(code))
        return word_vectors
    
    def get_word_vectors(self,sentence):
        return sentence

corpus = ["I am melkor",
          "I am mal ganis"]

vc = VectorConstructor(256,corpus)

cfg_diction = {"+":{"output_type":"int","input_types":["int","int"]},
               "1":{"output_type":"int","input_types":None},
               "2":{"output_type":"int","input_types":None}}

cfg = ContextFreeGrammar(cfg_diction,32)
model = Decoder(32,42,132,cfg)


encoder = GRUEncoder(256,32)
print(encoder(vc("I am melkor")).shape)

token_features = torch.randn([3,42])
input_signal = torch.randn([1,32])


optim = torch.optim.Adam(model.parameters(),lr=2e-3)
for epoch in range(1000):
    optim.zero_grad()
    p,l = model(input_signal,token_features,["+","1","2"],["+","+","1","2","+","2","1"])
    l.backward()
    optim.step()
    if (epoch%100==0):
        print(p,l)

model.monte_carlo_enabled = False
p,l = model(input_signal,token_features,["+","1","2"])
print(p)