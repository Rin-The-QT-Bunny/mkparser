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
cfg_diction = {"+":{"output_type":"int","input_types":["int","int"]},
               "1":{"output_type":"int","input_types":None},
               "2":{"output_type":"int","input_types":None}}

cfg = ContextFreeGrammar(cfg_diction,32)
decoder = Decoder(32,42,132,cfg)

class LanguageParser(nn.Module):
    def __init__(self,word_dim,signal_dim,decoder,corpus = []):
        super().__init__()
        self.vector_construtor = VectorConstructor(word_dim,corpus)
        self.encoder = GRUEncoder(word_dim,signal_dim)
        self.decoder = decoder
        self.token_features = nn.Parameter(torch.randn([3,42]))
        self.token_keys = ["+","1","2"]
    def forward(self,sentence,dfs_seq = None):
        z_code = self.encoder(self.vector_construtor(sentence))
        p,l = self.decoder(z_code,self.token_features,self.token_keys,dfs_seq)
        return p,l



model = LanguageParser(256,32,decoder,corpus)


optim = torch.optim.Adam(model.parameters(),lr=2e-3)
for epoch in range(200):
    optim.zero_grad()
    p,l = model("I am mal ganis",["+","+","1","2","+","2","1"])
    p2,l2 = model("I am melkor",["+","1","2"])
    l = l + l2
    l.backward()
    optim.step()
    if (epoch%100==0):
        print(p,l)

model.monte_carlo_enabled = False
p,l = model("I am mal ganis")
print(p)
p,l = model("I am melkor")
print(p)