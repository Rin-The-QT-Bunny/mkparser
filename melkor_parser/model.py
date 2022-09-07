from .decoder import *
from .encoder import *
from .utils import *

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

class LanguageParser(nn.Module):
    def __init__(self,word_dim,signal_dim,decoder,corpus = []):
        super().__init__()
        self.vector_construtor = VectorConstructor(word_dim,corpus)
        self.encoder = GRUEncoder(word_dim,signal_dim)
        self.decoder = decoder
        cfg = self.decoder.cfg
        self.token_keys = []
        for term in cfg.cfg_diction:
            if not term in self.token_keys:
                self.token_keys.append(term)
        L = len(self.token_keys)
        self.token_features = nn.Parameter(torch.randn([L,self.decoder.k_dim]))
    def forward(self,sentence,dfs_seq = None):
        z_code = self.encoder(self.vector_construtor(sentence))
        p,l = self.decoder(z_code,self.token_features,self.token_keys,dfs_seq)
        return p,l
