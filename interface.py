from melkor_parser.model import *

corpus = ["I am melkor",
          "I am mal ganis",
          "cross a 3 and add the 2"]
cfg_diction = {"+":{"output_type":"int","input_types":["int","int"]},
               "1":{"output_type":"int","input_types":None},
               "2":{"output_type":"int","input_types":None}}

def make_melkor_parser(corpus,cfg_diction,word_dim=256,signal_dim=32,key_dim=42,latent_dim=132):
    cfg = ContextFreeGrammar(cfg_diction,signal_dim)
    decoder = Decoder(signal_dim,key_dim,latent_dim,cfg)
    model = LanguageParser(word_dim,signal_dim,decoder,corpus)
    return model

model = make_melkor_parser(corpus,cfg_diction,signal_dim=132,key_dim=44)

data = [
    ["I am mal ganis",["+","1","+","2","1"]],
    ["I am melkor",["+","1","2"]],
    ["cross a 3 and add the 2",["+","+","2","1","+","1","2"]],
]

optim = torch.optim.Adam(model.parameters(),lr=2e-3)
for epoch in range(100):
    optim.zero_grad()
    l = 0
    for bind in data:
        p,l2 = model(bind[0],bind[1])
        l += l2
    l.backward()
    optim.step()

model.monte_carlo_enabled = False
for bind in data:
    print(model(bind[0])[0])