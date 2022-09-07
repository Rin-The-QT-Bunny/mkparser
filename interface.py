from model import *

corpus = ["I am melkor",
          "I am mal ganis"]
cfg_diction = {"+":{"output_type":"int","input_types":["int","int"]},
               "1":{"output_type":"int","input_types":None},
               "2":{"output_type":"int","input_types":None}}

def make_melkor_parser(corpus,cfg_diction,word_dim=256,signal_dim=32,key_dim=42,latent_dim=132):
    cfg = ContextFreeGrammar(cfg_diction,signal_dim)
    decoder = Decoder(signal_dim,key_dim,latent_dim,cfg)
    model = LanguageParser(word_dim,signal_dim,decoder,corpus)
    return model

model = make_melkor_parser(corpus,cfg_diction,signal_dim=132,key_dim=44)

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
print(p,l)
p,l = model("I am melkor")
print(p,l)