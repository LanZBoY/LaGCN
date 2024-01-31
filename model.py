"""
    定義classmodel或是神經網路架構的文件

    Author @JimLiu
"""

import torch
from nltk import word_tokenize

class Vocaburary:
    """
        定義自然語言處理領域的基本資料結構
    """
    def __init__(self, word_count : dict[str, int] = None, min_time = 0) -> None:
        self.wordset = set()
        for word in word_count.keys():
            if word_count[word] >= min_time:
                self.wordset.add(word)
        self.idx2word = [word for word in self.wordset]
        self.word2idx = {word:i for i, word in enumerate(self.idx2word)}
        
    def encode(self, tokens):
        ids = []
        for word in tokens:
            if word in self.wordset:
                ids.append(self.word2idx[word])
        return ids

    def __len__(self):
        return len(self.wordset)
    
    def __str__(self) -> str:
        return self.wordset.__str__()

class TextGCN(torch.nn.Module):

    def __init__(self, num_node, hidden_dim, n_class) -> None:
        super(TextGCN, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(size=(num_node, )).diagflat().to_sparse(), requires_grad = False)
        self.conv_1 = GraphConv(num_node, hidden_dim)
        self.conv_2 = GraphConv(hidden_dim, n_class)

    def forward(self, A):
        h = torch.nn.functional.relu(self.conv_1(A, self.weight))
        return self.conv_2(A, h)

class MuliLayerTextGCN(torch.nn.Module):

    def __init__(self, num_node, hidden_dim, n_class, N_layer = 2) -> None:
        super(MuliLayerTextGCN, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(size=(num_node, )).diagflat().to_sparse(), requires_grad = False)
        self.graphConvs = []
        for i in range(N_layer):
            if i == 0:
                self.graphConvs.append(GraphConv(num_node, hidden_dim))
            elif i == N_layer - 1:
                self.graphConvs.append(GraphConv(hidden_dim, n_class))
            else:
                self.graphConvs.append(GraphConv(hidden_dim, hidden_dim))
        self.graphConvs = torch.nn.ModuleList(self.graphConvs)

    def forward(self, A):

        h = torch.nn.functional.relu(self.graphConvs[0](A, self.weight))
        for gcn_layer in self.graphConvs[1:-1]:
            h = torch.nn.functional.relu(gcn_layer(A, h))
            
        return self.graphConvs[-1](A, h)


class Emb_TextGCN(torch.nn.Module):

    def __init__(self, doc_num : int, word_num : int, label_num : int, embedding_dim : int, hidden_dim : int) -> None:
        super(Emb_TextGCN, self).__init__()
        self.doc_num = doc_num
        self.word_num = word_num
        self.label_num = label_num

        self.doc_emb = torch.nn.Embedding(num_embeddings = self.doc_num, embedding_dim = embedding_dim)
        self.word_emb = torch.nn.Embedding(num_embeddings = self.word_num, embedding_dim = embedding_dim)
        self.label_emb = torch.nn.Embedding(num_embeddings = self.label_num, embedding_dim = embedding_dim)

        self.graphConv_1 = GraphConv(embedding_dim, hidden_dim)
        self.graphConv_2 = GraphConv(hidden_dim, self.label_num)

    def forward(self, G):
        X = torch.concat([self.doc_emb.weight, self.word_emb.weight, self.label_emb.weight])
        H = torch.nn.functional.relu(self.graphConv_1(G, X))
        return self.graphConv_2(G, H)

class GraphConv(torch.nn.Module):

    def __init__(self, in_feat, out_feat) -> None:
        super(GraphConv, self).__init__()
        self.W = torch.nn.Linear(in_feat, out_feat)

    def forward(self, A, X):
        return A @ self.W(X)
