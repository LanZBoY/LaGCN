"""
    定義工具的地方

    Author @JimLiu
"""
import dgl.sparse as dglsp
from nltk.corpus import stopwords
import re
import torch
from nltk import word_tokenize

def padding_Adj(A:torch.Tensor, max_size:list):
    return torch.nn.functional.pad(A, (0, max_size[0] - A.shape[0], 0, max_size[1] - A.shape[1]))

def nomalize_Adj(A : torch.Tensor) -> torch.Tensor:
    D_hat = A.to_dense().sum(dim=1)
    D_hat_invsqrt = (D_hat ** -0.5).diagflat().to_sparse()
    A_hat = D_hat_invsqrt @ A @ D_hat_invsqrt
    return A_hat

def generate_batch_Adjacency(A, batch_size) -> torch.Tensor:
    return torch.stack([A for _ in range(batch_size)])

def getDotMatrix(E: torch.Tensor):
    return E @ E.T

def getConsineSimMatrix(E : torch.Tensor):
    dotMat = getDotMatrix(E)
    sumSqrtMat = ((E**2).sum(dim=1)).sqrt()
    return dotMat/(sumSqrtMat.unsqueeze(dim=1) @ sumSqrtMat.unsqueeze(dim=0))

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def remove_stopwords(sentence) -> str:
    removewords = set(stopwords.words('english'))
    if type(sentence) is str:
            sentence = sentence.split()
    new_string = list()
    for word in sentence:
        if word in removewords:
            continue
        new_string.append(word)

    return " ".join(new_string)

def replace_num(self, string):
    num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
    result = re.sub(num, '<num>', string)
    return result

def replace_urls(self, string):
    url = re.compile(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
    result = re.sub(url, '<url>', string)
    result = ' '.join(re.split(' +|\n+', result)).strip()
    return result

def lean_str_sst(self, string):
    """
        Tokenization/string cleaning for the SST yelp_dataset
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
    other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
    string = re.sub(other_char, " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()