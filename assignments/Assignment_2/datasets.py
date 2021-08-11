import torch
from torch.utils.data import Dataset

import numpy
import pandas
from nltk import word_tokenize

from utils import get_max_length, create_glove_dict


class dataset(Dataset):
    def __init__(self, path_to_glove='glove.6B.200d.txt',
                 embedding_dim=200, prep_Data_from = 'train', purpose='train_model'):
        """
        NOTE-
        Beware of NaNs, drop them beforehand

        Dataset is the Liar Dataset. The description of the data can be found here -
        "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection - https://arxiv.org/abs/1705.00648"
        Download the dataset from - https://github.com/Tariq60/LIAR-PLUS
        Find the Glove vectors at https://nlp.stanford.edu/projects/glove/ and download the 822MB one.
        It contains 50d,100d, 200d and 300d vectors.
        300d with 400K vocab takes around 1.5GB RAM, choose file according to your system.
        We have prepared test cases using the 200d vectors. 

        :param path_to_glove: path to the desired glove vector file. File would be a .txt file
        :param embedding_dim: The dimension of vector you are choosing.
        :param prep_Data_from: Chose file from which you wanna prep data. 
        :param purpose: This is only used by the test.py file. This parameter should not concern you. When making your dataloaders, DO NOT pass this parameter. 
        """
        assert prep_Data_from in ['train', 'test', 'val']
        assert purpose in ['train_model', 'test_class']
        
        if purpose == 'train_model':
            path_to_train = 'train2.tsv'
            path_to_val = 'val2.tsv'
            path_to_test = 'test2.tsv'
        else:
            path_to_train = 'sample_train.tsv'
            path_to_test = 'sample_test.tsv'
            path_to_val = 'sample_val.tsv'

        train_Dataframe = pandas.read_csv(path_to_train, sep='\t', header=None).dropna()
        test_Dataframe = pandas.read_csv(path_to_test, sep='\t', header=None).dropna()
        val_Dataframe = pandas.read_csv(path_to_val, sep='\t', header=None).dropna()

        self.embeddings = create_glove_dict(path_to_glove)
        self.embedding_dim = embedding_dim
        self.dataframe = pandas.concat([train_Dataframe, test_Dataframe, val_Dataframe])

        self.justification_max = get_max_length(self.dataframe, 15)
        self.statement_max = get_max_length(self.dataframe, 3)

        if prep_Data_from == 'train':
            self.dataframe = train_Dataframe
        elif prep_Data_from == 'val':
            self.dataframe = val_Dataframe
        elif prep_Data_from == 'test':
            self.dataframe = test_Dataframe

        del train_Dataframe, test_Dataframe, val_Dataframe

        self.labels = {"true": 0,
                       "mostly-true": 1,
                       "half-true": 2,
                       "barely-true": 3,
                       "false": 4,
                       "pants-fire": 5}

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx): # 1.25 Mark
        data_dict = {}
        statement = self.dataframe.iloc[idx,3]
        statement = word_tokenize(statement.lower())
        justification = self.dataframe.iloc[idx,15]
        justification = word_tokenize(justification.lower())
        label = self.dataframe.iloc[idx,2]
        label = torch.tensor(self.labels[label])

        st_mat = numpy.zeros([self.embedding_dim,self.statement_max])
        js_mat = numpy.zeros([self.embedding_dim,self.justification_max])

        for pos, word in enumerate(statement):
            try:
                if word in self.embeddings.keys():
                    st_mat[:,pos] = self.embeddings[word]

            except KeyError:
                print('Word not in Vocab. Placing it at origin')
                continue

        for pos, word in enumerate(justification):
            try:
                if word in self.embeddings.keys():
                    js_mat[:,pos] = self.embeddings[word]
            
            except KeyError:
                print('Word not in Vocab. Placing it at origin')
                continue

        vectorized_statement = torch.tensor(st_mat).float()
        vectorized_justification = torch.tensor(js_mat).float()
        credit_history = torch.tensor(numpy.array([x for x in self.dataframe.iloc[idx, 9:14]]))   
        # 9 - Barely true counts
        # 10 - False counts
        # 11 half true counts
        # 12 mostly true counts
        # 13 pants on fire count
        
        data_dict = {"statement":vectorized_statement, "justification":vectorized_justification,"label":label,"credit_history":credit_history}
        del vectorized_statement, vectorized_justification, label, credit_history  # just free up some memory
        return data_dict

    def get_max_lenghts(self):
        return self.statement_max, self.justification_max

    def get_Data_shape(self):
        return self.dataframe.shape

