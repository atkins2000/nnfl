# 0.75 Marks. 
# To test your trainer and  arePantsonFire class, Just create random tensor and see if everything is working or not.  
from torch.utils.data import DataLoader
from datasets import *
from Encoder import *
from Attention import *
from LiarLiar import *
from utils import *
from trainer import *

# Your code goes here.
liar_dataset_train = dataset(prep_Data_from = 'train2.tsv')
liar_dataset_val = dataset(prep_Data_from = 'val2.tsv')
max_length_sentence,max_length_justification = liar_dataset_train.get_max_lenghts()

dataloader_train = DataLoader(liar_dataset_train,batch_size=50)
dataloader_val = DataLoader(liar_dataset_val,batch_size=25)

statement_encoder = Encoder(conv_layers = 5, hidden_dim = 512)
justification_encoder = Encoder(conv_layers = 5, hidden_dim = 512)

multiheadAttention = MultiHeadAttention(hid_dim = 512,n_heads=32)
positionFeedForward = PositionFeedforward(hid_dim = 512, feedForward_dim=2048)

model = arePantsonFire(statement_encoder,justification_encoder,multiheadAttention,positionFeedForward,512,max_length_sentence,max_length_justification,input_dim = liar_dataset_train.embedding_dim)

trainer(model, dataloader_train, dataloader_val, num_epochs = 1 , checkpoint=100, train_batch=1, test_batch=1, device='cuda:0')


# Do not change module_list , otherwise no marks will be awarded
module_list = [liar_dataset_train, liar_dataset_val, dataloader_train, dataloader_val, statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, model]
del  liar_dataset_val, liar_dataset_train, dataloader_train, dataloader_val


liar_dataset_test = dataset(prep_Data_from='test2.tsv')
test_dataloader = DataLoader(dataset=liar_dataset_test, batch_size=1)
infer(model=model, dataloader=test_dataloader)

