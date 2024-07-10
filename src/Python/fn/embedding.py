#!/usr/bin/env python
# coding: utf-8

# In[ ]:


######################################
#Package checker
#
#Auto install and import py package
#
#Args:
#   package name in str e.g. "torch"
#   
#Returns:
#   NULL
######################################
def import_or_install(package):
    import pip
    from pip._internal import main
    try:
        __import__(package)
    except ImportError:
        main(['install', '--user', package])  
        
        
        
        
######################################
#Embedding models
#
#Define an model for embedding computation, auto select gpu if available
#
#Args:
#   model name in str from https://huggingface.co/bert-base-uncased, e.g."bert-base-uncased"
#   
#Returns:
#   A list of paramater, cpu/gpu, tokenizer, and the off the shelf model
######################################




def model(name):
    import_or_install('torch')
    import torch
    import_or_install('transformers')
    from transformers import BertTokenizer, BertModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(name) 
    model = BertModel.from_pretrained(name)
    return device, tokenizer, model




######################################
#Embedding computation
#
#Transform words/sentence into a vector based on defined model
#
#Args:
#   A list of paramater, cpu/gpu, tokenizer, and the off the shelf model
#   
#Returns:
#   A vector
######################################
def get_cls_embedding(model,document):
    encoded_input = model[1](document,return_tensors='pt',
                              padding='longest',add_special_tokens=True,
                              truncation=True).to(model[0])
    output = model[2].to(model[0])(**encoded_input)
    last_hidden_states = output.last_hidden_state
    MLP_input = last_hidden_states[0][0]
    value_np = MLP_input.cpu().data.numpy()
    return value_np

