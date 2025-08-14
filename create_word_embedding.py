import json
import os
import numpy as np
from transformers import BertModel, BertTokenizer, AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import AlbertTokenizer, AlbertModel
from transformers import GPT2Tokenizer, GPT2Model
import nltk
from brainbert_pretrain_model import BrainBertModel

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import lmdb
import msgpack
from lz4.frame import decompress
import torch
from utils import normalization
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from models.sentence2vec import Sentence2Vec


# 创建train test 训练集
# type-test/run
# key:train/test
# embedding_model_name: bert-base-uncased etc.
def createDataSet(_type, key, embedding_model_name):
    datasetname = 'COCO2014'

    # bert-base-uncased
    # bert-large-uncased
    # bert-base-multilingual-cased
    # bert-large-uncased-whole-word-masking
    H_DIM = 1024
    # 加载embedding model
    if embedding_model_name in \
            ['bert-base-uncased', 'bert-base-multilingual-cased', 'bert-large-uncased',
             'bert-large-uncased-whole-word-masking']:
        tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
        model = BertModel.from_pretrained(embedding_model_name)
        if embedding_model_name in ['bert-base-uncased', 'bert-base-multilingual-cased']:
            H_DIM = 768

    elif embedding_model_name == 'roberta-large' or embedding_model_name == 'roberta-base':
        # roberta
        if embedding_model_name == 'roberta-base':
            H_DIM = 768
        config = AutoConfig.from_pretrained(embedding_model_name)
        config.output_hidden_states = True
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        model = AutoModelForMaskedLM.from_pretrained(embedding_model_name, config=config)

    elif embedding_model_name in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
                                  'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2',
                                  'albert-xxlarge-v2']:
        if embedding_model_name in ['albert-base-v1', 'albert-base-v2']:
            H_DIM = 768
        if embedding_model_name in ['albert-xlarge-v1', 'albert-xlarge-v2']:
            H_DIM = 2048
        if embedding_model_name in ['albert-xxlarge-v2', 'albert-xxlarge-v1']:
            H_DIM = 4096
        tokenizer = AlbertTokenizer.from_pretrained(embedding_model_name)
        model = AlbertModel.from_pretrained(embedding_model_name)
    elif embedding_model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        if embedding_model_name in ['gpt2']:
            H_DIM = 768
        if embedding_model_name in ['gpt2-medium']:
            H_DIM = 1024
        if embedding_model_name in ['gpt2-large']:
            H_DIM = 1280
        if embedding_model_name in ['gpt2-xl']:
            H_DIM = 1600
        tokenizer = GPT2Tokenizer.from_pretrained(embedding_model_name)
        model = GPT2Model.from_pretrained(embedding_model_name)
        # trainデータとtestデータの作成
    elif embedding_model_name == 'brainbert':
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        #         # brainBERT
        brainBERT_checkpoint = '/Storage/ying/resources/BrainBertTorch/output/ckpt/model_0_27.4_109500.pt'
        checkpoint = {k.replace('module.', ''): v for k, v in torch.load(brainBERT_checkpoint).items()}
        model_config = '/Storage/ying/project/BrainBertTorch/config/brainbert-large.json'
        model = BrainBertModel.from_pretrained(model_config, checkpoint)
    elif embedding_model_name.lower() == "glove":
        H_DIM = 300
        word2vec_output_file = 'models/glove.42B.300d' + '.word2vec'
        glove2word2vec("models/glove.42B.300d.txt", word2vec_output_file)
        # load the Stanford GloVe model
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    # k = get_glove_embedding(text, model)

    elif embedding_model_name == 'word2vec':
        H_DIM = 100
        model = Sentence2Vec('./models/word2vec.model')
        # k = get_w2v_embedding(text, model)
    # else:
    #     return None,None
    # """
    # 中間表現の次元

    # 脳活動データの次元設定
    brain_dim = 3104
    # データの作成
    X = np.empty((0, H_DIM), float)
    Y = np.empty((0, brain_dim), float)
    if key == 'train':
        brain_dbs = ["/home/sakura/resources/BrainBertTorch/brain/" + datasetname + "/pretrain_train.db",
                     "/home/sakura/resources/BrainBertTorch/brain/" + datasetname + "/pretrain_test.db"
                     ]
        text_dbs = ["/home/sakura/resources/BrainBertTorch/txt/" + datasetname + "/pretrain_train.db",
                    "/home/sakura/resources/BrainBertTorch/txt/" + datasetname + "/pretrain_test.db"
                    ]
    else:
        text_dbs = ["/home/sakura/resources/BrainBertTorch/txt/" + datasetname + "/pretrain_val.db"]
        brain_dbs = ["/home/sakura/resources/BrainBertTorch/brain/" + datasetname + "/pretrain_val.db"]

    cache_path = 'text_embedding/' + datasetname + '/' + embedding_model_name + f'_text_embedding_' + str(key) + '.json'
    if os.path.exists(cache_path):
        sen_embed_dict = json.load(open(cache_path, 'r'))
        # print(cache_path+":",len(sen_embed_dict))
    else:
        sen_embed_dict = {}
    # Training dataの作成
    # brain = {}
    for text_db, brain_db in zip(text_dbs, brain_dbs):

        # text = sen_embed_dict
        brain_env = lmdb.open(brain_db, readonly=True, create=False, lock=False)
        brain_txn = brain_env.begin(buffers=True)
        text_env = lmdb.open(text_db, readonly=True, create=False, lock=False)
        text_txn = text_env.begin(buffers=True)
        text_cursor = text_txn.cursor()

        for index, (k, v) in enumerate(tqdm(text_cursor, desc="iter")):  # 遍历sentence
            # print(index)
            value_ = msgpack.loads(decompress(text_txn.get(k)), raw=False)  # 获取lmdb数据
            sentence = value_['sentence']  # 定位当前句子
            brain_value = msgpack.loads(decompress(brain_txn.get(value_['img_fname'].encode('utf-8'))), raw=False)
            if datasetname == 'COCO2014':
                Y = np.append(Y, [brain_value['norm_bb']], axis=0)
            else:
                Y = np.append(Y, [brain_value['norm_bb']['data']], axis=0)
            if sentence in sen_embed_dict.keys():  # 如果数据embedding已经保存在文件里
                X = np.append(X, [sen_embed_dict.get(sentence)], axis=0)
            else:
                if embedding_model_name in \
                        ['bert-base-uncased', 'bert-large-uncased', 'bert-base-multilingual-cased',
                         'bert-large-uncased-whole-word-masking']:
                    # # 获取BERT的embedding
                    embedding_list = get_bert_embedding_tensor(sentence, tokenizer, model).tolist()
                elif embedding_model_name == 'roberta-large' or embedding_model_name == 'roberta-base':
                    embedding_list = get_roberta_embedding_tensor(sentence, tokenizer, model).tolist()
                elif embedding_model_name.lower() == "glove":
                    embedding_list = get_glove_embedding(sentence, model).tolist()
                    # word2vec_output_file = 'glove.42B.300d' + '.word2vec'
                    # glove2word2vec("glove.42B.300d.txt", word2vec_output_file)
                    # # load the Stanford GloVe model
                    # model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
                    # # k = get_glove_embedding(text, model)
                elif embedding_model_name == 'word2vec':
                    embedding_list = get_w2v_embedding(sentence, model).tolist()

                    # model = Sentence2Vec('./models/word2vec.model')
                    # k = get_w2v_embedding(text, model)
                elif embedding_model_name in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
                                              'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2',
                                              'albert-xlarge-v2',
                                              'albert-xxlarge-v2']:
                    embedding_list = get_albert_embedding_tensor(sentence, tokenizer, model).tolist()
                    # k = get_albert_embedding_tensor(text, tokenizer, model)
                elif embedding_model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                    # tokenizer = GPT2Tokenizer.from_pretrained(_type)
                    # model = GPT2Model.from_pretrained(_type)
                    embedding_list = get_gpt_embedding_tensor(sentence, tokenizer, model).tolist()
                elif embedding_model_name == 'brainbert':
                    # 获取brainBERT的embedding
                    brain_value_ = msgpack.loads(decompress(brain_txn.get(value_['img_fname'].encode('utf-8'))),
                                                 raw=False)
                    if datasetname == 'COCO2014':
                        img_feat = torch.tensor(brain_value_['features']).float()
                    else:
                        img_feat = torch.tensor(brain_value_['features']['data']).float()

                    sequence_output, pooled_output, hidden_states, attentions = get_brain_bert_attention_output(
                        sentence=sentence, img_feat=img_feat, tokenizer=tokenizer, model=model)
                    embedding_list = pooled_output.detach().squeeze(0).tolist()
                X = np.append(X, [embedding_list], axis=0)

                sen_embed_dict[sentence] = embedding_list

            if _type == 'test_' and X.shape[0] > 1:
                break
        else:
            brain_env.close()
            text_env.close()
            # break
            continue
    if not os.path.exists(cache_path):
        with open(cache_path, 'w') as f:
            json.dump(sen_embed_dict, f)
    X, _, _ = normalization(X)
    Y, _, _ = normalization(Y)

    return X, Y


#
# def get_bert_masked_lm_output(ids):
#     """
#     這段程式碼載入已經訓練好的 masked 語言模型並對有 [MASK] 的句子做預測
#     """
#     from transformers import BertForMaskedLM
#
#     # 除了 tokens 以外我們還需要辨別句子的 segment ids
#     tokens_tensor = torch.tensor([ids])  # (1, seq_len)
#     segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
#     maskedLM_model = BertForMaskedLM.from_pretrained(bert_type)
#     # clear_output()
#
#     # 使用 masked LM 估計 [MASK] 位置所代表的實際 token
#     maskedLM_model.eval()
#     with torch.no_grad():
#         outputs = maskedLM_model(tokens_tensor, segments_tensors)
#         predictions = outputs[0].detach().squeeze(0)
#         # (1, seq_len, num_hidden_units) 得到了每个词的30502个概率分布
#     return predictions

# GloVe
def get_glove_embedding(sentence, model):
    # We just need to run this code once, the function glove2word2vec saves the Glove embeddings in the word2vec format
    # that will be loaded in the next section
    tokenized_sent = word_tokenize(sentence.lower())

    embs = []
    for t in tokenized_sent:
        try:
            if t in ['waistcoat-']:
                t = 'waistcoat'
            if t == 'waistcoat-pocket':
                t = 'pocket'
            if t == 'cherry-tart':
                t = 'cherry'
            if t == 'sadnwiches':
                t = 'sandwiches'
            emb = model.get_vector(t)
            emb = emb.tolist()
        except Exception as e:
            emb = [0] * 300
            pass
        embs.append(emb)
        continue

    # a = np.array(embs).mean(axis=1)
    # b =  np.array(embs).mean(axis=0)
    # # Show a word embedding
    return np.array(embs).mean(axis=0)


# word2vec
def get_w2v_embedding(sentence, model):
    tokenized_sent = word_tokenize(sentence.lower())

    embs = []
    for t in tokenized_sent:
        emb = model.get_vector(t)
        # print(t)
        # print(emb) 存在问题：大量单词embedding=0
        embs.append(emb.tolist())
    # a = np.array(embs).mean(axis=1)
    # b =  np.array(embs).mean(axis=0)
    # Show a word embedding
    return np.array(embs).mean(axis=0)


def get_brain_bert_attention_output(img_feat, sentence, tokenizer, model):
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids'][0]
    input_id_list = input_ids.tolist()  # Batch index 0
    input_id_list.extend([104])  # Batch index 0
    input_ids = torch.Tensor(input_id_list).long()
    output = model(brain_feature=img_feat, input_ids=input_ids)

    return output


def get_roberta_embedding_tensor(sentence, tokenizer, model):
    sentence = tokenizer.encode(sentence, padding=False, max_length=512, truncation=True, return_tensors='pt')

    output = model(sentence)

    return output[-1][-1][:, 0, :].detach().squeeze(0)


def get_bert_embedding_tensor(sentence, tokenizer, model):
    text_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, return_attention_mask=True)
    input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
    token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
    attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)

    res = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # bert-base-multilingual-cased;bert-large-uncased-whole-word-masking
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    # get cls's output
    k = res[-1].detach().squeeze(0)
    #     print(k.shape)
    return k


def get_albert_embedding_tensor(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    return output[-1].detach().squeeze(0)


def get_gpt_embedding_tensor(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    return output[0][:, 0, :].detach().squeeze(0)


if __name__ == "__main__":
    text = "Replace me by any text you'd like."
    _type = "glove"
    k = []
    # k1 = get_bert_embedding_tensor('指的都是一個可以用來代表某詞彙。')
    # k2 = get_roberta_embedding_tensor('指的都是一個可以用來代表某詞彙。')
    # glove_input_file = glove_filename
    if _type == "glove":
        word2vec_output_file = 'models/glove.42B.300d' + '.word2vec'
        glove2word2vec("models/glove.42B.300d.txt", word2vec_output_file)
        # load the Stanford GloVe model
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        # k = get_glove_embedding(text,model)

    elif _type == 'word2vec':

        model = Sentence2Vec('./models/word2vec.model')
        k = get_w2v_embedding(text, model)

    elif _type in ['albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1',
                   'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2']:
        tokenizer = AlbertTokenizer.from_pretrained(_type)
        model = AlbertModel.from_pretrained(_type)
        k = get_albert_embedding_tensor(text, tokenizer, model)
    elif _type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        tokenizer = GPT2Tokenizer.from_pretrained(_type)
        model = GPT2Model.from_pretrained(_type)
        k = get_gpt_embedding_tensor(text, tokenizer, model)

    # import spacy
    #
    # nlp = spacy.load("en_trf_bertbaseuncased_lg")
    # apple1 = nlp("Apple shares rose on the news.")
    # apple2 = nlp("Apple sold fewer iPhones this quarter.")
    # apple3 = nlp("Apple pie is delicious.")
    #
    # # sentence similarity
    # print(apple1.similarity(apple2))  # 0.69861203
    # print(apple1.similarity(apple3))  # 0.5404963

    # print(k)
    # print(k2)
    # c = correlation_c(np.array(k1.tolist()),np.array(k2.tolist()))
    # print(c)
