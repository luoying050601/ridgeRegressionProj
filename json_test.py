import json
# train = json.load(open("/Storage/ying/resources/BrainBertTorch/brain/alice_ae/pretrain_train.db/nbb.json", 'r'))
# test = json.load(open("/Storage/ying/resources/BrainBertTorch/brain/alice_ae/pretrain_test.db/nbb.json", 'r'))
#
# train = json.load(open('/home/sakura/resources/BrainBertTorch/txt/alice_ae/pretrain_train.db/txt2brain.json', 'r'))
# test = json.load(open("/Storage/ying/resources/BrainBertTorch/txt/alice_ae/pretrain_test.db/brain2txt.json", 'r'))
# brainbert_text_embedding_train = json.load(open("/Storage/ying/project/ridgeRegression/brainbert_text_embedding_train_user.json", 'r'))
# print(train,test)
import lmdb
import msgpack
from tqdm import tqdm
from lz4.frame import decompress, compress

brain_dbs = ["/home/sakura/resources/BrainBertTorch/brain/alice_ae/pretrain_train.db",
             # "/home/sakura/resources/BrainBertTorch/brain/alice_ae/pretrain_test.db"
             ]

text_dbs = ["/home/sakura/resources/BrainBertTorch/txt/alice_ae/pretrain_train.db",
            # "/home/sakura/resources/BrainBertTorch/txt/alice_ae/pretrain_test.db"
            ]

for text_db, brain_db in zip(text_dbs, brain_dbs):
 brain_env = lmdb.open(brain_db, readonly=True, create=False, lock=False)
 brain_txn = brain_env.begin(buffers=True)
 text_env = lmdb.open(text_db, readonly=True, create=False, lock=False)
 text_txn = text_env.begin(buffers=True)
 _cursor = text_txn.cursor()
 # bert_text_embedding_train
 # brainbert_text_embedding_test
 # # cache_path = bert_model + f'_text_embedding_' + str(key) + '.json'
 # if os.path.exists(cache_path):
 #  sen_embed_dict = json.load(open(cache_path, 'r'))
 # else:
 #  sen_embed_dict = {}
 count = 0
 for index, (k, v) in enumerate(tqdm(_cursor, desc="iter")):
  count = count+1
  print(
   count
  )# 遍历
  value_ = msgpack.loads(decompress(text_txn.get(k)), raw=False)

  if value_['img_fname'] == 'alice_ae_18_278.npz' or value_['img_fname'] == 'alice_ae_18_280.npz':
   sentence = value_['sentence']
   print(value_['img_fname'],sentence)


#  for index, (k, v) in enumerate(tqdm(_cursor, desc="iter")):  # 遍历
#   value_ = msgpack.loads(decompress(brain_txn.get(k)), raw=False)
#   # sentence = value_['sentence']
#   count = 1 + count
# #   print(count,value_['img_fname'])
