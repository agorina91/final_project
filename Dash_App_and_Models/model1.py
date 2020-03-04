import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k,recall_at_k
from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, RandomizedSearchCV

df = pd.read_csv('skindataall.csv', index_col=[0])

data = df[['User_id', 'Product_Url', 'Rating_Stars']]
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_df(data, reader=reader)

trainset, testset = train_test_split(data, test_size=.2)

svd = SVD()
svd.fit(trainset)

predictions = svd.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

dump.dump('svd_model', algo=svd, predictions=predictions, verbose=1)
