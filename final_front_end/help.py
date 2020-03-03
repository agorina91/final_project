import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib
import matplotlib.pyplot as plt

#skindataall.csv
df = pd.read_csv('/Users/agorina/Desktop/final_front_end/skindataall.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()

#baseline recommender
def recommend_products_by_user_features(skintone, skintype, eyecolor, haircolor, percentile=0.85):
    ddf = df[(df['Skin_Tone'] == skintone) & (df['Hair_Color'] == haircolor) & (df['Skin_Type'] == skintype) & (df['Eye_Color'] == eyecolor)]

    recommendations = ddf[(ddf['Rating_Stars'].notnull())][['Rating_Stars', 'Product_Url', 'Product']]
    recommendations = recommendations.sort_values('Rating_Stars', ascending=False).head(250)

    print('Based on your features, these are the top products for you:')
    return recommendations

recommend_products_by_user_features('Light', 'Combination', 'Green', 'Brunette').head()

joblib.dump(recommend_products_by_user_features, 'recommend_products_by_user_features.joblib')
