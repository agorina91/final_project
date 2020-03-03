import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64

import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib
import matplotlib.pyplot as plt

import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD

df = pd.read_csv('skindataall.csv', index_col=[0])

def dicts(df, colname):
    vals = list(set(df[colname]))
    l = []
    for i in vals:
        dic = {}
        dic['label'] = i
        dic['value'] = i
        l.append(dic)
    return l

products_dict = dicts(df, 'Product')

def Table(df):
    rows = []
    for i in range(len(df)):
        row = []
        for col in df.columns:
            value = df.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'Product':
                cell = html.Td(html.A(href=df.iloc[i]['Product_Url'], children=value))
            elif col == 'Product_Url':
                continue
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in ['Product', 'Ingredients', 'Rating']])] + rows
        )

markdown_text = '''
Based on your preference, these are the top products for you:
'''


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#FAEBD7',
    'text': '#111111'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H4(children='Skincare recommendations based on your favorites',
			style={
				'textAlign': 'center',
				'color': colors['text'],
				'backgroundColor': colors['background']
			}
	),
    html.Label('Your favorite product!'),
    dcc.Dropdown(
		id='product-selector',
        options=products_dict,
		placeholder='Select your favorite product'
    ),

    markdown_text,

    html.Div(id='output')

    ])

@app.callback(
	Output('output', 'children'),
    [Input('product-selector', 'value')]
    )

def content_recommender(product):
    try:
        df_cont = df[['Product', 'Product_id', 'Ingredients', 'Product_Url', 'Ing_Tfidf', 'Rating']]
        df_cont.drop_duplicates(inplace=True)
        df_cont = df_cont.reset_index(drop=True)
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(df_cont['Ingredients'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        titles = df_cont[['Product', 'Ing_Tfidf', 'Rating', 'Product_Url']]
        indices = pd.Series(df_cont.index, index=df_cont['Product'])
        idx = indices[product]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        product_indices = [i[0] for i in sim_scores]

    except KeyError:
        return None

    return Table(titles.iloc[product_indices])

if __name__ == '__main__':
    app.run_server(debug=True)
