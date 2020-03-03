import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64

import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib

from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k,recall_at_k

import pickle

df = pd.read_csv('skindataall.csv', index_col=[0])

with open('mf_model.pkl', 'rb') as f:
    mf_model = pickle.load(f)

def dicts(df, colname):
    vals = list(set(df[colname]))
    l = []
    for i in vals:
        dic = {}
        dic['label'] = i
        dic['value'] = i
        l.append(dic)
    return l

def create_interaction_matrix(df, user_col, item_col, rating_col, norm= False, threshold = None):
    interactions = df.groupby([user_col, item_col])[rating_col].sum().unstack().reset_index().fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions

interaction_matrix = create_interaction_matrix(df=df, user_col='User_id', item_col = 'Product_id', rating_col='Rating_Stars')

def create_user_dict(interactions):
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict

user_dict = create_user_dict(interaction_matrix)

def create_item_dict(df, id_col, name_col):
    item_dict ={}
    for i in df.index:
        item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]
    return item_dict

product_dict = create_item_dict(df = df, id_col = 'Product_id', name_col = 'Product')

user_dictionary = dicts(df, 'User_id')

markdown_text = '''
This user may like the following products:
'''

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#FAEBD7',
    'text': '#111111'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H4(children='Skincare recommendations to users (for business)',
			style={
				'textAlign': 'center',
				'color': colors['text'],
				'backgroundColor': colors['background']
			}
	),
    html.Label('List of user ids'),
    dcc.Dropdown(
		id='user-selector',
        options=user_dictionary,
		placeholder='Select user'
    ),

    markdown_text,

    html.Div(id='output')

    ])

@app.callback(
	Output('output', 'children'),
    [Input('user-selector', 'value')]
    )

def sample_recommendation_user(user_id, model=mf_model, interactions=interaction_matrix, user_dict=user_dict,
                               item_dict=product_dict, threshold = 4, nrec_items = 10, show = True):

    try:
        n_users, n_items = interactions.shape
        user_x = user_dict[user_id]
        scores = pd.Series(model.predict(user_x,np.arange(n_items)))
        scores.index = interactions.columns
        scores = list(pd.Series(scores.sort_values(ascending=False).index))

        known_items = list(pd.Series(interactions.loc[user_id,:] \
                                     [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
        scores = [x for x in scores if x not in known_items]
        return_score_list = scores[0:nrec_items]
        known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
        scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))

        h_list = []
        for i in scores:
            h_list.append(html.H6(i))

    except KeyError:
        return None

    return h_list

if __name__ == '__main__':
    app.run_server(debug=True)
