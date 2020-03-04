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

from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k, recall_at_k

import pickle

#image_filename = 'skimage.png'
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

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

tones_dict = dicts(df, 'Skin_Tone')
types_dict = dicts(df, 'Skin_Type')
eyes_dict = dicts(df, 'Eye_Color')
hair_dict = dicts(df, 'Hair_Color')

products_dictionary = dicts(df, 'Product')

user_dictionary = dicts(df, 'User_id')

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
        [html.Tr([html.Th(col) for col in ['V', 'V']])] + rows
        )

separation_string = '''




'''

intro_text = '''
__This simple app makes skincare recommendations! Whether you are new to the world of beauty and self care, or already have your favorite products__,
__one of the recommenders below can help. Our first recommender uses your personal features to estimate what products may work best for you__,
__and the second one requires you to select a product that you already love, and the system will give you the names of similar items. Both of them__
__also provide links to [Sephora.com](https://www.sephora.com/), so you can instatntly buy or check for more information.__

__Third recommender is for business use. You can type in a unique user id and the system will check for products that this user likes and recommend similar!__
'''


markdown_text_1 = '''
__Based on your features, these are the top products for you:__
'''

markdown_text_2 = '''
__Based on your preference, these are the top products for you:__
'''

markdown_text_3 = '''
__This user may like the following products:__
'''

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#https://codepen.io/chriddyp/pen/bWLwgP.css

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



colors = {
    #'background': '#1DB954',
    "text": "#111111",
    "background-image" : "url('/assets/wallpaperskin_retouched.png')",
    "background-size": "cover",
}

app.layout = html.Div(style=colors, children=[
    html.H1(children='Skincare Recommendations',
			style={
				'textAlign': 'center',
				'color': colors['text'],
				'backgroundColor': colors["background-image"],
                 'font-family': 'Bangers'
			}
	),


        dcc.Markdown(children=intro_text),

        dcc.Markdown(children=separation_string),


	html.Label('Skin Tone'),
    dcc.Dropdown(
		id='skintone-selector',
        options=tones_dict,
		placeholder='Select your skin tone'
    ),

	html.Label('Skin Type'),
    dcc.Dropdown(
		id='skintype-selector',
        options=types_dict,
        placeholder='Select your skin type'
    ),

	html.Label('Eye color'),
    dcc.Dropdown(
		id='eyecolor-selector',
		options=eyes_dict,
        placeholder='Select your eye color'
    ),

	html.Label('Hair color'),
    dcc.Dropdown(
		id='haircolor-selector',
        options=hair_dict,
        placeholder='Select your eye color'
    ),

    dcc.Markdown(children=separation_string),

	dcc.Markdown(children=markdown_text_1),

    html.Div(id='output_1'),

    dcc.Markdown(children=separation_string),

    html.H2(children='Skincare recommendations based on your favorites',
			style={
				'textAlign': 'center',
				'color': colors['text'],
				'backgroundColor': colors["background-image"],
                'font-family': 'Bangers'
			}
	),
    html.Label('Your favorite product!'),
    dcc.Dropdown(
		id='product-selector',
        options=products_dictionary,
		placeholder='Select your favorite product'
    ),

    dcc.Markdown(children=markdown_text_2),

    html.Div(id='output_2'),

    dcc.Markdown(children=separation_string),

    html.H2(children='Skincare recommendations to users (for business)',
			style={
				'textAlign': 'center',
				'color': colors['text'],
				'backgroundColor': colors["background-image"],
                'font-family': 'Bangers'
			}
	),
    html.Label('List of user ids'),
    dcc.Dropdown(
		id='user-selector',
        options=user_dictionary,
		placeholder='Select user'
    ),

    dcc.Markdown(children=markdown_text_3),

    html.Div(id='output_3')

    ])

@app.callback(
	Output('output_1', 'children'),
    [Input('skintone-selector', 'value'),
    Input('skintype-selector', 'value'),
    Input('eyecolor-selector', 'value'),
    Input('haircolor-selector', 'value')])

def recommend_products_by_user_features(skintone, skintype, eyecolor, haircolor):
    ddf = df[(df['Skin_Tone'] == skintone) & (df['Hair_Color'] == haircolor) & (df['Skin_Type'] == skintype) & (df['Eye_Color'] == eyecolor)]
    recommendations = ddf[(ddf['Rating_Stars'].notnull())]
    data = recommendations[['Rating_Stars', 'Product_Url', 'Product']]

    data = data.sort_values('Rating_Stars', ascending=False).head()

    return Table(data)

@app.callback(
	Output('output_2', 'children'),
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

@app.callback(
	Output('output_3', 'children'),
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
