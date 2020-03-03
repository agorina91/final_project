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

image_filename = 'pink_flowers.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

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

tones_dict = dicts(df, 'Skin_Tone')
types_dict = dicts(df, 'Skin_Type')
eyes_dict = dicts(df, 'Eye_Color')
hair_dict = dicts(df, 'Hair_Color')

#products_dict = dicts(df, 'Product')

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
        [html.Tr([html.Th(col) for col in ['Rating', 'Product']])] + rows
        )
markdown_text = '''
Based on your features, these are the top products for you:
'''

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#FAEBD7',
    'text': '#111111'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H4(children='Skincare Recommendations',
			style={
				'textAlign': 'center',
				'color': colors['text'],
				'backgroundColor': colors['background']
			}
	),


	html.Img(
		src='data:image/png;base64,{}'.format(encoded_image.decode()),
		style={
			'height': '50%',
			'width': '50%'
			#'textAlign': 'center'
		}),


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

    #html.Label('Your favorite product!'),
    #dcc.Dropdown(
	#	id='product-selector',
    #    options=products_dict,
	#	placeholder='Select your favorite product'
    #),

	markdown_text,

    html.Div(id='output')

])

@app.callback(
	Output('output', 'children'),
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
    #return html.Table(
    #    [html.Tr([html.Th(col) for col in data.columns])] +
    #    [html.Tr([
    #        html.Td(data.iloc[i][col]) for col in data.columns
    #    ]) for i in range(min(len(data), 10))]
    #)


if __name__ == '__main__':
    app.run_server(debug=True)
