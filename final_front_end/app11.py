import pandas as pd
import pickle
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
from navbar import Navbar
import base64
import numpy as np
import re

image_filename = 'pink_flowers.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

df = pd.read_csv('skindataall.csv', index_col=[0])

nav = Navbar()

colors = {
    'background': '#FAEBD7',
    'text': '#111111'
}

header = html.H4(children='Skincare Recommendations',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'backgroundColor': colors['background']
        }
)

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

dropdown1 = html.Div(dcc.Dropdown(
    id='skintone-selector',
    options=tones_dict,
    placeholder='Select your skin tone'
))

dropdown2 = html.Div(dcc.Dropdown(
    id='skintype-selector',
    options=types_dict,
    placeholder='Select your skin type'
))

dropdown3 = html.Div(dcc.Dropdown(
    id='eyecolor-selector',
    options=eyes_dict,
    placeholder='Select your eye color'
))

dropdown4 = html.Div(dcc.Dropdown(
    id='haircolor-selector',
    options=hair_dict,
    placeholder='Select your eye color'
))

output = html.Div(id='output')

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


def App11():
    layout = html.Div([
        nav,
        header,
        dropdown1,
        dropdown2,
        dropdown3,
        dropdown4,
        output
    ])
    return layout

if __name__ == '__main__':
    app.run_server(debug=True)
