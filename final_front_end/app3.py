import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64

import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib

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

user_dict = dicts(df, 'User_id')

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
        options=user_dict,
		placeholder='Select user'
    ),

    markdown_text

    #html.Div(id='output')

    ])

if __name__ == '__main__':
    app.run_server(debug=True)
