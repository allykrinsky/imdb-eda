from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dash import dash_table
import dash_bootstrap_components as dbc

df = pd.read_csv('movies_sample.csv')
for_display = df[['preview', 'rating', 'label']]
for_display.columns = ['Review', 'Rating', 'Classification']

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

def tfidf():
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['review'])
    return tfidf_matrix


def pca(tfidf):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf.toarray())
    return pca_result


def tsne(tfidf):
    tsne = TSNE(n_components=2, perplexity=2, random_state=0)
    tsne_result = tsne.fit_transform(tfidf.toarray())

    return tsne_result

tfidf = tfidf()
pca = pca(tfidf)
tsne = tsne(tfidf)

pca_fig = px.scatter(x=pca[:, 0], y=pca[:, 1], color=df['label'])
tsne_fig = px.scatter(x=tsne[:, 0], y=tsne[:, 1], color=df['label'])


label_fig = px.bar(df, x='label')

rating_count = df.groupby(['rating'])['label'].count().reset_index()
rating_fig = px.bar(rating_count, x='rating', y='label')


app.layout = html.Div([
    html.H1(children='IMDB Movie Reviews', style={
        'textAlign': 'center',
        'marginBottom': '20px',
        'fontFamily': 'Arial, sans-serif',
        'color': '#2c3e50'
    }),
    
    html.H2(children='Movie Reviews Dataset', style={
        'textAlign': 'left',
        'fontFamily': 'Arial, sans-serif',
        'color': '#34495e',
        'marginTop': '20px'
    }),
    
    dash_table.DataTable(
        for_display.to_dict('records'),
        [{"name": i, "id": i} for i in for_display.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial, sans-serif',
            'color': '#2c3e50',
            'backgroundColor': '#ecf0f1',
            'border': '1px solid #bdc3c7'
        }, 
        style_cell_conditional=[
            {'if': {'column_id': 'Review'}, 'width': '60%'},
            {'if': {'column_id': 'Rating'}, 'width': '20%'},
        ],
        style_header={
            'backgroundColor': '#2980b9',
            'color': 'white',
            'fontWeight': 'bold'
        },
        page_size=10, 
        id='tbl'
    ),
    
    dbc.Alert(id='tbl_out', style={'marginTop': '20px'}),

    # First row of visualizations
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H2('Rating Bar Chart', style={
                    'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'color': '#34495e'}),
                dcc.Graph(id='rating-graph', figure=rating_fig, style={'border': '1px solid #bdc3c7'})
            ], width=6),

            dbc.Col([
                html.H2('Label Bar Chart', style={
                    'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'color': '#34495e'}),
                dcc.Graph(id='review-graph', figure=label_fig, style={'border': '1px solid #bdc3c7'})
            ], width=6)
        ], style={'marginTop': '20px'})
    ]),

    # Second row of visualizations
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H2('PCA Visualization', style={
                    'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'color': '#34495e'}),
                dcc.Graph(id='pca-graph', figure=pca_fig, style={'border': '1px solid #bdc3c7'})
            ], width=6),

            dbc.Col([
                html.H2('t-SNE Visualization', style={
                    'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'color': '#34495e'}),
                dcc.Graph(id='tsne-graph', figure=tsne_fig, style={'border': '1px solid #bdc3c7'})
            ], width=6)
        ], style={'marginTop': '20px'})
    ])
], style={'padding': '20px', 'backgroundColor': '#f4f6f7'})

# app.layout = [
#     html.H1(children='IMDB Movie Reviews', style={'textAlign':'center'}),
#     html.H2(children='Movie Reviews Dataset', style={'textAlign':'left'}),
#     dash_table.DataTable(for_display.to_dict('records'), [{"name": i, "id": i} for i in for_display.columns],
#                          style_cell={
#                             'textAlign': 'left',
#                             # 'width' : 'fit',
#                             'overflow': 'hidden',
#                             'textOverflow': 'ellipsis',
#                             'maxWidth': 0}, 
#                         style_cell_conditional=[
#                             {'if': {'column_id': 'Review'},
#                             'width': '60%'},
#                             {'if': {'column_id': 'Rating'},
#                             'width': '20%'},
#                         ],
#                          page_size=10, id='tbl'),
#     dbc.Alert(id='tbl_out'),
#     html.H2(children='Rating Bar Chart', style={'textAlign':'center'}),
#     dcc.Graph(
#         id='rating-graph',
#         figure=rating_fig
#     ),
#     html.H2(children='Label Bar Chart', style={'textAlign':'center'}),
#     dcc.Graph(
#         id='review-graph',
#         figure=label_fig
#     ),
#     html.H2(children='PCA Visualization', style={'textAlign':'center'}),
#     dcc.Graph(
#         id='pca-graph',
#         figure=pca_fig
#     ),
#     html.H2(children='t-SNE Visualization', style={'textAlign':'center'}),
#     dcc.Graph(
#         id='tsne-graph',
#         figure=tsne_fig
#     )
# ]

@callback(Output('tbl_out', 'children'), Input('tbl', 'active_cell'))
def update_graphs(active_cell):
    if active_cell :
        full_review = df.iloc[active_cell['row'], 3]
        return str(full_review)

    else:   
        return "Click the table"


if __name__ == '__main__':
    app.run(debug=True)
