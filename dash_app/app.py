# import data
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# import web
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64

path = r'C:/Users/Julien/Documents/WildCodeSchool/Projet/Projet_3/dash_app/'

app = dash.Dash(__name__)

df_desc = pd.read_csv(path + 'dataset/desciption_dash.csv', sep=",")
df_renale = pd.read_csv(path + 'dataset/mrc_disease.csv', sep=",")
df_diabete = pd.read_csv(path + 'dataset/diabete_disease.csv', sep=",")
df_liver = pd.read_csv(path + 'dataset/liver_disease.csv', sep=",")
df_hearth = pd.read_csv(path + 'dataset/hearth_disease.csv', sep=",")
df_seins = pd.read_csv(path + 'dataset/seins_disease.csv', sep=",")

# Configuration de l'application avec plusieurs pages
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Mise en page de l'accueil                  #F7EFEF
home_layout = html.Div(style={'background': '#FFFFFF', 'color': '#000000', 'textAlign': 'center', 'padding': '20px', 'font' : "oblique 16px Arial, Helvetica"}, children=[
    html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'height': '100px', 'line-height': '100px', 'font' : "oblique 30px Arial, Helvetica"}, children=[
        html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/logo.png', 'rb').read())).decode()), style={'width': "30%"}),
        dcc.Link('Accueil', href='/'),
        dcc.Link('Prediction', href='/prediction'),
        dcc.Link('Graphique', href='/graphique'),
        html.P(""),
        dcc.Link([html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/info_icon.jpg', 'rb').read())).decode()), style={'width': "10%"})], href='/info')
    ]),
    html.Div(style={'display': 'flex', 'flex-direction': 'column', 'border': '0px solid', 'justify-content': 'space-between', 'padding': '20px', 'margin-top': '60px', 'height': '600px', 'width': '80%'}, children=[
        html.P('    Maladie chronique rénale', style={'font' : "oblique 22px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(df_desc.iloc[0, 0], style={'font' : "oblique 16px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(),
        html.P('    Diabète', style={'font' : "oblique 22px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(df_desc.iloc[1, 0], style={'font' : "oblique 16px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(),
        html.P('    Cancer du foie', style={'font' : "oblique 22px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(df_desc.iloc[2, 0], style={'font' : "oblique 16px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(),
        html.P('    Cancer cardiaque', style={'font' : "oblique 22px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(df_desc.iloc[3, 0], style={'font' : "oblique 16px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(),
        html.P('    Cancer du seins', style={'font' : "oblique 22px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(df_desc.iloc[4, 0], style={'font' : "oblique 16px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(' '),
        html.P(' ')
    ])
])





# Mise en page des pédiction
page_prediction_layout =html.Div(style={'background': '#FFFFFF', 'color': '#000000', 'textAlign': 'center', 'padding': '20px', 'font' : "oblique 16px Arial, Helvetica"}, children=[
    html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'height': '100px', 'line-height': '100px', 'font' : "oblique 30px Arial, Helvetica"}, children=[
        html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/logo.png', 'rb').read())).decode()), style={'width': "30%"}),
        dcc.Link('Acceuil', href='/'),
        dcc.Link('Prediction', href='/prediction'),
        dcc.Link('Graphique', href='/graphique'),
        html.P(""),
        dcc.Link([html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/info_icon.jpg', 'rb').read())).decode()), style={'width': "10%"})], href='/info')
    ]),
    html.Div(style={'display': 'flex', 'border': '0px solid', 'padding': '20px', 'margin-top': '60px', 'height': 'auto', 'width': '95%'}, children=[
        html.Div(style={'display': 'flex', 'flex-direction': 'column', 'border': '2px solid', 'height': '100%', 'width': '20%'}, children=[
            html.Button('Maladie Rénale', id='button_new_encadrement_R', style={'font-size': '20px'}),
            html.Div(id='encadrement_container_R'),
            html.Button('Maladie Diabète', id='button_new_encadrement_D', style={'font-size': '20px'}),
            html.Div(id='encadrement_container_D'),
            html.Button('Maladie du Foie', id='button_new_encadrement_F', style={'font-size': '20px'}),
            html.Div(id='encadrement_container_F'),
            html.Button('Maladie Cardiaque', id='button_new_encadrement_C', style={'font-size': '20px'}),
            html.Div(id='encadrement_container_C'),
            html.Button('Cancer du Seins', id='button_new_encadrement_S', style={'font-size': '20px'}),
            html.Div(id='encadrement_container_S'),
        ]),
        html.Div(style={'width': '10%'}),
        html.Div(style={'display': 'flex', 'flex-direction': 'column', 'border': '0px solid', 'height': '600px', 'width': '70%'}, children=[
            html.Div(style={'display': 'flex', 'flex-direction': 'column', 'border': '2px solid', 'text-align': 'left', 'height': '30%', 'width': '100%'}, children=[
                html.P("Description :", style={'padding-left': '100px'}),
                html.Div(id='output_description', style={'padding-left': '40px', 'max-width': 'calc(100% - 150px)'})
           ]),
            html.Div(style={'height': '10%'}),
            html.Div(style={'display': 'flex', 'border': '2px solid', 'height': '20%', 'width': '100%'}, children=[ # encadrement du milieu
                html.Button('Launch Rénale', id='lancement_de_l_algo_R', n_clicks=0, style={'font-size': '20px', 'width': '20%'}),
                html.Button('Launch Diabete', id='lancement_de_l_algo_D', n_clicks=0, style={'font-size': '20px', 'width': '20%'}),
                html.Button('Launch Foie', id='lancement_de_l_algo_F', n_clicks=0, style={'font-size': '20px', 'width': '20%'}),
                html.Button('Launch Cardiaque', id='lancement_de_l_algo_C', n_clicks=0, style={'font-size': '20px', 'width': '20%'}),
                html.Button('Launch Seins', id='lancement_de_l_algo_S', n_clicks=0, style={'font-size': '20px', 'width': '20%'}),
            ]),
            html.Div(style={'height': '10%'}),
            html.Div(style={'display': 'flex', 'flex-direction': 'column', 'border': '2px solid', 'height': '30%', 'width': '100%'}, children=[ # Resultat prediction
                html.P(id='resultat_R'),
                html.P(id='resultat_D'),
                html.P(id='resultat_F'),
                html.P(id='resultat_C'),
                html.P(id='resultat_S'),
            ]),
        ]),
    ]),
    html.Div(id='output_ghost_button', style={'height': '50px', 'width': '50px'})
])


# Fonction qui permets de toujours pouvoir utilisé la description lorsque les 5 ongles ne sont pas ouverts
@app.callback(
    Output('output_ghost_button', 'children'), [Input('button_new_encadrement_R', 'n_clicks'), Input('button_new_encadrement_D', 'n_clicks'),
    Input('button_new_encadrement_F', 'n_clicks'), Input('button_new_encadrement_C', 'n_clicks'), Input('button_new_encadrement_S', 'n_clicks')]
)
def add_new_encadrement(n_clicks_R, n_clicks_D, n_clicks_F, n_clicks_C, n_clicks_S):
    encadrement_list = []
    encadrement_R = [html.Div(children=[
        html.Button('button 1', id='but_desc_R_1', style={'border': '2px solid'}), html.Button(id='but_desc_R_2'), html.Button(id='but_desc_R_3'), html.Button(id='but_desc_R_4'), 
        html.Button(id='but_desc_R_5'), html.Button(id='but_desc_R_6'), html.Button(id='but_desc_R_7'), html.Button(id='but_desc_R_8')
    ])]
    encadrement_D = [html.Div(children=[
        html.Button('button 1', id='but_desc_D_1'), html.Button(id='but_desc_D_2'), html.Button(id='but_desc_D_3'), html.Button(id='but_desc_D_4'), 
        html.Button(id='but_desc_D_5'), html.Button(id='but_desc_D_6'), html.Button(id='but_desc_D_7'), html.Button(id='but_desc_D_8')
    ])]
    encadrement_F = [html.Div(children=[
        html.Button('button 1', id='but_desc_F_1'), html.Button(id='but_desc_F_2'), html.Button(id='but_desc_F_3'), 
        html.Button(id='but_desc_F_4'), html.Button(id='but_desc_F_5'), html.Button(id='but_desc_F_6')
    ])]
    encadrement_C = [html.Div(children=[
        html.Button('button 1', id='but_desc_C_1'), html.Button(id='but_desc_C_2'), html.Button(id='but_desc_C_3'), html.Button(id='but_desc_C_4'), 
        html.Button(id='but_desc_C_5'), html.Button(id='but_desc_C_6'), html.Button(id='but_desc_C_7'), html.Button(id='but_desc_C_8'),
        html.Button(id='but_desc_C_9'), html.Button(id='but_desc_C_10'), html.Button(id='but_desc_C_11'), html.Button(id='but_desc_C_12'), 
    ])]
    encadrement_S = [html.Div(children=[
        html.Button('button 1', id='but_desc_S_1'), html.Button(id='but_desc_S_2'), html.Button(id='but_desc_S_3'), html.Button(id='but_desc_S_4')
    ])]

    encadrement_list = []
    context = dash.callback_context

    if n_clicks_R % 2 == 0:
        print('nice click')
        encadrement_list.append(encadrement_R)

    if n_clicks_D % 2 == 0:
        encadrement_list.append(encadrement_D)

    if n_clicks_F % 2 == 0:
        encadrement_list.append(encadrement_F)

    if n_clicks_C % 2 == 0:
        encadrement_list.append(encadrement_C)

    if n_clicks_S % 2 == 0:
        encadrement_list.append(encadrement_S)

    return encadrement_list


# Renale   ->  ['Haemoglobin', 'Specific_Gravity', 'Blood_Urea', 'Blood_Glucose_Random', 'Blood_Pressure', 'Pus_Cell', 'appetit', 'Sugar']    <- Renale
@app.callback(
    Output('encadrement_container_R', 'children'), [Input('button_new_encadrement_R', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'margin-left': '10px'}, children=[
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Haemoglobin', id='but_desc_R_1', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col1_R', type='text', placeholder='input a value', style={'width': '25%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Specific Gravity', id='but_desc_R_2', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col2_R', type='text', placeholder='input a value', style={'width': '25%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Blood Urea', id='but_desc_R_3', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col3_R', type='text', placeholder='input a value', style={'width': '25%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Blood Glucose Random', id='but_desc_R_4', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col4_R', type='text', placeholder='input a value', style={'width': '25%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Blood Pressure', id='but_desc_R_5', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col5_R', type='text', placeholder='input a value', style={'width': '25%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Pus Cell', id='but_desc_R_6', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col6_R', type='text', placeholder='input a value', style={'width': '25%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Appetit', id='but_desc_R_7', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col7_R', type='text', placeholder='input a value', style={'width': '25%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row'}, children=[
                html.Button('Sugar', id='but_desc_R_8', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col8_R', type='text', placeholder='input a value', style={'width': '25%', 'margin-left': 'auto'}),
            ]),
        ]))
    return encadrement_list

# Diabete   ->  ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']  <- Diabete
@app.callback(
    Output('encadrement_container_D', 'children'), [Input('button_new_encadrement_D', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'margin-left': '10px'}, children=[
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Pregnancies', id='but_desc_D_1', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col1_D', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Glucose', id='but_desc_D_2', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col2_D', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Blood Pressure', id='but_desc_D_3', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col3_D', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('SkinThickness', id='but_desc_D_4', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col4_D', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Insulin', id='but_desc_D_5', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col5_D', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('BMI', id='but_desc_D_6', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col6_D', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Diabetes Pedigree Function', id='but_desc_D_7', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col7_D', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row'}, children=[
                html.Button('Age', id='but_desc_D_8', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col8_D', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
        ]))
    return encadrement_list

# Foie   ->  ['Direct_Bilirubin', 'Total_Bilirubin', 'Alamine_Aminotransferase', 'Alkaline_Phosphotase', 'Albumin', 'Albumin_and_Globulin_Ratio']  <- Foie
@app.callback(
    Output('encadrement_container_F', 'children'), [Input('button_new_encadrement_F', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'margin-left': '10px'}, children=[
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Direct Bilirubin', id='but_desc_F_1', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col1_F', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Total Bilirubin', id='but_desc_F_2', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col2_F', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Alamine Aminotransferase', id='but_desc_F_3', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col3_F', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Alkaline Phosphotase', id='but_desc_F_4', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col4_F', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('Albumin', id='but_desc_F_5', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col5_F', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row'}, children=[
                html.Button('Albumin and Globulin Ratio', id='but_desc_F_6', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col6_F', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
        ]))
    return encadrement_list

# Cardiaque    -> ['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'] <- Cardiaque
@app.callback(
    Output('encadrement_container_C', 'children'), [Input('button_new_encadrement_C', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'margin-left': '10px'}, children=[
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('age', id='but_desc_C_1', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col1_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('sex', id='but_desc_C_2', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col2_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('cp', id='but_desc_C_3', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col3_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('trestbps', id='but_desc_C_4', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col4_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('chol', id='but_desc_C_5', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col5_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('restecg', id='but_desc_C_6', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col6_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('thalach', id='but_desc_C_7', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col7_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('exang', id='but_desc_C_8', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col8_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('oldpeak', id='but_desc_C_9', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col9_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('slope', id='but_desc_C_10', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col10_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('ca', id='but_desc_C_11', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col11_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('thal', id='but_desc_C_12', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col12_C', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ])
        ]))
    return encadrement_list

# Seins ->  [area_mean, concavity_mean, texture_mean, smoothness_mean]  <- Seins
@app.callback(
    Output('encadrement_container_S', 'children'), [Input('button_new_encadrement_S', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'margin-left': '10px'}, children=[
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('area mean', id='but_desc_S_1', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col1_S', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('concavity mean', id='but_desc_S_2', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col2_S', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('texture mean', id='but_desc_S_3', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col3_S', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
            html.Div(style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '10px'}, children=[
                html.Button('smoothness mean', id='but_desc_S_4', style={'background-color': 'transparent', 'border': '0px solid'}),
                dcc.Input(id='V_col4_S', type='text', placeholder='input a value', style={'width': '30%', 'margin-left': 'auto'}),
            ]),
        ]))
    return encadrement_list


@app.callback(
    Output('output_description', 'children'),
    [Input(f'but_desc_R_{i}', 'n_clicks') for i in range(1, 9)] +
    [Input(f'but_desc_D_{i}', 'n_clicks') for i in range(1, 9)] +
    [Input(f'but_desc_F_{i}', 'n_clicks') for i in range(1, 7)] +
    [Input(f'but_desc_C_{i}', 'n_clicks') for i in range(1, 13)] +
    [Input(f'but_desc_S_{i}', 'n_clicks') for i in range(1, 5)]
)
def update_output(*n_clicks_list):
    context = dash.callback_context
    button_id = context.triggered_id.split('.')[0] if context.triggered_id else None

    for i, n_clicks in enumerate(n_clicks_list, start=1):
        print(f'test{i} -> {n_clicks} --> {button_id}')
        if button_id == f'but_desc_R_{i}':
            return df_desc.iloc[i, 1]
        elif button_id == f'but_desc_D_{i}':
            return df_desc.iloc[i+8, 1]
        elif button_id == f'but_desc_F_{i}':
            return df_desc.iloc[i+16, 1]
        elif button_id == f'but_desc_C_{i}':
            return df_desc.iloc[i+22, 1]
        elif button_id == f'but_desc_S_{i}':
            return df_desc.iloc[i+34, 1]

    return "(<-_->)"


# Renale
@app.callback(
    Output('resultat_R', 'children'), [Input('lancement_de_l_algo_R', 'n_clicks')],
    [State('V_col1_R', 'value'), State('V_col2_R', 'value'), State('V_col3_R', 'value'), State('V_col4_R', 'value'),
     State('V_col5_R', 'value'), State('V_col6_R', 'value'), State('V_col7_R', 'value'), State('V_col8_R', 'value')]
)
def update_output(n_clicks, V_col1_R, V_col2_R, V_col3_R, V_col4_R, V_col5_R, V_col6_R, V_col7_R, V_col8_R):
    if n_clicks > 0:
        X = df_renale[['Haemoglobin', 'Specific_Gravity', 'Blood_Urea', 'Blood_Glucose_Random', 'Blood_Pressure', 'Pus_Cell', 'appetit', 'Sugar']]
        y = df_renale['resultat']
        model = MLPClassifier(hidden_layer_sizes=(100, 50, 10), learning_rate_init=0.005180393817686888, max_iter=272, random_state=0)
        model = BaggingClassifier(base_estimator=model,n_estimators=8)
        model.fit(X, y)
        new_data = pd.DataFrame({
            'Haemoglobin': [V_col1_R], 'Specific_Gravity': [V_col2_R], 'Blood_Urea': [V_col3_R], 'Blood_Glucose_Random': [V_col4_R],
            'Blood_Pressure': [V_col5_R], 'Pus_Cell': [V_col6_R], 'appetit': [V_col7_R], 'Sugar': [V_col8_R]})
        prediction = model.predict(new_data)
        if prediction[0] == 1:
            return "D'après les donées que vous nous avez passé ainsi que notre annalyse, vous avez probabablement une maladie chronique rénale. Nous vous Consayons de consulter votre médecin\n "
        else:
            return "D'après les donées que vous nous avez passé ainsi que notre annalyse, vous n'avez probabablement une maladie chronique rénale\n "
    else:
        "No info"


# Diabete
@app.callback(
    Output('resultat_D', 'children'), [Input('lancement_de_l_algo_D', 'n_clicks')],
    [State('V_col1_D', 'value'), State('V_col2_D', 'value'), State('V_col3_D', 'value'), State('V_col4_D', 'value'),
     State('V_col5_D', 'value'), State('V_col6_D', 'value'), State('V_col7_D', 'value'), State('V_col8_D', 'value')]
)
def update_output(n_clicks, V_col1_D, V_col2_D, V_col3_D, V_col4_D, V_col5_D, V_col6_D, V_col7_D, V_col8_D):
    if n_clicks > 0 :
        X = df_diabete.drop('Outcome', axis = 1)
        y = df_diabete['Outcome']
        model_treeClassif = DecisionTreeClassifier(criterion= 'gini', max_depth= 5, min_samples_leaf= 2, min_samples_split= 10)
        model_treeClassif.fit(X, y)
        new_data = pd.DataFrame({
            'Pregnancies': [V_col1_D], 'Glucose': [V_col2_D], 'BloodPressure': [V_col3_D], 'SkinThickness': [V_col4_D],
            'Insulin': [V_col5_D], 'BMI': [V_col6_D], 'DiabetesPedigreeFunction': [V_col7_D], 'Age': [V_col8_D]})
        prediction = model_treeClassif.predict(new_data)
        if prediction[0] == 1:
            return "D'après les donées que vous nous avez passé ainsi que notre annalyse, vous avez probabablement le diabete. Nous vous Consayons de consulter votre médecin\n "
        else:
            return "D'après les donées que vous nous avez passé ainsi que notre annalyse, vous n'avez probabablement le diabete\n "
    else:
        return "No info"



# Foie
@app.callback(
    Output('resultat_F', 'children'), [Input('lancement_de_l_algo_F', 'n_clicks')],
    [State('V_col1_F', 'value'), State('V_col2_F', 'value'), State('V_col3_F', 'value'), State('V_col4_F', 'value'), State('V_col5_F', 'value'), State('V_col6_F', 'value')]
)
def update_output(n_clicks, V_col1_F, V_col2_F, V_col3_F, V_col4_F, V_col5_F, V_col6_F):
    if n_clicks > 0:
        X = df_liver[['Direct_Bilirubin', 'Total_Bilirubin', 'Alamine_Aminotransferase', 'Alkaline_Phosphotase', 'Albumin', 'Albumin_and_Globulin_Ratio']]
        y = df_liver['Dataset']
        model_neural = MLPClassifier(hidden_layer_sizes=(100, 50, 10), learning_rate_init=0.0037244159149844845, max_iter=209, random_state=0)
        model_neural.fit(X, y)
        new_data = pd.DataFrame({
            'Direct_Bilirubin': [V_col1_F], 'Total_Bilirubin': [V_col2_F], 'Alamine_Aminotransferase': [V_col3_F],
            'Alkaline_Phosphotase': [V_col4_F], 'Albumin': [V_col5_F], 'Albumin_and_Globulin_Ratio': [V_col6_F]})
        prediction = model_neural.predict(new_data)
        if prediction[0] == 1:
            return "D'après les donées que vous nous avez passé ainsi que notre annalyse, vous avez probabablement une maladie de foie. Nous vous Consayons de consulter votre médecin\n "
        else:
            return "D'après les donées que vous nous avez passé ainsi que notre annalyse, vous n'avez probabablement une maladie de foie\n "
    else:
        return "No info"


# Cardiaque
@app.callback(
    Output('resultat_C', 'children'), [Input('lancement_de_l_algo_C', 'n_clicks')],
    [State('V_col1_C', 'value'), State('V_col2_C', 'value'), State('V_col3_C', 'value'), State('V_col4_C', 'value'),
     State('V_col5_C', 'value'), State('V_col6_C', 'value'), State('V_col7_C', 'value'), State('V_col8_C', 'value'),
     State('V_col9_C', 'value'), State('V_col10_C', 'value'), State('V_col11_C', 'value'), State('V_col12_C', 'value')]
)
def update_output(n_clicks, V_col1_C, V_col2_C, V_col3_C, V_col4_C, V_col5_C, V_col6_C, V_col7_C, V_col8_C, V_col9_C, V_col10_C, V_col11_C, V_col12_C):
    if n_clicks > 0:
        X = df_hearth[['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
        y = df_hearth['target']
        model = GaussianNB(var_smoothing=6.488135039273248e-10)
        model.fit(X, y)
        new_data = pd.DataFrame({
            'age': [V_col1_C], 'sex': [V_col2_C], 'cp': [V_col3_C], 'trestbps': [V_col4_C], 'chol': [V_col5_C], 'restecg': [V_col6_C],
            'thalach': [V_col7_C], 'exang': [V_col8_C], 'oldpeak': [V_col9_C], 'slope': [V_col10_C], 'ca': [V_col11_C], 'thal': [V_col12_C]})
        prediction = model.predict(new_data)
        if prediction[0] == 1:
            return "D'après les donées que vous nous avez passé ainsi que notre annalyse, vous avez probabablement une maladie cardiaque. Nous vous Consayons de consulter votre médecin\n "
        else:
            return "D'après les donées que vous nous avez passé ainsi que notre annalyse, vous n'avez probabablement une maladie cardiaque\n "
    else:
        return "No info"


# Seins
@app.callback(
    Output('resultat_S', 'children'), [Input('lancement_de_l_algo_S', 'n_clicks')],
    [State('V_col1_S', 'value'), State('V_col2_S', 'value'), State('V_col3_S', 'value'), State('V_col4_S', 'value')]
)
def update_output(n_clicks, V_col1_S, V_col2_S, V_col3_S, V_col4_S):
    if n_clicks > 0:
        X = df_seins[['area_mean', 'concavity_mean', 'texture_mean', 'smoothness_mean']]
        y = df_seins['diagnosis']
        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
        random_forest_model.fit(X, y)
        new_data = pd.DataFrame([{
                'area_mean': V_col1_S, 'concavity_mean': V_col2_S,
                'texture_mean': V_col3_S, 'smoothness_mean': V_col4_S}],
        columns=['area_mean', 'concavity_mean', 'texture_mean', 'smoothness_mean'])
        prediction = random_forest_model.predict(new_data)
        if prediction[0] == 1:
            return "Après l'évaluation de vos variables médicales, vous avez une grande probabilité d'être attient(e) d'une tumeur maligne. Consultez votre médecin avec urgence\n "
        else:
            return "Après l'évaluation de vos variables médicales, vous avez une grande probabilité d'être attient(e) d'une tumeur bénigne. Pour plus d'information, consultez votre médecin\n "
    else:
        return "No info"





# Mise en page des graphiques
page_graphique = html.Div(style={'background': '#FFFFFF', 'color': '#000000', 'textAlign': 'center', 'padding': '20px', 'font' : "oblique 16px Arial, Helvetica"}, children=[
    html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'height': '100px', 'line-height': '100px', 'font' : "oblique 30px Arial, Helvetica"}, children=[
        html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/logo.png', 'rb').read())).decode()), style={'width': "30%"}),
        dcc.Link('Acceuil', href='/'),
        dcc.Link('Prediction', href='/prediction'),
        dcc.Link('Graphique', href='/graphique'),
        html.P(""),
        dcc.Link([html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/info_icon.jpg', 'rb').read())).decode()), style={'width': "10%"})], href='/info')
    ]), 
    html.Div(style={'display': 'flex', 'border': '0px solid', 'padding': '20px',  'margin-top': '70px', 'height': 'auto', 'width': '90%'}, children=[
        html.Div(style={'display': 'flex', 'flex-direction': 'column', 'border': '2px solid', 'height': '100%', 'width': '20%'}, children=[
            html.P("Liste des Graphique", style={'font-size': '30px'}),
            html.Button('Maladie Rénale', id='button_new_encadrement_graph_R', style={'font-size': '20px', 'background-color': '#D9E2F0'}),
            html.Div(id='encadrement_container_graph_R'),
            html.Button('Maladie Diabète', id='button_new_encadrement_graph_D', style={'font-size': '20px', 'background-color': '#D9E2F0'}),
            html.Div(id='encadrement_container_graph_D'),
            html.Button('Maladie du Foie', id='button_new_encadrement_graph_F', style={'font-size': '20px', 'background-color': '#D9E2F0'}),
            html.Div(id='encadrement_container_graph_F'),
            html.Button('Maladie Cardiaque', id='button_new_encadrement_graph_C', style={'font-size': '20px', 'background-color': '#D9E2F0'}),
            html.Div(id='encadrement_container_graph_C'),
            html.Button('Cancer du Seins', id='button_new_encadrement_graph_S', style={'font-size': '20px', 'background-color': '#D9E2F0'}),
            html.Div(id='encadrement_container_graph_S'),
        ]),
        html.Div(style={'width': '10%'}),
        html.Div(id='output_graphique', style={'display': 'flex', 'border': '0px solid', 'height': '700px', 'width': '70%'})
    ]),
])


@app.callback(
    Output('encadrement_container_graph_R', 'children'), [Input('button_new_encadrement_graph_R', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-between', 'margin': '4px'}, children=[
                html.Button('Graphique de corrélation', id='button_graph_R_1', style={'border': '1px solid', 'font-size': '15px'}),
                html.Button('Graphique de chi_square', id='button_graph_R_2', style={'border': '1px solid', 'font-size': '15px'}),
        ]))
    return encadrement_list


@app.callback(
    Output('encadrement_container_graph_D', 'children'), [Input('button_new_encadrement_graph_D', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-between', 'margin': '4px'}, children=[
                html.Button('Graphique de corrélation', id='button_graph_D_1', style={'border': '1px solid', 'font-size': '15px'}),
                html.Button('Graphique de Boxplot', id='button_graph_D_2', style={'border': '1px solid', 'font-size': '15px'}),
        ]))
    return encadrement_list


@app.callback(
    Output('encadrement_container_graph_F', 'children'), [Input('button_new_encadrement_graph_F', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-between', 'margin': '4px'}, children=[
                html.Button('Graphique de corrélation', id='button_graph_F_1', style={'border': '1px solid', 'font-size': '15px'}),
                html.Button('Graphique de kruskal', id='button_graph_F_2', style={'border': '1px solid', 'font-size': '15px'}),
        ]))
    return encadrement_list


@app.callback(
    Output('encadrement_container_graph_C', 'children'), [Input('button_new_encadrement_graph_C', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-between', 'margin': '4px'}, children=[
                html.Button('Graphique de corrélation', id='button_graph_C_1', style={'border': '1px solid', 'font-size': '15px'}),
                html.Button('Graphique de kruskal', id='button_graph_C_2', style={'border': '1px solid', 'font-size': '15px'})
        ]))
    return encadrement_list

@app.callback(
    Output('encadrement_container_graph_S', 'children'), [Input('button_new_encadrement_graph_S', 'n_clicks')]
)
def add_new_encadrement(n_clicks):
    encadrement_list = []
    n_clicks = n_clicks or 0
    if n_clicks % 2 == 1:
        encadrement_list.append(html.Div(style={'border': '0px solid', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-between', 'margin': '4px'}, children=[
                html.Button('Graphique de corrélation', id='button_graph_S_1', style={'border': '1px solid', 'font-size': '15px'}),
                html.Button('Nom du  S2', id='button_graph_S_2', style={'border': '1px solid', 'font-size': '15px'}) # grap_seins_cas_bénignes_maliges_by_rayon_moyen_de_la_tumeur
        ]))
    return encadrement_list


@app.callback(
    Output('output_graphique', 'children'),
    [Input('button_graph_R_1', 'n_clicks'), Input('button_graph_R_2', 'n_clicks'),
     Input('button_graph_D_1', 'n_clicks'), Input('button_graph_D_2', 'n_clicks'),
     Input('button_graph_F_1', 'n_clicks'), Input('button_graph_F_2', 'n_clicks'),
     Input('button_graph_C_1', 'n_clicks'), Input('button_graph_C_2', 'n_clicks'),
     Input('button_graph_S_1', 'n_clicks'), Input('button_graph_S_2', 'n_clicks'),]
)
def update_output(n_clicks_1, n_clicks_2, n_clicks_3, n_clicks_4, n_clicks_5, n_clicks_6, n_clicks_7, n_clicks_8, n_clicks_9, n_clicks_10):
    context = dash.callback_context
    button_id = context.triggered_id.split('.')[0] if context.triggered_id else None
    print('good')
    if button_id == 'button_graph_R_1' and n_clicks_1:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_renale_corr.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_R_2' and n_clicks_2:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_renale_chi_square.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_D_1' and n_clicks_3:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_diabete_corr.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_D_2' and n_clicks_4:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/graph_diabete_boxplot.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_F_1' and n_clicks_5:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_foie_corr.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_F_2' and n_clicks_6:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_foie_kruskal.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_C_1' and n_clicks_7:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_coeur_corr.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_C_2' and n_clicks_8:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_coeur_kruskal.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_S_1' and n_clicks_9:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_seins_corr.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    elif button_id == 'button_graph_S_2' and n_clicks_10:
        return html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/grap_seins_beligne_maligne.png', 'rb').read())).decode()), style={'width': 'auto', 'height': '700px', 'margin': 'auto'})
    else:
        return html.p("HELLO")





page_info = html.Div(style={'background': '#FFFFFF', 'color': '#000000', 'textAlign': 'center', 'padding': '20px', 'font' : "oblique 16px Arial, Helvetica"}, children=[
    html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'height': '100px', 'line-height': '100px', 'font' : "oblique 30px Arial, Helvetica"}, children=[
        html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/logo.png', 'rb').read())).decode()), style={'width': "30%"}),
        dcc.Link('Acceuil', href='/'),
        dcc.Link('Prediction', href='/prediction'),
        dcc.Link('Graphique', href='/graphique'),
        html.P(""),
        dcc.Link([html.Img(src='data:image/png;base64,{}'.format((base64.b64encode(open(path + 'image/info_icon.jpg', 'rb').read())).decode()), style={'width': "10%"})], href='/info')
    ]),
    html.Div(style={'display': 'flex', 'padding': '20px', 'margin-top': '60px', 'justify-content': 'space-between', 'flex-direction': 'column', 'width': '80%', 'height': '500px'}, children=[
        html.P(df_desc.iloc[0, 2], style={'font' : "oblique 16px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(' '),
        html.P(df_desc.iloc[1, 2], style={'font' : "oblique 16px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(' '),
        html.P(df_desc.iloc[2, 2], style={'font' : "oblique 16px Arial, Helvetica", 'textAlign': 'left'}),
        html.P(' ')
    ])
])


# Gestion du contenu en fonction de l'URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/prediction':
        return page_prediction_layout
    elif pathname == '/graphique':
        return page_graphique
    elif pathname == '/info':
        return page_info
    else:
        return home_layout
    
if __name__ == '__main__':
    app.run_server(debug=False)
