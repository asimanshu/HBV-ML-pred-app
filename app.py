#!/usr/bin/env python3.10.4
# coding: utf-8

# In[1]:


from dash import Dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from sklearn import metrics
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import statsmodels as sm


# In[2]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# In[3]:


data_set = pd.read_excel("Data.xlsx")


# In[4]:


data_set = data_set.replace("M", "Male")
data_set = data_set.replace("M ", "Male")
data_set = data_set.replace("F", "Female")
data_set = data_set.replace("F ", "Female")

data_set['HBV_DNA_Load'] = np.log10(data_set['HBV_DNA_Load'])
data_set['HBeAg'] = np.log10(data_set['HBeAg'])
data_set['HBsAg'] = np.log10(data_set['HBsAg'])
data_set['Age'] = np.log10(data_set['Age'])

models = {'Linear': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor,
          'Random Forest': RandomForestRegressor}


# In[5]:


app.layout = dbc.Container([
    html.Div(children=[
        dbc.NavbarSimple(children=[dbc.DropdownMenu(children=[
                                       dbc.DropdownMenuItem(html.A("Background", href="#nav-background")),
                                       dbc.DropdownMenuItem(html.A("Distribution", href="#nav-distribution")),
                                       dbc.DropdownMenuItem(html.A("Regression", href="#nav-reg-mod")),
                                       dbc.DropdownMenuItem(html.A("Actual vs Predicted", href="#nav-act-pred")),
                                       dbc.DropdownMenuItem(html.A("Prediction Error", href="#nav-pred-err")),
                                       dbc.DropdownMenuItem(html.A("Residuals", href="#nav-residuals")),
                                       dbc.DropdownMenuItem(html.A("Cross validation", href="#nav-cross-val")),
                                       dbc.DropdownMenuItem(html.A("Grid Search cv", href="#nav-grid-cv")),
                                       dbc.DropdownMenuItem(html.A("Prediction of HBV DNA", href="#pred-mod-dna")),
                                       dbc.DropdownMenuItem(html.A("About Data", href="#nav-data")),
                                       dbc.DropdownMenuItem(html.A("Conclusion", href="#nav-conc")),
                                       dbc.DropdownMenuItem(html.A("About Author", href="#nav-author")),
                                   ],
                                       nav=True,
                                       in_navbar=True,
                                       label="Jump to",
                                   ),
                                   ],
                         brand="HBV Machine Learning Dashboard",
                         brand_href="#",
                         color="primary",
                         dark=True,
                         )
    ]),
    html.Div(children=[
        html.H1(children='Hepatitis B Virus Analysis Dashboard', style={'text-align': 'center'}),
        html.H3(id='nav-background', children='Prediction Analyses of Hepatitis B Virus (HBV) infection in population of Bihar.', style={'text-align': 'center'}),
        html.Div(children=[
            html.P("Viral hepatitis is a systemic infection that predominantly affects the liver and leads to its inflammation. Its infection causes both chronic and acute diseases. This issue has been considered as a public health problem globally and the cause for issues like fatty liver, autoimmunity, alcohol, other viral infection and drug toxicity. Infections with the hepatitis viruses are major risk factors for fibrosis and cirrhosis, and HCC. Infection caused by hepatitis is responsible for the deaths of 1.34 million in 2015. These deaths are compared to the number of deaths caused by tuberculosis and even higher than number of deaths caused by HIV."),
            html.Div([html.A(href='https://www.cdc.gov/hepatitis/hbv/index.htm', children=[html.P('Centers for Disease Control and Prevention')])]),
            html.P("This dashboard presents prediction analyses using different machine learning algorithms in 149 HBV infected clinical samples from the population of Bihar."),
            ]),
    ]),
    html.Div(children=[
        html.H2(["Feature-wise distribution in HBV cases"], style={'text-align': 'center'}, id='nav-distribution'),
        html.Div([html.Label('Select Feature for distribution'),
                  dcc.Dropdown(
                      id='feat',
                      options=[
                          {'label': 'Age', 'value': 'Age'},
                          {'label': 'Gender', 'value': 'Gender', 'disabled': True},
                          {'label': 'HBsAg', 'value': 'HBsAg'},
                          {'label': 'HBeAg', 'value': 'HBeAg'},
                          {'label': 'HBV DNA Load', 'value': 'HBV_DNA_Load'},
                      ],
                      value='Age', multi=False, persistence=True
                  ),
                  ], style={'width': '42%', 'display': 'inline-block'}),
        html.Div([html.Label('Select marginal type'),
                  dcc.Dropdown(
                      id='distro',
                      options=[
                          {'label': 'Box', 'value': 'box'},
                          {'label': 'Violin', 'value': 'violin'},
                          {'label': 'Rug', 'value': 'rug'},
                      ],
                      value='box', multi=False, persistence=True
                  ),
                  ], style={'width': '42%', 'float': 'right', 'display': 'inline-block'}),
        html.Div([
            html.Div(dcc.Graph(id='histo')),
        ]),
    ]),
    html.Div(children=[
        html.H2(["Regression Models in HBV cases"], style={'text-align': 'center'}, id='nav-reg-mod'),
        html.Div([html.Label('Select Feature'),
                  dcc.Dropdown(
                      id='feat1',
                      options=[
                          {'label': 'Age', 'value': 'Age'},
                          {'label': 'Gender', 'value': 'Gender', 'disabled': True},
                          {'label': 'HBsAg', 'value': 'HBsAg'},
                          {'label': 'HBeAg', 'value': 'HBeAg'},
                          {'label': 'HBV DNA Load', 'value': 'HBV_DNA_Load', 'disabled': True},
                      ],
                      value='Age', multi=False, persistence=True
                  ),
                  ], style={'width': '42%', 'display': 'inline-block'}),
        html.Div([html.Label('Select regression model'),
                  dcc.Dropdown(
                      id='regtype',
                      options=["Linear", "Decision Tree", "k-NN", "Random Forest"],
                      value="Linear", multi=False, persistence=True
                  ),
                  ], style={'width': '42%', 'float': 'right', 'display': 'inline-block'}),
        html.Div([
            html.Div(dcc.Graph(id='reg')),
        ]),
    ]),
    html.Div(children=[
        html.H2(["Actual vs Predicted"], style={'text-align': 'center'}, id='nav-act-pred'),
        html.Div([html.Label('Select model'),
                  dcc.Dropdown(
                      id='act_pred_mod',
                      options=["Linear", "Decision Tree", "k-NN", "Random Forest"],
                      value="Linear", multi=False, persistence=True
                  ),
                  ]),
        html.Div([
            html.Div(dcc.Graph(id='act_pred')),
        ]),
    ]),
    html.Div(children=[
        html.H2(["Prediction error analysis"], style={'text-align': 'center'}, id='nav-pred-err'),
        html.Div([html.Label('Select model'),
                  dcc.Dropdown(
                      id='modtype',
                      options=["Linear", "Decision Tree", "k-NN", "Random Forest"],
                      value="Linear", multi=False, persistence=True
                  ),
                  ]),
        html.Div([
            html.Div(dcc.Graph(id='prederr')),
        ]),
    ]),
    html.Div(children=[
        html.H2(["Residuals analyses"], style={'text-align': 'center'}, id='nav-residuals'),
        html.Div([html.Label('Select model'),
                  dcc.Dropdown(
                      id='regmod',
                      options=["Linear", "Decision Tree", "k-NN", "Random Forest"],
                      value="Linear", multi=False, persistence=True
                  )
                  ], style={'width': '42%', 'display': 'inline-block'}),
        html.Div([html.Label('Select marginal type'),
                  dcc.RadioItems(
                      id='marg_type',
                      options=[
                          {'label': 'Box', 'value': 'box'},
                          {'label': 'Violin', 'value': 'violin'},
                          {'label': 'Rug', 'value': 'rug'},
                      ],
                      value='box'
                  ),
                  ], style={'width': '30%', 'float': 'right', 'display': 'inline-block'}),
        html.Div([
            html.Div(dcc.Graph(id='residuals')),
        ]),
    ]),
    html.Div(children=[
        html.H2(["Cross-validation"],
                style={'text-align': 'center'}, id='nav-cross-val'),
        html.Div([html.Label('Select model'),
                  dcc.RadioItems(
                      id='radio',
                      options=[{'label': 'Linear', 'value': 'Linear'},
                               {'label': 'kNN', 'value': 'kNN'},
                               ],
                      value='Linear'
                  )
                  ]),
        html.Div([
            html.Div(dcc.Graph(id='reg_coef')),
        ]),
    ]),
    html.Div(children=[
        html.H2(["Grid search cross validation (cv)"], style={'text-align': 'center'}, id='nav-grid-cv'),
        html.Div([html.Label('Select cv number'),
                  dcc.Dropdown(
                      id='cv_no',
                      options=[{'label': '2', 'value': 2},
                               {'label': '3', 'value': 3},
                               {'label': '4', 'value': 4},
                               {'label': '5', 'value': 5},
                               {'label': '6', 'value': 6},
                               {'label': '7', 'value': 7},
                               {'label': '8', 'value': 8},
                               {'label': '9', 'value': 9},
                               ], value=2
                  )
                  ], style={'width': '42%', 'display': 'inline-block'}),
        html.Div([html.Label('Select model'),
                  dcc.RadioItems(
                      id='rad_cv',
                      options=["Decision Tree", "Random Forest"],
                      value="Decision Tree"
                  )
                  ], style={'width': '30%', 'float': 'right', 'display': 'inline-block'}),
        html.Div([
            html.Div(dcc.Graph(id='grid_cv_hmap')),
        ]),
        html.Div([
            html.Div(dcc.Graph(id='grid_cv_box')),
        ]),
    ]),
    html.Div(children=[
        html.P("*Note: While selecting RandomForest for grid search, please give it few seconds (apprx. 15-30s) to process and return the graph (for larger cv number in random forest, takes longer to process, approx. 30s). Gridsearch technique is computationally expensive process, please be patient. "),
    ]),
    html.Div(children=[
        html.H3(["Prediction of HBV DNA Load from unknown values"], style={'text-align': 'center'}, id='pred-mod-dna'),
        html.P('*Note: Please input age, clinical values of HBsAg and HBeAg to get the prediction of HBV DNA load.'),
        html.Div([html.Label('Select model'),
                  dcc.RadioItems(
                      id='mod_pred',
                      options=["Linear", "Decision Tree", "k-NN", "Random Forest"],
                      value="Decision Tree"
                  )
                  ]),
        html.Div([html.Label('Input Age'),
                  dcc.Input(id="input1", placeholder="Age", type="text")
                  ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([html.Label('Input HBsAg value'),
                  dcc.Input(id="input2", placeholder="Clinical HBsAg value", type="text")
                  ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([html.Label('Input HBeAg value'),
                  dcc.Input(id="input3", placeholder="Clinical HBeAg vlaue", type="text")
                  ], style={'width': '30%', 'float': 'right', 'display': 'inline-block'}),
        html.Div([
            html.Div(html.Table([
                html.Tr([html.Td(['Predicted HBV DNA Load:']), html.Td(id='predout')])])),
        ]),
    ]),
    html.Div(children=[
        html.H3("About Data", style={'text-align': 'center'}, id='nav-data'),
        html.P("For the current study, 149 patients were randomly selected from group of patients, who were presented with the symptoms of Hepatitis B viral infection. The samples were collected from Patna Medical Hospital and College, Nalanda Medical Hospital and College and Indira Gandhi Institute Medical Sciences, Patna. An intravenous blood was collected with the consent of patients and transferred in the EDTA charged vials from the patients. Hepatitis B surface Antigen (HBsAg) and Hepatitis B envelop Antigen (HBeAg) were determined by protocol of standard ELISA method. The HBV DNA load was evaluated by amplifying the precore DNA of HBV using Real Time (RT) – Polymerase Chain Reaction (PCR). The entire work was performed at Amplipath Diagnostic and Research Centre, Patna. The study was approved by local human ethics committee."),
    ]),
    html.Div(children=[
        html.H2("Conclusion", style={'text-align': 'center'}, id='nav-conc'),
        html.P("Cross validation for linear model shows HBsAg has the highest weight or the effect over the prediction of HBV DNA Load, followed by HBeAg and Age. The grid search as well as n-neighbor curve suggest the k-neighbor for the dataset is 7. Grid search of cross validation for Decision Tree and RandomForest suggest high r-squared with maximum depth up to 4. It is noteworthy to understand that all the hyperparameter (2-9 in the given selection) have higher r-squared except for one, which has relatively lower r-squared. Although, the maximum r-squared for decision tree and random forest is almost similar, the interquartile of r-squared is greater in random forest than that of decision tree."),
        html.P("In conclusion, it is always best approach to validate your data before selecting your model and validate your model before implementing on real data. I have presented different types of models and showed how they perform in the specific dataset. Nonetheless, success of a model largely depends on the data set. If kNN and random forest works for me, they may not work for your dataset. Hence it is wise to take every measure to validate your model."),
    ]),
    html.Div(children=[
        html.H3("About Author", style={'text-align': 'centre'}, id='nav-author'),
        html.H4("Aseem Kumar Anshu"),
        html.P("I have completed Ph.D and currently working at Amplipath Diagnostic and Research Centre, Patna, Bihar. "),
    ]),
    html.Div(children=[
        html.H6("Source"),
        html.Div([html.A(href='https://plotly.com/', children=[html.P('Plotly')])]),
    ]),
    html.Div(children=[
        html.P("*Disclaimer: The data is property of Amplipath Diagnostic and Research Centre, Patna. Use of the data or the graphical results without permission and/or citation is discouraged."),
    ]),
], style={'verticalAlign': 'middle', 'text-align': 'justify', 'text-justify': 'inter-word'}, className="mb-5")


# In[6]:


@app.callback(
    Output('histo', 'figure'),
    [Input('feat', 'value'),
     Input('distro', 'value')]
)
def build_graph(column_chosen1, type_chosen):
    dff = data_set
    fig = px.histogram(dff, x=column_chosen1, marginal=type_chosen, color='Gender', labels={
        "Age": "Log10 Age",
        "HBsAg": "Log10 HBsAg",
        "HBeAg": "Log10 HBeAg",
        "HBV_DNA_Load": "Log10 HBV DNA Load"
    })
    return fig


# In[7]:


@app.callback(
    Output('reg', 'figure'),
    [Input('feat1', 'value'),
     Input('regtype', 'value')]
)
def build_reg(column_chosen2, model_chosen1):
    dff = data_set
    X = dff[column_chosen2]
    y = dff['HBV_DNA_Load']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_train = X_train.values.reshape((-1, 1))

    model = models[model_chosen1]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
    	go.Scatter(x=X_test.squeeze(), y=y_test,
                   name='test', mode='markers'),
        go.Scatter(x=X_train.squeeze(), y=y_train,
                   name='train', mode='markers'),
        go.Scatter(x=x_range, y=y_range,
                   name='prediction')
    ])
    if column_chosen2 == 'Age':
        fig.update_layout(title=model_chosen1, xaxis_title='Log10 Age', yaxis_title="Log10 HBV DNA Load")
    elif column_chosen2 == 'HBsAg':
        fig.update_layout(title=model_chosen1, xaxis_title='Log10 HBsAg', yaxis_title="Log10 HBV DNA Load")
    else:
        fig.update_layout(title=model_chosen1, xaxis_title='Log10 HBeAg', yaxis_title="Log10 HBV DNA Load")
    return fig

# In[8]:


@app.callback(
    Output('act_pred', 'figure'),
    [Input('act_pred_mod', 'value')]
)
def act_pred_plot(model_chosen):
    dff = data_set
    X = dff[['Age', 'HBsAg', 'HBeAg']]
    y = dff['HBV_DNA_Load']
    
    model = models[model_chosen]()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    fig = px.scatter(x=y, y=y_pred, labels={'x': 'ground truth', 'y': 'prediction'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max()
    )
    return fig


# In[9]:


@app.callback(
    Output('prederr', 'figure'),
    [Input('modtype', 'value')]
)
def pred_error(model_chosen2):
    dff = data_set
    train_idx, test_idx = train_test_split(dff.index, random_state=42)
    dff['split'] = 'train'
    dff.loc[test_idx, 'split'] = 'test'

    X = dff[['Age', 'HBsAg', 'HBeAg']]
    y = dff['HBV_DNA_Load']

    X_train = dff.loc[train_idx, ['Age', 'HBsAg', 'HBeAg']]
    y_train = dff.loc[train_idx, 'HBV_DNA_Load']

    model = models[model_chosen2]()
    model.fit(X_train, y_train)
    dff['prediction'] = model.predict(X)

    fig = px.scatter(
        dff, x='HBV_DNA_Load', y='prediction',
        marginal_x='histogram', marginal_y='histogram',
        color='split', trendline='ols'
    )
    fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max()
    )
    fig.update_layout(xaxis_title="Log10 HBV DNA Load")
    return fig


# In[10]:


@app.callback(
    Output('residuals', 'figure'),
    [Input('regmod', 'value'),
     Input('marg_type', 'value')]
)
def res_plot(model_chosen3, marg_chosen):
    dff = data_set
    train_idx, test_idx = train_test_split(dff.index, random_state=42)
    dff['split'] = 'train'
    dff.loc[test_idx, 'split'] = 'test'

    X = dff[['Age', 'HBsAg', 'HBeAg']]

    X_train = dff.loc[train_idx, ['Age', 'HBsAg', 'HBeAg']]
    y_train = dff.loc[train_idx, 'HBV_DNA_Load']

    model = models[model_chosen3]()
    model.fit(X_train, y_train)
    dff['prediction'] = model.predict(X)
    dff['residual'] = dff['prediction'] - dff['HBV_DNA_Load']

    fig = px.scatter(dff, x='prediction', y='residual', marginal_y=marg_chosen, color='split', trendline='ols')
    return fig


# In[11]:


@app.callback(
    Output('reg_coef', 'figure'),
    [Input('radio', 'value')]
)
def val_plot(model_chosen4):
    dff = data_set
    if model_chosen4 == 'Linear':
        X = dff[['Age', 'HBsAg', 'HBeAg']]
        y = dff['HBV_DNA_Load']
        model = linear_model.LinearRegression()
        model.fit(X, y)
        colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]
        fig = px.bar(x=X.columns, y=model.coef_, color=colors,
                     color_discrete_sequence=['red', 'blue'],
                     labels=dict(x='Feature', y='Linear coefficient'),
                     title='Weight of each feature for predicting HBV DNA Load')
        return fig
    else:
        train, test = train_test_split(dff, random_state=42)
        X_train = train[['Age', 'HBsAg', 'HBeAg']]
        y_train = train['HBV_DNA_Load']
        X_test = test[['Age', 'HBsAg', 'HBeAg']]
        y_test = test['HBV_DNA_Load']
        rmse_val = []
        for K in range(15):
            K = K+1
            model = neighbors.KNeighborsRegressor(n_neighbors=K)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            error = sqrt(mean_squared_error(y_test, pred))
            rmse_val.append(error)
            curve = pd.DataFrame(rmse_val)
            curve.rename(columns={0: 'RMSE'}, inplace=True)
        fig = px.line(y=curve['RMSE'])
        fig.update_layout(yaxis_title="RMSE", xaxis_title="k", title='k-neighbors = 5')
        return fig


# In[12]:


@app.callback(
    Output('grid_cv_hmap', 'figure'),
    [Input('rad_cv', 'value'),
     Input('cv_no', 'value')]
)
def grid_hplot(model_chosen, cv_chosen):
    n_fold = cv_chosen
    dff = data_set
    dff = dff.sample(frac=1, random_state=0)

    X = dff[['Age', 'HBsAg', 'HBeAg']]
    y = dff['HBV_DNA_Load']
    model = models[model_chosen]()
    grid_param = {
        'criterion': ['mse', 'friedman_mse', 'mae'],
        'max_depth': range(2, 5)
    }
    grid_cv = GridSearchCV(model, grid_param, cv=n_fold)
    grid_cv.fit(X, y)
    gridcv_df = pd.DataFrame(grid_cv.cv_results_)

    melted = (
        gridcv_df.rename(columns=lambda col: col.replace('param_', '')).melt(
            value_vars=[f'split{i}_test_score' for i in range(n_fold)],
            id_vars=['mean_test_score', 'mean_fit_time', 'criterion', 'max_depth'],
            var_name="cv_split",
            value_name="r_squared"
        )
    )

    melted['cv_split'] = (
        melted['cv_split'].str.replace('_test_score', '').str.replace('split', '')
    )
    fig_hmap = px.density_heatmap(
        melted, x="max_depth", y='criterion',
        histfunc="sum", z="r_squared",
        title='Grid search results on individual hyperparameter',
        hover_data=['mean_fit_time'],
        facet_col="cv_split", facet_col_wrap=3,
        labels={'mean_test_score': "mean_r_squared"}
    )
    return fig_hmap
    
    
    

@app.callback(
    Output('grid_cv_box', 'figure'),
    [Input('rad_cv', 'value'),
     Input('cv_no', 'value')]
)
def grid_boxplot(model_chosen, cv_chosen):
    n_fold = cv_chosen
    dff = data_set
    dff = dff.sample(frac=1, random_state=0)

    X = dff[['Age', 'HBsAg', 'HBeAg']]
    y = dff['HBV_DNA_Load']
    model = models[model_chosen]()
    grid_param = {
        'criterion': ['mse', 'friedman_mse', 'mae'],
        'max_depth': range(2, 5)
    }
    grid_cv = GridSearchCV(model, grid_param, cv=n_fold)
    grid_cv.fit(X, y)
    gridcv_df = pd.DataFrame(grid_cv.cv_results_)

    melted = (
        gridcv_df.rename(columns=lambda col: col.replace('param_', '')).melt(
            value_vars=[f'split{i}_test_score' for i in range(n_fold)],
            id_vars=['mean_test_score', 'mean_fit_time', 'criterion', 'max_depth'],
            var_name="cv_split",
            value_name="r_squared"
        )
    )

    melted['cv_split'] = (
        melted['cv_split'].str.replace('_test_score', '').str.replace('split', '')
    )
    fig_box = px.box(
        melted, x='max_depth', y='r_squared',
        title='Grid search results ',
        hover_data=['mean_fit_time'],
        points='all',
        color="criterion",
        hover_name='cv_split',
        labels={'mean_test_score': "mean_r_squared"}
    )
    return fig_box
    
    
    
@app.callback(
    Output('predout', 'children'),
    [Input('mod_pred', 'value'),
     Input('input1', 'value'),
     Input('input2', 'value'),
     Input('input3', 'value')]
)
def pred_dna(model_chosen, age_entered, hbsag_entered, hbeag_entered):
    dff = data_set
    
    dff = data_set
    
    X = dff[['Age', 'HBsAg', 'HBeAg']]
    y = dff['HBV_DNA_Load']
    
    age_entered = np.array(age_entered)
    hbsag_entered = np.array(hbsag_entered)
    hbeag_entered = np.array(hbeag_entered)
    
    log_age = math.log10(age_entered)
    log_hbsag = math.log10(hbsag_entered)
    log_hbeag = math.log10(hbeag_entered)
    
    x_new = [[log_age, log_hbsag, log_hbeag]]

    model = models[model_chosen]()
    model.fit(X,y)
    
    y_pred = model.predict(x_new)
    
    y_pred_org = np.exp(y_pred)
    
    return y_pred_org


# In[13]:


if __name__ == "__main__":
    app.run_server()


# In[ ]:





# In[ ]:




