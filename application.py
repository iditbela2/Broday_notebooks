import plotly as py
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import scipy.stats


########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server
app.title='Normal approximation to Binomial distribution'
  
########### Set up the layout 
app.layout = html.Div([

    dcc.Graph(id='Normal approximation'),
    
    dcc.Textarea(
        id='n - Bernouli trials',
        value='Choose n (number of Bernouli trials)',
        style={'width': '100%', 'height': 10},
    ),

    dcc.Slider(
        id='n',
        min=0,
        max=100,
        value=10,
        marks={i: str(i) for i in range(0,110,10)},
        step=1
    ),
    
    dcc.Textarea(
    id='p - probabiliy of success',
    value='Choose p (probabiliy of success)',
    style={'width': '100%', 'height': 10},
    ),
    
    dcc.Slider(
        id='p',
        min=0,
        max=1,
        value=0.5,
        marks={i: str(np.round(i,2)) for i in np.arange(0,1.05,0.05)},
        step=0.01
    )
])

@app.callback(
    Output('Normal approximation', 'figure'),
    [Input('n', 'value'),
     Input('p', 'value')
     ])

def update_graph(n,p):
    
    data = []
    
    chosen_data = np.arange(0,2*n*p+10)
    # Create traces
    the_plot = go.Scatter(

        x=chosen_data,
        y=scipy.stats.binom.pmf(chosen_data, n, p),
        mode = 'lines',#lines/markers
#         name = 'n={},p={}'.format(n,p),
        opacity=.38,
        line = dict(width=5,color='LightSeaGreen')
    ) 
           
    layout = go.Layout(           
        title={
            'text': "Normal approximation to Binomial distribution<br> n*p={}, n*(1-p)={}".format(np.round(n*p,1),np.round(n*(1-p),1)),
            'y':0.9,
            'x':0.6,
            'xanchor': 'right',
            'yanchor': 'top'},
        yaxis = dict(
            title = 'PDF'
        ),
        xaxis = dict(
            title = '\nk successes out of n trials\n'
        ),
        font=dict(
        size=14
        ),
        plot_bgcolor='rgb(255,255,250)'
    )
            
            
    data.append(the_plot)

    return go.Figure(
        data=data,
        layout=layout)

########### Run the app
if __name__ == '__main__':
    application.run(debug=True, port=8080)




