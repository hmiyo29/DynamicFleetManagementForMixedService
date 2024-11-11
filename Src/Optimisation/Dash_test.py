from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd

df = pd.read_excel('C:/Users/heisu/OneDrive/ドキュメント/TU Delft/Thesis/repo/DynamicFleetManagementForMixedService-1/Src/Results_insertion/output_insertion_low_[2, 2]_0.xlsx', sheet_name='Sheet1')

app = Dash()

app.layout = [
    html.H1(children='Dynamic Pickup and Delivery Problem for Mixed Service Solved with Insertion Heuristic', style={'textAlign':'center'}), dash_table.DataTable(data=df.to_dict('records')), 
    # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    dcc.Graph(id='graph-content', figure=px.line(df, x='Time', y='Total_dist'))
]

# @callback(
#     Output('graph-content', 'figure'),
#     Input('dropdown-selection', 'value')
# )
# def update_graph(value):
#     # dff = df[df.country==value]
#     return px.line(df, x='Time', y='Total_dist')

if __name__ == '__main__':
    app.run(debug=True)
