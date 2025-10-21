"""
Interactive Dashboard for Typicality Analysis
Deploy on Railway with: python dashboard.py
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Load data
def load_data():
    # Try to find the merged data file (test set)
    data_files = list(Path('.').glob('merged_data_*_test.csv'))
    if not data_files:
        raise ValueError("No merged data file found! Looking for merged_data_*_test.csv")
        # Fall back to old naming
        #data_files = list(Path('.').glob('merged_data_*seed0.csv'))
    if not data_files:
        raise FileNotFoundError("No merged data file found! Looking for merged_data_*_test.csv")
    
    data_file = data_files[0]
    df = pd.read_csv(data_file)
    
    # Add G-V gap
    df['gv_gap'] = df['gen_score'] - df['disc_score']
    
    return df, str(data_file)

def load_cars_data():
    """Load the cars hypernym predictions data"""
    cars_file = Path('hypernym_predictions/cars_combined_clean_predictions_with_validator.csv')
    if not cars_file.exists():
        return None
    
    df_cars = pd.read_csv(cars_file)
    
    # Subsample for performance, but keep all test_completions
    np.random.seed(42)
    if 'source' in df_cars.columns:
        # Keep all test completions, subsample others
        df_test = df_cars[df_cars['source'] == 'test_completions']
        df_other = df_cars[df_cars['source'] != 'test_completions'].sample(frac=0.1, random_state=42)
        df_cars = pd.concat([df_test, df_other], ignore_index=True)
    else:
        # No source column, subsample all
        df_cars = df_cars.sample(frac=0.1, random_state=42)
    
    return df_cars

df, data_filename = load_data()
df_cars = load_cars_data()

# Precompute median thresholds for binary splits
median_noun2 = df['log_prob_noun2'].median()
median_wordfreq_noun2 = df['log_wordfreq_noun2'].median()

# Initialize the Dash app
app = dash.Dash(__name__, title='Typicality Analysis Dashboard')
server = app.server  # For Railway deployment

# Define colors
colors = {
    'background': '#ffffff',
    'text': '#2c3e50',
    'positive': '#3498db',  # blue
    'negative': '#e74c3c',  # red
    'grid': '#ecf0f1'
}

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'fontFamily': 'Arial, sans-serif'}, children=[
    # Header
    html.Div([
        html.H1('Typicality Analysis Dashboard',
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': 10}),
        html.P(f'Data: {data_filename} | {len(df)} examples | {(df["ground_truth"]==1).sum()} positive, {(df["ground_truth"]==0).sum()} negative',
               style={'textAlign': 'center', 'color': colors['text'], 'fontSize': 14}),
    ], style={'padding': '20px'}),
    
    # Typicality selector
    html.Div([
        html.Label('Color By:', 
                   style={'color': colors['text'], 'fontWeight': 'bold', 'marginRight': 10}),
        dcc.Dropdown(
            id='typicality-selector',
            options=[
                {'label': 'Ground Truth (Positive/Negative)', 'value': 'ground_truth'},
                {'label': 'P(noun2) - Unconditional Hypernym', 'value': 'log_prob_noun2'},
                {'label': 'P(noun1) - Unconditional Hyponym', 'value': 'log_prob_noun1'},
                {'label': 'P(noun2|context) - Conditional Hypernym', 'value': 'log_prob_noun2_given_context'},
                {'label': 'P(noun2) - Binary Split (High/Low)', 'value': 'log_prob_noun2_binary'},
                {'label': 'WordFreq(noun2) - Corpus Frequency', 'value': 'log_wordfreq_noun2'},
                {'label': 'WordFreq(noun2) - Binary Split (High/Low)', 'value': 'log_wordfreq_noun2_binary'}
            ],
            value='ground_truth',
            style={'width': '500px', 'display': 'inline-block'}
        ),
    ], style={'padding': '0 40px 20px 40px'}),
    
    # Main content - MAIN PLOT
    html.Div([
        dcc.Graph(id='main-scatter', style={'height': '80vh'}),
    ], style={'padding': '0 40px 40px 40px'}),
    
    # Cars hypernym predictions plot
    html.Div([
        html.H2('Cars Hypernym Predictions: Generator vs Validator',
                style={'textAlign': 'center', 'color': colors['text'], 'marginTop': 20}),
        html.P(f'Sampled predictions for "cars" | {len(df_cars) if df_cars is not None else 0} predictions (1/10 subsample)',
               style={'textAlign': 'center', 'color': colors['text'], 'fontSize': 14}),
        dcc.Graph(id='cars-scatter', style={'height': '80vh'}),
    ], style={'padding': '0 40px 40px 40px'}) if df_cars is not None else html.Div(),
    
    # # COMMENTED OUT - Extra clutter
    # html.Div([
    #     # Left panel - Main scatter plot
    #     html.Div([
    #         html.H3('Generator vs Validator', style={'color': colors['text']}),
    #         html.P('Hover over points to see (noun1, noun2) pairs', 
    #                style={'color': colors['text'], 'fontSize': 12, 'fontStyle': 'italic'}),
    #         dcc.Graph(id='main-scatter', style={'height': '600px'}),
    #     ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    #     
    #     # Right panel - Typicality comparison
    #     html.Div([
    #         html.H3('Typicality Scores', style={'color': colors['text']}),
    #         dcc.Dropdown(
    #             id='typicality-dropdown',
    #             options=[
    #                 {'label': 'P(noun2) - Unconditional Hypernym', 'value': 'log_prob_noun2'},
    #                 {'label': 'P(noun1) - Unconditional Hyponym', 'value': 'log_prob_noun1'},
    #                 {'label': 'P(noun2|context) - Conditional Hypernym', 'value': 'log_prob_noun2_given_context'}
    #             ],
    #             value='log_prob_noun2_given_context',
    #             style={'marginBottom': '20px'}
    #         ),
    #         dcc.Graph(id='typicality-scatter', style={'height': '600px'}),
    #     ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    # ]),
    # 
    # # Bottom panel - G-V Gap analysis
    # html.Div([
    #     html.H3('Generator-Validator Gap Analysis', style={'color': colors['text'], 'textAlign': 'center'}),
    #     
    #     # Statistics cards
    #     html.Div([
    #         # Overall stats
    #         html.Div([
    #             html.H4('Overall', style={'color': colors['text'], 'marginBottom': 10}),
    #             html.P(f"Mean G-V Gap: {df['gv_gap'].mean():.3f}", style={'fontSize': 16}),
    #             html.P(f"Std: {df['gv_gap'].std():.3f}", style={'fontSize': 14}),
    #         ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center', 
    #                   'padding': '20px', 'border': f'2px solid {colors["grid"]}', 'margin': '10px'}),
    #         
    #         # Positive examples
    #         html.Div([
    #             html.H4('Positive Examples', style={'color': colors['positive'], 'marginBottom': 10}),
    #             html.P(f"Mean G-V Gap: {df[df['ground_truth']==1]['gv_gap'].mean():.3f}", style={'fontSize': 16}),
    #             html.P(f"n = {(df['ground_truth']==1).sum()}", style={'fontSize': 14}),
    #         ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center',
    #                   'padding': '20px', 'border': f'2px solid {colors["positive"]}', 'margin': '10px'}),
    #         
    #         # Negative examples
    #         html.Div([
    #             html.H4('Negative Examples', style={'color': colors['negative'], 'marginBottom': 10}),
    #             html.P(f"Mean G-V Gap: {df[df['ground_truth']==0]['gv_gap'].mean():.3f}", style={'fontSize': 16}),
    #             html.P(f"n = {(df['ground_truth']==0).sum()}", style={'fontSize': 14}),
    #         ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center',
    #                   'padding': '20px', 'border': f'2px solid {colors["negative"]}', 'margin': '10px'}),
    #     ], style={'textAlign': 'center', 'marginBottom': 30}),
    #     
    #     # G-V Gap plot
    #     dcc.Graph(id='gv-gap-plot', style={'height': '400px'}),
    #     
    # ], style={'padding': '20px'}),
    # 
    # # Footer with model performance
    # html.Div([
    #     html.H3('Model Performance', style={'color': colors['text'], 'textAlign': 'center'}),
    #     html.Div(id='model-stats', style={'textAlign': 'center', 'fontSize': 14}),
    # ], style={'padding': '20px', 'backgroundColor': colors['grid']}),
])


# Callbacks
@app.callback(
    Output('main-scatter', 'figure'),
    Input('typicality-selector', 'value')
)
def update_main_scatter(color_var):
    """Create main scatter plot: Generator (x) vs Validator (y)"""
    
    fig = go.Figure()
    
    # Prepare custom data with all typicality measures and ground truth
    customdata = df[['noun1', 'noun2', 'index', 'ground_truth',
                     'log_prob_noun2', 'log_prob_noun1', 
                     'log_prob_noun2_given_context']].values
    
    if color_var == 'ground_truth':
        # Mode 1: Color by ground truth (red/blue for neg/pos)
        for gt, color, name in [(1, colors['positive'], 'Positive (Hypernym)'), 
                                (0, colors['negative'], 'Negative (Non-hypernym)')]:
            df_subset = df[df['ground_truth'] == gt]
            subset_indices = df_subset.index
            
            fig.add_trace(go.Scatter(
                x=df_subset['gen_score'],
                y=df_subset['disc_score'],
                mode='markers',
                name=name,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                customdata=customdata[subset_indices],
                hovertemplate='<b>%{customdata[0]} → %{customdata[1]}</b><br>' +
                              'Generator: %{x:.3f}<br>' +
                              'Validator: %{y:.3f}<br>' +
                              'Ground Truth: %{customdata[3]}<br>' +
                              'P(noun2): %{customdata[4]:.3f}<br>' +
                              'P(noun1): %{customdata[5]:.3f}<br>' +
                              'P(noun2|context): %{customdata[6]:.3f}<br>' +
                              'Index: %{customdata[2]}<extra></extra>'
            ))
    elif color_var in ['log_prob_noun2_binary', 'log_wordfreq_noun2_binary']:
        # Mode 3/4: Binary split with best fit lines
        # Split data based on median
        if color_var == 'log_prob_noun2_binary':
            df['high_typicality'] = df['log_prob_noun2'] >= median_noun2
            threshold_val = median_noun2
            label = 'P(noun2)'
        else:  # log_wordfreq_noun2_binary
            df['high_typicality'] = df['log_wordfreq_noun2'] >= median_wordfreq_noun2
            threshold_val = median_wordfreq_noun2
            label = 'WordFreq(noun2)'
        
        # Define colors
        high_color = '#00FF00'  # Lime green
        low_color = '#800080'   # Purple
        
        # Plot each group separately
        for is_high, color, name in [(True, high_color, f'High {label} (≥{threshold_val:.2f})'), 
                                      (False, low_color, f'Low {label} (<{threshold_val:.2f})')]:
            df_subset = df[df['high_typicality'] == is_high]
            subset_indices = df_subset.index
            
            # Compute Pearson correlation
            corr, p_value = stats.pearsonr(df_subset['gen_score'], df_subset['disc_score'])
            
            # Compute best fit line
            slope, intercept, r_value, p_val_reg, std_err = stats.linregress(
                df_subset['gen_score'], df_subset['disc_score']
            )
            
            # Plot points
            fig.add_trace(go.Scatter(
                x=df_subset['gen_score'],
                y=df_subset['disc_score'],
                mode='markers',
                name=f'{name}<br>r={corr:.3f}',
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                customdata=customdata[subset_indices],
                hovertemplate='<b>%{customdata[0]} → %{customdata[1]}</b><br>' +
                              'Generator: %{x:.3f}<br>' +
                              'Validator: %{y:.3f}<br>' +
                              'Ground Truth: %{customdata[3]}<br>' +
                              'P(noun2): %{customdata[4]:.3f}<br>' +
                              'P(noun1): %{customdata[5]:.3f}<br>' +
                              'P(noun2|context): %{customdata[6]:.3f}<br>' +
                              'Index: %{customdata[2]}<extra></extra>'
            ))
            
            # Add best fit line
            x_fit = np.array([df_subset['gen_score'].min(), df_subset['gen_score'].max()])
            y_fit = slope * x_fit + intercept
            
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name=f'{name} fit',
                line=dict(color=color, width=3, dash='solid'),
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        # Mode 2: Color by typicality (continuous gradient)
        typicality_labels = {
            'log_prob_noun2': 'P(noun2)',
            'log_prob_noun1': 'P(noun1)',
            'log_prob_noun2_given_context': 'P(noun2|context)',
            'log_wordfreq_noun2': 'WordFreq(noun2)'
        }
        
        fig.add_trace(go.Scatter(
            x=df['gen_score'],
            y=df['disc_score'],
            mode='markers',
            name='All points',
            marker=dict(
                color=df[color_var],
                colorscale='Viridis',  # Good gradient: purple (low) to yellow (high)
                size=8,
                opacity=0.7,
                line=dict(width=0.5, color='white'),
                colorbar=dict(
                    title=typicality_labels[color_var],
                    titleside='right',
                    tickmode='linear',
                    tick0=df[color_var].min(),
                    dtick=(df[color_var].max() - df[color_var].min()) / 5
                ),
                showscale=True
            ),
            customdata=customdata,
            hovertemplate='<b>%{customdata[0]} → %{customdata[1]}</b><br>' +
                          'Generator: %{x:.3f}<br>' +
                          'Validator: %{y:.3f}<br>' +
                          'Ground Truth: %{customdata[3]}<br>' +
                          'P(noun2): %{customdata[4]:.3f}<br>' +
                          'P(noun1): %{customdata[5]:.3f}<br>' +
                          'P(noun2|context): %{customdata[6]:.3f}<br>' +
                          'Index: %{customdata[2]}<extra></extra>'
        ))
    
    # Add horizontal line at log(0.5) threshold
    threshold = np.log(0.5)  # Natural log, ≈ -0.693
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation_text=f"Threshold: ln(0.5) = {threshold:.3f}",
        annotation_position="right"
    )
    
    # Set axis ranges with padding - SEPARATE for each axis
    x_range = [df['gen_score'].min() - 1, df['gen_score'].max() + 1]
    y_range = [df['disc_score'].min() - .2, df['disc_score'].max() + .2]
    
    fig.update_layout(
        xaxis_title='Generator Score',
        yaxis_title='Validator Score (Discriminator)',
        hovermode='closest',
        plot_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=50, r=50, t=30, b=50),
        height=800,
        width=1200,
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'], range=x_range)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'], range=y_range)
    
    return fig


@app.callback(
    Output('cars-scatter', 'figure'),
    Input('cars-scatter', 'id')  # Dummy input, no interaction needed
)
def update_cars_scatter(_):
    """Create scatter plot for cars hypernym predictions: Generator (x) vs Validator (y)"""
    
    fig = go.Figure()
    
    if df_cars is None or len(df_cars) == 0:
        # Return empty figure if no data
        fig.update_layout(
            annotations=[{
                'text': 'No cars prediction data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        return fig
    
    # Split by source if available
    if 'source' in df_cars.columns:
        # Test completions in red
        df_test = df_cars[df_cars['source'] == 'test_completions']
        if len(df_test) > 0:
            customdata_test = df_test[['noun1', 'predicted_hypernym', 'log_prob', 'validator_log_prob']].values
            fig.add_trace(go.Scatter(
                x=df_test['log_prob'],
                y=df_test['validator_log_prob'],
                mode='markers',
                name='Test Completions (Hand-crafted)',
                marker=dict(
                    color='red',
                    size=8,
                    opacity=0.9,
                    line=dict(width=1, color='darkred')
                ),
                customdata=customdata_test,
                hovertemplate='<b>%{customdata[0]} → %{customdata[1]}</b><br>' +
                              'Generator Score: %{customdata[2]:.3f}<br>' +
                              'Validator Score: %{customdata[3]:.3f}<br>' +
                              'Source: Test Completions<extra></extra>'
            ))
        
        # All other predictions in blue
        df_other = df_cars[df_cars['source'] != 'test_completions']
        if len(df_other) > 0:
            customdata_other = df_other[['noun1', 'predicted_hypernym', 'log_prob', 'validator_log_prob']].values
            fig.add_trace(go.Scatter(
                x=df_other['log_prob'],
                y=df_other['validator_log_prob'],
                mode='markers',
                name='Model Predictions',
                marker=dict(
                    color='blue',
                    size=3,
                    opacity=0.6,
                    line=dict(width=0)
                ),
                customdata=customdata_other,
                hovertemplate='<b>%{customdata[0]} → %{customdata[1]}</b><br>' +
                              'Generator Score: %{customdata[2]:.3f}<br>' +
                              'Validator Score: %{customdata[3]:.3f}<br>' +
                              '<extra></extra>'
            ))
    else:
        # No source column, plot all in blue
        customdata = df_cars[['noun1', 'predicted_hypernym', 'log_prob', 'validator_log_prob']].values
        fig.add_trace(go.Scatter(
            x=df_cars['log_prob'],
            y=df_cars['validator_log_prob'],
            mode='markers',
            name='Predictions',
            marker=dict(
                color='blue',
                size=3,
                opacity=0.8,
                line=dict(width=0)
            ),
            customdata=customdata,
            hovertemplate='<b>%{customdata[0]} → %{customdata[1]}</b><br>' +
                          'Generator Score: %{customdata[2]:.3f}<br>' +
                          'Validator Score: %{customdata[3]:.3f}<br>' +
                          '<extra></extra>'
        ))
    
    fig.update_layout(
        xaxis_title='Generator Score (log_prob)',
        yaxis_title='Validator Score (validator_log_prob)',
        hovermode='closest',
        plot_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=50, r=50, t=30, b=50),
        height=800,
        width=1200,
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'])
    
    return fig


# # COMMENTED OUT - Unused callbacks
# pass


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)

