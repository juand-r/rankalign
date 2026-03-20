"""
Training score trajectory dashboard.
Shows gen_score and val_score over training steps as line plots.
Each noun2 is a separate line.
"""

import os
import glob
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objects as go
import argparse

# Global data - keyed by task
ALL_TASKS_DATA = {}  # task -> {trajectory_data, all_noun2s, steps}
AVAILABLE_TASKS = []
LOG_DIR = None
LABEL_LOOKUPS = {}  # task -> label_lookup


def discover_available_tasks(log_dir):
    """Find all available hypernym tasks in the log directory."""
    pattern = os.path.join(log_dir, "*hypernym-*-step0.csv")
    files = glob.glob(pattern)
    
    tasks = set()
    for f in files:
        basename = os.path.basename(f)
        # Extract task name from pattern like "...-hypernym-bananas-all-..."
        import re
        match = re.search(r'hypernym-([a-zA-Z]+)-', basename)
        if match:
            tasks.add(match.group(1))
    
    return sorted(tasks)


def load_training_logs(log_dir, task_pattern):
    """Load all step CSVs for a given task pattern."""
    pattern = os.path.join(log_dir, f"*{task_pattern}*-step*.csv")
    files = glob.glob(pattern)
    files = [f for f in files if '-pair.csv' not in f]
    
    step_files = []
    for f in files:
        basename = os.path.basename(f)
        step_part = basename.split('-step')[-1].replace('.csv', '')
        try:
            step = int(step_part)
            step_files.append((step, f))
        except ValueError:
            continue
    
    step_files.sort(key=lambda x: x[0])
    return step_files


def load_ground_truth(task):
    """Load ground truth labels."""
    data_path = f"/datastor1/jdr/gv-gap/rankalign/data/fixed-hypernyms/hypernym_{task}_google-gemma-2-2b_train-fixed.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        label_lookup = {}
        for _, row in df.iterrows():
            noun2 = row.get('fixed_hypernym_generator', row.get('noun2', ''))
            gt = row.get('gpt4_ground_truth', '')
            label_lookup[(row['noun1'], noun2)] = 1 if gt == 'Yes' else 0
        return label_lookup
    return None


def load_trajectories_for_task(task, log_dir, label_lookup):
    """Build trajectory data for a specific task."""
    step_files = load_training_logs(log_dir, f'hypernym-{task}')
    if not step_files:
        return None
    
    # First pass: collect all data
    all_data = []
    for step, filepath in step_files:
        df = pd.read_csv(filepath)
        df['step'] = step
        df['label'] = df.apply(
            lambda row: label_lookup.get((row['noun1'], row['noun2']), -1),
            axis=1
        )
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    steps = sorted(combined['step'].unique())
    # Filter out NaN values and convert to string for sorting
    all_noun2s = sorted([str(n) for n in combined['noun2'].unique() if pd.notna(n)])
    
    # Build trajectories for each noun2
    trajectory_data = {}
    for noun2 in all_noun2s:
        noun_data = combined[combined['noun2'] == noun2].sort_values('step')
        label = noun_data['label'].iloc[0] if len(noun_data) > 0 else -1
        trajectory_data[noun2] = {
            'steps': noun_data['step'].values,
            'gen_scores': noun_data['gen_score'].values,
            'val_scores': noun_data['val_score'].values,
            'label': label
        }
    
    # Compute axis limits
    all_gen = combined['gen_score'].values
    all_val = combined['val_score'].values
    gen_min, gen_max = all_gen.min(), all_gen.max()
    val_min, val_max = all_val.min(), all_val.max()
    gen_pad = (gen_max - gen_min) * 0.1
    val_pad = (val_max - val_min) * 0.1
    
    return {
        'trajectory_data': trajectory_data,
        'all_noun2s': all_noun2s,
        'steps': steps,
        'axis_limits': {
            'gen_min': gen_min - gen_pad, 'gen_max': gen_max + gen_pad,
            'val_min': val_min - val_pad, 'val_max': val_max + val_pad,
            'step_min': min(steps), 'step_max': max(steps)
        }
    }


def create_app(initial_task):
    """Create the Dash app."""
    global ALL_TASKS_DATA, AVAILABLE_TASKS
    
    app = Dash(__name__)
    
    # Get initial task data
    initial_data = ALL_TASKS_DATA.get(initial_task, {})
    initial_noun2s = initial_data.get('all_noun2s', [])
    initial_trajectory = initial_data.get('trajectory_data', {})
    
    app.layout = html.Div([
        html.H1("Training Score Trajectories", 
                style={'textAlign': 'center', 'marginBottom': '10px'}),
        
        # Task selector at top
        html.Div([
            html.Label("Select Task: ", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='task-selector',
                options=[{'label': f'hypernym-{t}', 'value': t} for t in AVAILABLE_TASKS],
                value=initial_task,
                style={'width': '300px', 'display': 'inline-block'},
                clearable=False
            ),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        html.Div([
            # Left panel - controls
            html.Div([
                html.H3("Filter Points (noun2)"),
                html.Div([
                    html.Button("Select All", id='select-all-btn', n_clicks=0,
                               style={'marginRight': '10px', 'marginBottom': '10px'}),
                    html.Button("Select None", id='select-none-btn', n_clicks=0,
                               style={'marginRight': '10px', 'marginBottom': '10px'}),
                    html.Button("Positive Only", id='select-pos-btn', n_clicks=0,
                               style={'marginRight': '10px', 'marginBottom': '10px'}),
                    html.Button("Negative Only", id='select-neg-btn', n_clicks=0,
                               style={'marginBottom': '10px'}),
                ]),
                html.Div([
                    dcc.Checklist(
                        id='noun-selector',
                        options=[
                            {'label': f' {n} {"✓" if initial_trajectory.get(n, {}).get("label", -1)==1 else "✗"}', 'value': n} 
                            for n in initial_noun2s
                        ],
                        value=initial_noun2s[:10],  # Start with first 10 selected
                        labelStyle={'display': 'block', 'padding': '2px 0', 'cursor': 'pointer'},
                        inputStyle={'marginRight': '8px'}
                    ),
                ], style={'maxHeight': '600px', 'overflowY': 'auto', 'border': '1px solid #ddd', 
                          'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#fafafa'}),
                
            ], style={'width': '20%', 'padding': '20px', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Right panel - plots
            html.Div([
                dcc.Graph(id='gen-plot', style={'height': '45vh'}),
                dcc.Graph(id='val-plot', style={'height': '45vh'}),
            ], style={'width': '78%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
        ], style={'display': 'flex'}),
    ], style={'fontFamily': 'Arial, sans-serif'})
    
    # Callback to update noun selector when task changes
    @app.callback(
        Output('noun-selector', 'options'),
        Output('noun-selector', 'value'),
        Input('task-selector', 'value'),
        Input('select-all-btn', 'n_clicks'),
        Input('select-none-btn', 'n_clicks'),
        Input('select-pos-btn', 'n_clicks'),
        Input('select-neg-btn', 'n_clicks'),
    )
    def update_noun_selector(task, all_clicks, none_clicks, pos_clicks, neg_clicks):
        from dash import ctx
        
        task_data = ALL_TASKS_DATA.get(task, {})
        trajectory_data = task_data.get('trajectory_data', {})
        all_noun2s = task_data.get('all_noun2s', [])
        
        options = [
            {'label': f' {n} {"✓" if trajectory_data.get(n, {}).get("label", -1)==1 else "✗"}', 'value': n} 
            for n in all_noun2s
        ]
        
        triggered = ctx.triggered_id
        if triggered == 'task-selector':
            # Task changed - select first 10
            return options, all_noun2s[:10]
        elif triggered == 'select-all-btn':
            return options, all_noun2s
        elif triggered == 'select-none-btn':
            return options, []
        elif triggered == 'select-pos-btn':
            return options, [n for n in all_noun2s if trajectory_data.get(n, {}).get('label') == 1]
        elif triggered == 'select-neg-btn':
            return options, [n for n in all_noun2s if trajectory_data.get(n, {}).get('label') == 0]
        
        return options, all_noun2s[:10]
    
    @app.callback(
        Output('gen-plot', 'figure'),
        Output('val-plot', 'figure'),
        Input('task-selector', 'value'),
        Input('noun-selector', 'value'),
    )
    def update_plots(task, selected_nouns):
        task_data = ALL_TASKS_DATA.get(task, {})
        trajectory_data = task_data.get('trajectory_data', {})
        axis_limits = task_data.get('axis_limits', {})
        
        # Generator score plot
        gen_fig = go.Figure()
        val_fig = go.Figure()
        
        if not selected_nouns or not trajectory_data:
            gen_fig.update_layout(title=f'Generator Score - hypernym-{task}')
            val_fig.update_layout(title=f'Validator Score - hypernym-{task}')
            return gen_fig, val_fig
        
        for noun2 in selected_nouns:
            data = trajectory_data.get(noun2)
            if data is None:
                continue
            
            color = 'green' if data['label'] == 1 else 'red' if data['label'] == 0 else 'gray'
            
            gen_fig.add_trace(go.Scattergl(
                x=data['steps'], y=data['gen_scores'],
                mode='lines', name=noun2,
                line=dict(color=color, width=1),
                opacity=0.7,
                hovertemplate=f'{noun2}<br>Step: %{{x}}<br>Gen: %{{y:.2f}}<extra></extra>'
            ))
            
            val_fig.add_trace(go.Scattergl(
                x=data['steps'], y=data['val_scores'],
                mode='lines', name=noun2,
                line=dict(color=color, width=1),
                opacity=0.7,
                hovertemplate=f'{noun2}<br>Step: %{{x}}<br>Val: %{{y:.2f}}<extra></extra>'
            ))
        
        gen_fig.update_layout(
            title=f'Generator Score - hypernym-{task}',
            xaxis_title='Training Step',
            yaxis_title='Generator Score (log-prob)',
            xaxis=dict(range=[axis_limits.get('step_min', 0), axis_limits.get('step_max', 100)]),
            yaxis=dict(range=[axis_limits.get('gen_min', -15), axis_limits.get('gen_max', 0)]),
            showlegend=False,
            hovermode='closest'
        )
        
        val_fig.update_layout(
            title=f'Validator Score - hypernym-{task}',
            xaxis_title='Training Step',
            yaxis_title='Validator Score (log-odds)',
            xaxis=dict(range=[axis_limits.get('step_min', 0), axis_limits.get('step_max', 100)]),
            yaxis=dict(range=[axis_limits.get('val_min', -2), axis_limits.get('val_max', 2)]),
            showlegend=False,
            hovermode='closest'
        )
        
        # Add threshold line on val plot
        val_fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        return gen_fig, val_fig
    
    return app


def main():
    global ALL_TASKS_DATA, AVAILABLE_TASKS, LOG_DIR, LABEL_LOOKUPS
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='bananas', help='Initial task to display')
    parser.add_argument('--log-dir', type=str, default='/datastor1/jdr/gv-gap/rankalign/outputs/training-logs')
    parser.add_argument('--port', type=int, default=8889)
    args = parser.parse_args()
    
    LOG_DIR = args.log_dir
    
    # Discover available tasks
    print("Discovering available tasks...")
    AVAILABLE_TASKS = discover_available_tasks(args.log_dir)
    print(f"Found {len(AVAILABLE_TASKS)} tasks: {AVAILABLE_TASKS}")
    
    if not AVAILABLE_TASKS:
        print("No tasks found!")
        return
    
    # Load ground truth for all tasks
    print("Loading ground truth labels...")
    for task in AVAILABLE_TASKS:
        label_lookup = load_ground_truth(task)
        if label_lookup:
            LABEL_LOOKUPS[task] = label_lookup
            print(f"  {task}: {len(label_lookup)} labels")
    
    # Pre-load data for all tasks
    print("Loading trajectory data for all tasks...")
    for task in AVAILABLE_TASKS:
        label_lookup = LABEL_LOOKUPS.get(task, {})
        task_data = load_trajectories_for_task(task, args.log_dir, label_lookup)
        if task_data:
            ALL_TASKS_DATA[task] = task_data
            print(f"  {task}: {len(task_data['all_noun2s'])} noun2s, {len(task_data['steps'])} steps")
    
    # Use initial task or first available
    initial_task = args.task if args.task in AVAILABLE_TASKS else AVAILABLE_TASKS[0]
    
    app = create_app(initial_task)
    print(f"\n🚀 Dashboard on http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
