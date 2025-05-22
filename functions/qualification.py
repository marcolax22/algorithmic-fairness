# -------------------------------------------------------------------------------
# import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import sys
    
# -------------------------------------------------------------------------------
# function for visualization of the model
def visualize_qualification_scores(
    qual_male_first_stage_df,
    qual_female_first_stage_df,
    qual_male_second_stage_df,
    qual_female_second_stage_df,
    fig_size=(12, 6),
    palette='Set2',
    show_summary=True,
    show_debug=False
):
    
    # Create DataFrame for visualization
    data = []

    # Add first stage data using index [0] - handle Series properly
    try:
        # If it's a pandas Series, convert to list
        if isinstance(qual_male_first_stage_df[0], pd.Series):
            male_first = qual_male_first_stage_df[0].tolist()
        else:
            male_first = qual_male_first_stage_df[0]
            
        if isinstance(qual_female_first_stage_df[0], pd.Series):
            female_first = qual_female_first_stage_df[0].tolist()
        else:
            female_first = qual_female_first_stage_df[0]
        
        for score in male_first:
            data.append({'Stage': 'Algorithmic Stage', 'Score': score, 'Gender': 'Male'})
        for score in female_first:
            data.append({'Stage': 'Algorithmic Stage', 'Score': score, 'Gender': 'Female'})
    except (IndexError, TypeError) as e:
        if show_debug:
            print(f"Error with first stage data: {e}")
            print(f"Type of qual_male_first_stage_df: {type(qual_male_first_stage_df)}")
            if len(qual_male_first_stage_df) > 0:
                print(f"Type of first element: {type(qual_male_first_stage_df[0])}")

    # Add second stage A data (index 0)
    try:
        # If it's a pandas Series, convert to list
        if isinstance(qual_male_second_stage_df[0], pd.Series):
            male_second_a = qual_male_second_stage_df[0].tolist()
        else:
            male_second_a = qual_male_second_stage_df[0]
            
        if isinstance(qual_female_second_stage_df[0], pd.Series):
            female_second_a = qual_female_second_stage_df[0].tolist()
        else:
            female_second_a = qual_female_second_stage_df[0]
        
        for score in male_second_a:
            data.append({'Stage': 'Human Decision\n(No Discrimination)', 'Score': score, 'Gender': 'Male'})
        for score in female_second_a:
            data.append({'Stage': 'Human Decision\n(No Discrimination)', 'Score': score, 'Gender': 'Female'})
    except (IndexError, TypeError) as e:
        if show_debug:
            print(f"Error with Second Stage A data: {e}")

    # Add second stage B data (index 1)
    try:
        # If it's a pandas Series, convert to list
        if isinstance(qual_male_second_stage_df[1], pd.Series):
            male_second_b = qual_male_second_stage_df[1].tolist()
        else:
            male_second_b = qual_male_second_stage_df[1]
            
        if isinstance(qual_female_second_stage_df[1], pd.Series):
            female_second_b = qual_female_second_stage_df[1].tolist()
        else:
            female_second_b = qual_female_second_stage_df[1]
        
        for score in male_second_b:
            data.append({'Stage': 'Human Decision\n(Discrimination)', 'Score': score, 'Gender': 'Male'})
        for score in female_second_b:
            data.append({'Stage': 'Human Decision\n(Discrimination)', 'Score': score, 'Gender': 'Female'})
    except (IndexError, TypeError) as e:
        if show_debug:
            print(f"Error with Second Stage B data: {e}")

    # Add second stage C data (index 2)
    try:
        # If it's a pandas Series, convert to list
        if isinstance(qual_male_second_stage_df[2], pd.Series):
            male_second_c = qual_male_second_stage_df[2].tolist()
        else:
            male_second_c = qual_male_second_stage_df[2]
            
        if isinstance(qual_female_second_stage_df[2], pd.Series):
            female_second_c = qual_female_second_stage_df[2].tolist()
        else:
            female_second_c = qual_female_second_stage_df[2]
        
        for score in male_second_c:
            data.append({'Stage': 'Human Decision\n(High Discrimination)', 'Score': score, 'Gender': 'Male'})
        for score in female_second_c:
            data.append({'Stage': 'Human Decision\n(High Discrimination)', 'Score': score, 'Gender': 'Female'})
    except (IndexError, TypeError) as e:
        if show_debug:
            print(f"Error with Second Stage C data: {e}")

    # Convert to DataFrame and ensure Score is numeric
    df_gender = pd.DataFrame(data)
    df_gender['Score'] = pd.to_numeric(df_gender['Score'], errors='coerce')

    # Print data info to debug
    if show_debug:
        print(f"DataFrame contains {len(df_gender)} rows")
        print(f"Data types: {df_gender.dtypes}")
        print(f"Any NaN values: {df_gender.isna().any().any()}")

    # Create the plot
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=fig_size)
    ax = sns.boxplot(x='Stage', y='Score', hue='Gender', data=df_gender, palette=palette)
    
    # set title
    plt.title("")
    plt.ylabel("Qualification Score")
    plt.xlabel("Selection Stage")
    plt.tight_layout()

    # Get the figure for returning
    fig = plt.gcf()

    # Print summary statistics
    if show_summary:
        print("\nSummary Statistics:")
        summary = df_gender.groupby(['Stage', 'Gender'])['Score'].agg(['mean', 'std', 'median']).round(4)
        print(summary)
    
    return fig, df_gender

# -------------------------------------------------------------------------------
# function for all visualization of the model
def visualize_qualification_grid(
    model_list=['rf_1', 'rf_2', 'rf_3', 'rf_4', 'lm_1', 'lm_2', 'lm_3', 'lm_4'],
    variable_dict=None,
    fig_size=(20, 25),
    palette='Set2',
    show_summary=True,
    show_debug=False,
    visualize_function=None
):
    """
    Create a grid of qualification score visualizations using an existing 
    visualization function for each individual plot.
    """
    
    # Validate variable_dict
    if variable_dict is None:
        raise ValueError("You must provide variable_dict (either locals() or a custom dictionary)")
    
    # Create figure with GridSpec for more control
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(4, 2, figure=fig)
    
    # Define model order and display names 
    model_display_names = [model.upper() for model in model_list]
    
    # Process each model
    for i, model_id in enumerate(model_list):
        # Get the display name
        model_name = model_display_names[i]

        model_title = ['Logistic Regression',
                       'Logistic Regression (Statistical Parity)',
                       'Logistic Regression (Equalized Odds)',
                       'Logistic Regression (Equal Opportunity)',
                       'Random Forest',
                       'Random Forest (Statistical Parity)',
                       'Random Forest (Equalized Odds)',
                       'Random Forest (Equal Opportunity)']
        
        # Construct variable names based on model_id
        male_first_var = f"qual_male_first_stage_df_{model_id}"
        female_first_var = f"qual_female_first_stage_df_{model_id}"
        male_second_var = f"qual_male_second_stage_df_{model_id}"
        female_second_var = f"qual_female_second_stage_df_{model_id}"
        
        # Get the variables either directly from variable_dict if it's a dict containing model IDs
        # or look for the specific variable names in the locals dict
        if model_id in variable_dict:
            # If variable_dict contains the model data directly
            male_first, female_first, male_second, female_second = variable_dict[model_id]
        else:
            # Check if these variables exist in variable_dict (locals())
            if not all(var in variable_dict for var in [male_first_var, female_first_var, male_second_var, female_second_var]):
                if show_debug:
                    print(f"Missing data for {model_name}, skipping...")
                    print(f"Looking for: {male_first_var}, {female_first_var}, {male_second_var}, {female_second_var}")
                continue
                
            # Extract the variables
            male_first = variable_dict[male_first_var]
            female_first = variable_dict[female_first_var]
            male_second = variable_dict[male_second_var]
            female_second = variable_dict[female_second_var]
        
        # Calculate row and column for subplot
        row = i % 4  
        col = i // 4  
        
        # Call the original visualization function to get the figure and dataframe
        # Use original_stdout to capture print statements from the original function
        import io
        original_stdout = sys.stdout
        captured_output = io.StringIO()
        
        try:
            sys.stdout = captured_output 
            
            # Call the original visualization function
            subplot_fig, df_gender = visualize_function(
                male_first, female_first, male_second, female_second,
                show_summary=show_summary, show_debug=False  
            )
            
            # Close the original figure since we'll copy its content
            plt.close(subplot_fig)
            
            # Create a new subplot in our grid
            ax = fig.add_subplot(gs[row, col])
            
            # Recreate the plot in our grid
            sns.boxplot(x='Stage', y='Score', hue='Gender', data=df_gender, 
                       palette=palette, ax=ax)
            ax.set_ylim(50, 80) 
            ax.get_legend().remove()

            
            # Customize plot
            ax.set_title(model_title[i], weight='bold')
            ax.set_ylabel("Qualification Score")
            ax.set_xlabel("Selection Stage")
                
            # Print captured output if requested
            if show_summary and captured_output.getvalue():
                print(f"\nSummary for {model_name}:")
                print(captured_output.getvalue())
                
        finally:
            sys.stdout = original_stdout  
        
    # Create a single legend above all plots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Gender", loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])  