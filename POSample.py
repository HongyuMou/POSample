import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import linregress
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import t
from scipy import stats
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D

import os
import warnings
import sys


import time
start_time = time.time()

def POSample(csv_file_path):
    
    # Extract the directory from the file_path
    directory = os.path.dirname(csv_file_path)
    
    # Create a path for the 'figures' subdirectory
    figures_directory = os.path.join(directory, 'figures')
    # Create the 'figures' directory if it doesn't exist
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)
        
    # Create a path for the 'tables' subdirectory
    tables_directory = os.path.join(directory, 'tables')
    # Create the 'tables' directory if it doesn't exist
    if not os.path.exists(tables_directory):
        os.makedirs(tables_directory)
    

    # Specifically to suppress SettingWithCopyWarning:
    warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)

    # Read data from CSV
    data_all = pd.read_csv(csv_file_path)
    data_all_available = data_all[data_all['Available'] == 1]

    # Modify here to iterate over each unique program
    unique_programs = data_all['program_id'].unique()

    for program_id in unique_programs:
        
            print("-" * 40)
            print(f"Program: {program_id}")

            data = data_all[data_all['program_id'] == program_id]

            # Calculate the percentage of students in the entire sample with GPA >= GPA_cutoff
            percentage_above_cutoff_total = len(data[data['gpa'] >= data['GPA_cutoff']]) / len(data) * 100
            
            data['plot_percentage_gpa_cutoff'] = (100 - percentage_above_cutoff_total) / 100

            def custom_min_max_scaling(percentiles, min_value, max_value):
                """Apply custom Min-Max scaling."""
                range_value = max_value - min_value
                scaled_percentiles = (percentiles - min_value) / range_value
                return scaled_percentiles

            # Calculate Percentile Ranking Among Admitted for Q1
            # Create a boolean mask for those who were admitted by Q1
            admitted_by_Q1 = data['Admitted_Q1'] == 1
            # Initialize an array with NaN values to store percentiles
            percentile_ranking_Q1 = np.full(len(data), np.nan)
            # Calculate GPA-based percentile ranking only for those admitted by Q1
            admitted_GPA_Q1 = data.loc[admitted_by_Q1, 'percentile_GPA_applyQ1']
            # Calculate percentile ranks
            percentile_ranks = rankdata(admitted_GPA_Q1, method='average') / np.sum(admitted_by_Q1)
            min_rank, max_rank = np.min(percentile_ranks), np.max(percentile_ranks)
            scaled_percentile_ranks = custom_min_max_scaling(percentile_ranks, min_rank, max_rank)
            # Fill the percentile ranks into the correct positions
            percentile_ranking_Q1[admitted_by_Q1] = scaled_percentile_ranks
            # Add the 'Percentile_Ranking_Q1' column to the DataFrame
            data['percentile_GPA_addimitQ1'] = percentile_ranking_Q1


            # Calculate Percentile Ranking Among Admitted for Q2
            # Create a boolean mask for those who were admitted by Q2
            admitted_by_Q2 = data['Admitted_Q2'] == 1
            # Initialize an array with NaN values to store percentiles
            percentile_ranking_Q2 = np.full(len(data), np.nan)
            # Calculate S-based percentile ranking only for those admitted by Q2
            admitted_S_Q2 = data.loc[admitted_by_Q2, 'S']
            # Calculate percentile ranks for Q2
            percentile_ranks_Q2 = rankdata(admitted_S_Q2, method='average') / np.sum(admitted_by_Q2) 
            min_rank, max_rank = np.min(percentile_ranks_Q2), np.max(percentile_ranks_Q2)
            scaled_percentile_ranks = custom_min_max_scaling(percentile_ranks_Q2, min_rank, max_rank)
            # Fill the percentile ranks into the correct positions for Q2
            percentile_ranking_Q2[admitted_by_Q2] = scaled_percentile_ranks -1
            # Add the 'Percentile_Ranking_Q2' column to the DataFrame
            data['percentile_S_addimitQ2'] = percentile_ranking_Q2

            filtered_data = data[data['Applied_Q2'] == 1] 
            filtered_data = filtered_data.copy()

            # Create custom bin edges for GPA
            num_bins = 20
            bin_edges_GPA = [0] + [1/num_bins + i * ((1-1/num_bins) - 1/num_bins) / (num_bins - 2) for i in range(num_bins - 1)] + [1]

            # Use .loc to ensure that you are modifying the original DataFrame
            # Calculate bins and midpoints for GPA
            filtered_data.loc[:, 'GPA_bin_edges_GPA_Q2'], bins_edges = pd.cut(
                filtered_data['percentile_GPA_applyQ1'], 
                bins=bin_edges_GPA, 
                retbins=True, 
                right=False, 
                include_lowest=True
            )
            filtered_data.loc[:, 'GPA_bin_midpoints_Q2'] = filtered_data['GPA_bin_edges_GPA_Q2'].apply(lambda x: (x.left + x.right) / 2)

            # Calculate bins and midpoints for Score
            filtered_data.loc[:, 'S_bin_edges_GPA_Q2'], bins_edges = pd.cut(
                filtered_data['percentile_S_applyQ2'], # !! not percentile_S_addmitQ2
                bins=bin_edges_GPA, 
                retbins=True, 
                right=False, 
                include_lowest=True
            )
            filtered_data.loc[:, 'S_bin_midpoints_Q2'] = filtered_data['S_bin_edges_GPA_Q2'].apply(lambda x: (x.left + x.right) / 2)

            # Create equal interval bins for GPA and Score
            filtered_data.loc[:, 'GPA_bin_Q2'] = pd.cut(
                filtered_data['percentile_GPA_applyQ1'], 
                bins=num_bins, 
                labels=np.arange(num_bins)
            )
            filtered_data.loc[:, 'Score_bin_Q2'] = pd.cut(
                filtered_data['percentile_S_applyQ2'], 
                bins=num_bins, 
                labels=np.arange(num_bins)
            )


            data['S_cutoff'] = 0
            plot_percentage_S_cutoff_appliedQ2 = len(data[(data['S']<= 0) & (data['Applied_Q2']== 1)]) / len(data[data['Applied_Q2'] == 1])


            # ## Equation (1): OLS Regression for rows where GPA >= GPA_cutoff

            eq1_admitted_ols_data = data[data['gpa'] >= data['GPA_cutoff']].dropna(subset=['percentile_GPA_applyQ1', 'Y', 'background'])
            eq1_admitted_X = eq1_admitted_ols_data[['percentile_GPA_applyQ1', 'background']]
            eq1_admitted_y = eq1_admitted_ols_data['Y']
            # Add a constant to the model (intercept)
            eq1_admitted_X = sm.add_constant(eq1_admitted_X)
            # Fit the OLS model
            eq1_admitted_model = sm.OLS(eq1_admitted_y, eq1_admitted_X).fit()
            # print(eq1_admitted_model.summary())
            eq1_admitted_predicted_Y = eq1_admitted_model.predict(eq1_admitted_X)

            # Extract coefficients for Equation (1)
            coefficients_eq1_admitted = eq1_admitted_model.params
            p_values_eq1_admitted = eq1_admitted_model.pvalues
            equation1 = f"Y hat = {coefficients_eq1_admitted[0]:.2f} "
            for variable in coefficients_eq1_admitted.index[1:]:
                equation1 += f"+ ({coefficients_eq1_admitted[variable]:.2f})*{variable} "
                equation1 += f"(p={p_values_eq1_admitted[variable]:.2g}) "
            print("\nEquation (1) for the admitted:")
            print(equation1)

            # Calculate Y_GPA for all percentile_GPA_applyQ1 values
            data['Y_GPA_predicted'] = eq1_admitted_model.params[0] + eq1_admitted_model.params[1] * data['percentile_GPA_applyQ1'] + eq1_admitted_model.params[2] * data['background']
            data['residuals_admitted_Q1'] = data['Y'][data['percentile_GPA_applyQ1'] >= data['plot_percentage_gpa_cutoff']] - data['Y_GPA_predicted'][data['percentile_GPA_applyQ1'] >= data['plot_percentage_gpa_cutoff']]
            # Standard deviation: sigma_1 for conditional probability
            sigma_1 = np.std(data['residuals_admitted_Q1'] , ddof=3)
            print(f"\nStd of Equation (1) for the admitted: {sigma_1:.2f}")


            # ## Equation (1): OLS Regression for rows where GPA < GPA_cutoff

            eq1_notadmitted_ols_data = data[data['gpa'] < data['GPA_cutoff']].dropna(subset=['percentile_GPA_applyQ1', 'Y', 'background'])
            eq1_notadmitted_X = eq1_notadmitted_ols_data[['percentile_GPA_applyQ1', 'background']]
            eq1_notadmitted_y = eq1_notadmitted_ols_data['Y']
            # Add a constant to the model (intercept)
            eq1_notadmitted_X = sm.add_constant(eq1_notadmitted_X)
            # Fit the OLS model
            eq1_notadmitted_model = sm.OLS(eq1_notadmitted_y, eq1_notadmitted_X).fit()
            eq1_notadmitted_predicted_Y = eq1_notadmitted_model.predict(eq1_notadmitted_X)

            # Extract coefficients for Equation (1)
            coefficients_eq1_notadmitted = eq1_notadmitted_model.params
            p_values_eq1_notadmitted = eq1_notadmitted_model.pvalues
            equation1 = f"Y hat = {coefficients_eq1_notadmitted[0]:.2f} "
            for variable in coefficients_eq1_notadmitted.index[1:]:
                equation1 += f"+ ({coefficients_eq1_notadmitted[variable]:.2f})*{variable} "
                equation1 += f"(p={p_values_eq1_notadmitted[variable]:.2g}) "
            print("\nEquation (1) for the not admitted:")
            print(equation1)

            # Calculate Y_GPA for all percentile_GPA_applyQ1 values
            data['Y_GPA_predicted_notadmitted'] = eq1_notadmitted_model.params[0] \
                                                  + eq1_notadmitted_model.params[1] * data['percentile_GPA_applyQ1'] \
                                                  + eq1_notadmitted_model.params[2] * data['background']

            # Now calculate residuals
            data['residuals_notadmitted_Q1'] = data['Y'][data['percentile_GPA_applyQ1'] < data['GPA_cutoff']] \
                                               - data['Y_GPA_predicted_notadmitted'][data['percentile_GPA_applyQ1'] < data['GPA_cutoff']]
            # Standard deviation: sigma_1 for conditional probability
            sigma_1_notadmitted = np.std(data['residuals_notadmitted_Q1'] , ddof=3)
            print(f"\nStd of Equation (1) for the not admitted: {sigma_1_notadmitted:.2f}")



            # Plot: Define ranges for admitted and not admitted groups
            X_plot_range_admitted = np.linspace(eq1_admitted_X['percentile_GPA_applyQ1'].min(), eq1_admitted_X['percentile_GPA_applyQ1'].max(), 100)
            X_plot_range_notadmitted = np.linspace(eq1_notadmitted_X['percentile_GPA_applyQ1'].min(), eq1_notadmitted_X['percentile_GPA_applyQ1'].max(), 100)

            # Function to create plot data
            def create_plot_data(model, X_plot_range, background_value):
                X_plot = pd.DataFrame({'const': 1, 'percentile_GPA_applyQ1': X_plot_range, 'background': background_value * np.ones(len(X_plot_range))})
                Y_predicted_plot = model.predict(X_plot)
                return X_plot_range, Y_predicted_plot

            # Plotting
            plt.figure(figsize=(12, 6))

            # Admitted group, background 0 and 1
            X_range, Y_predicted = create_plot_data(eq1_admitted_model, X_plot_range_admitted, 0)
            plt.plot(X_range, Y_predicted, color='orange', label='Admitted, Background 0')
            X_range, Y_predicted = create_plot_data(eq1_admitted_model, X_plot_range_admitted, 1)
            plt.plot(X_range, Y_predicted, color='red', label='Admitted, Background 1')

            # Not admitted group, background 0 and 1
            X_range, Y_predicted = create_plot_data(eq1_notadmitted_model, X_plot_range_notadmitted, 0)
            plt.plot(X_range, Y_predicted, color='orange', linestyle='--', label='Not Admitted, Background 0')
            X_range, Y_predicted = create_plot_data(eq1_notadmitted_model, X_plot_range_notadmitted, 1)
            plt.plot(X_range, Y_predicted, color='red', linestyle='--', label='Not Admitted, Background 1')

            # Add a vertical line for GPA cutoff
            plt.axvline(x=data['plot_percentage_gpa_cutoff'].iloc[0], color='black', linestyle='--', linewidth=2, label='GPA Cutoff')

            # Customize plot
            plt.xlabel('Percentile GPA Apply Q1')
            plt.ylabel('Y')
            plt.title('OLS Regression Lines for Q1 Admitted and Not Admitted Groups Across Backgrounds (Equation 1)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_OLS_eq1.png'))
            plt.close()
            

            # Equation (2)
            S_cutoff = 0
            eq2_admitted_ols_data = data[data['S'] > S_cutoff].dropna(subset=['percentile_GPA_applyQ1', 'percentile_S_applyQ2', 'Y', 'background'])
            eq2_admitted_X = eq2_admitted_ols_data[['percentile_GPA_applyQ1', 'percentile_S_applyQ2', 'background']]
            eq2_admitted_y = eq2_admitted_ols_data['Y']
            # Add a constant to the model (intercept)
            eq2_admitted_X = sm.add_constant(eq2_admitted_X)
            # Fit the OLS model
            eq2_admitted_model = sm.OLS(eq2_admitted_y, eq2_admitted_X).fit()
            # print(eq2_admitted_model.summary())
            eq2_admitted_predicted_Y = eq2_admitted_model.predict(eq2_admitted_X)

            # Extract coefficients for Equation (2)
            coefficients_eq2_admitted = eq2_admitted_model.params
            p_values_eq2_admitted = eq2_admitted_model.pvalues
            equation2 = f"Y = {coefficients_eq2_admitted[0]:.2f} "
            for variable in coefficients_eq2_admitted.index[1:]:
                equation2 += f"+ ({coefficients_eq2_admitted[variable]:.2f})*{variable} "
                equation2 += f"(p={p_values_eq2_admitted[variable]:.2g}) "
            print("\nEquation (2) for the admitted:")
            print(equation2)
            
            

            # Calculate Y_S for all percentile_S_applyQ2 values
            data['Y_s_GPA_Q2_predicted'] = eq2_admitted_model.params[0] + eq2_admitted_model.params[1] * data['percentile_GPA_applyQ1'] + eq2_admitted_model.params[2] * data['percentile_S_applyQ2'] + eq2_admitted_model.params[3] * data['background']
            data['residuals_admitted_Q2'] = data['Y'][data['percentile_S_applyQ2'] >= plot_percentage_S_cutoff_appliedQ2] - data['Y_s_GPA_Q2_predicted'][data['percentile_S_applyQ2'] > plot_percentage_S_cutoff_appliedQ2]
            # Standard deviation: sigma_2 for conditional probability
            sigma_2 = np.std(data['residuals_admitted_Q2'] , ddof=4)
            print(f"\nStd of Equation (2) for the admitted: {sigma_2:.2f}")


            # not admitted:
            eq2_notadmitted_ols_data = data[data['S'] <= S_cutoff].dropna(subset=['percentile_GPA_applyQ1', 'percentile_S_applyQ2', 'Y', 'background'])
            eq2_notadmitted_X = eq2_notadmitted_ols_data[['percentile_GPA_applyQ1', 'percentile_S_applyQ2', 'background']]
            eq2_notadmitted_y = eq2_notadmitted_ols_data['Y']
            # Add a constant to the model (intercept)
            eq2_notadmitted_X = sm.add_constant(eq2_notadmitted_X)
            # Fit the OLS model
            eq2_notadmitted_model = sm.OLS(eq2_notadmitted_y, eq2_notadmitted_X).fit()
            eq2_notadmitted_predicted_Y = eq2_notadmitted_model.predict(eq2_notadmitted_X)

            coefficients_eq2_notadmitted = eq2_notadmitted_model.params
            p_values_eq2_notadmitted = eq2_notadmitted_model.pvalues
            equation2 = f"Y = {coefficients_eq2_notadmitted[0]:.2f} "
            for variable in coefficients_eq2_notadmitted.index[1:]:
                equation2 += f"+ ({coefficients_eq2_notadmitted[variable]:.2f})*{variable} "
                equation2 += f"(p={p_values_eq2_notadmitted[variable]:.2g}) "
            print("\nEquation (2) for the not admitted:")
            print(equation2)

            # Calculate Y_S for all percentile_S_applyQ2 values
            data['Y_s_GPA_Q2_predicted_notadmitted'] = eq2_notadmitted_model.params[0] + eq2_notadmitted_model.params[1] * data['percentile_GPA_applyQ1'] + eq2_notadmitted_model.params[2] * data['percentile_S_applyQ2'] + eq2_notadmitted_model.params[3] * data['background']
            data['residuals_notadmitted_Q2'] = data['Y'][data['percentile_S_applyQ2'] <= plot_percentage_S_cutoff_appliedQ2] - data['Y_s_GPA_Q2_predicted'][data['percentile_S_applyQ2'] <= plot_percentage_S_cutoff_appliedQ2]
            sigma_2_notadmitted = np.std(data['residuals_notadmitted_Q2'] , ddof=4)
            print(f"\nStd of Equation (2) for the not admitted: {sigma_2_notadmitted:.2f}")



            # Plot: 
            X_plot_range_admitted = np.linspace(eq2_admitted_X['percentile_S_applyQ2'].min(), eq2_admitted_X['percentile_S_applyQ2'].max(), 100)
            X_plot_range_notadmitted = np.linspace(eq2_notadmitted_X['percentile_S_applyQ2'].min(), eq2_notadmitted_X['percentile_S_applyQ2'].max(), 100)


            # Function to create plot data for Equation (2)
            def create_plot_data_eq2(model, X_plot_range, background_value):
                X_plot = pd.DataFrame({
                    'const': 1, 
                    'percentile_GPA_applyQ1': np.mean(eq2_admitted_X['percentile_GPA_applyQ1']), # Use average value for plotting
                    'percentile_S_applyQ2': X_plot_range, 
                    'background': background_value * np.ones(len(X_plot_range))
                })
                Y_predicted_plot = model.predict(X_plot)
                return X_plot_range, Y_predicted_plot

            # Plotting
            plt.figure(figsize=(12, 6))

            # Admitted group, background 0 and 1 for Equation (2)
            X_range, Y_predicted = create_plot_data_eq2(eq2_admitted_model, X_plot_range_admitted, 0)
            plt.plot(X_range, Y_predicted, color='orange', label='Admitted, Background 0')
            X_range, Y_predicted = create_plot_data_eq2(eq2_admitted_model, X_plot_range_admitted, 1)
            plt.plot(X_range, Y_predicted, color='red', label='Admitted, Background 1')

            # Not Admitted group, background 0 and 1 for Equation (2)
            X_range, Y_predicted = create_plot_data_eq2(eq2_notadmitted_model, X_plot_range_notadmitted, 0)
            plt.plot(X_range, Y_predicted, color='orange', linestyle='--', label='Not Admitted, Background 0')
            X_range, Y_predicted = create_plot_data_eq2(eq2_notadmitted_model, X_plot_range_notadmitted, 1)
            plt.plot(X_range, Y_predicted, color='red', linestyle='--', label='Not Admitted, Background 1')

            # Add a vertical line for S cutoff
            plt.axvline(x=plot_percentage_S_cutoff_appliedQ2, color='black', linestyle='--', linewidth=2, label='S Cutoff')

            # Customize plot
            plt.xlabel('Percentile S Apply Q2')
            plt.ylabel('Y')
            plt.title('OLS Regression Lines for Q2 Admitted and Not Admitted Group Across Backgrounds (Equation 2)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_OLS_eq2.png'))
            plt.close()


            ## Check if the residuals from equation (1) and (2) look normal
            # Convert the column to numeric if it's not already
            data.loc[:, 'residuals_admitted_Q1'] = pd.to_numeric(data['residuals_admitted_Q1'], errors='coerce')
            data.loc[:, 'residuals_admitted_Q2'] = pd.to_numeric(data['residuals_admitted_Q2'], errors='coerce')
            # Drop NA values that might have resulted from conversion errors
            data_test1 = data.dropna(subset=['residuals_admitted_Q1'])
            data_test2 = data.dropna(subset=['residuals_admitted_Q2'])
            # Perform the Kolmogorov-Smirnov test for Q1 residuals
            standardized_residuals_Q1 = (data_test1['residuals_admitted_Q1'] - data_test1['residuals_admitted_Q1'].mean()) / data_test1['residuals_admitted_Q1'].std()
            ks_statistic_Q1, p_value_Q1 = stats.kstest(standardized_residuals_Q1, 'norm')
            # Interpret the results for Q1
            alpha = 0.05
            result_Q1 = 'Normal' if p_value_Q1 > alpha else 'Not Normal'

            # Perform the Kolmogorov-Smirnov test for Q2 residuals
            standardized_residuals_Q2 = (data_test2['residuals_admitted_Q2'] - data_test2['residuals_admitted_Q2'].mean()) / data_test2['residuals_admitted_Q2'].std()
            ks_statistic_Q2, p_value_Q2 = stats.kstest(standardized_residuals_Q2, 'norm')
            # Interpret the results for Q2
            result_Q2 = 'Normal' if p_value_Q2 > alpha else 'Not Normal'


            # Create a figure with two subplots
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 5))

            # Plot histogram for residuals on the first subplot (Q1)
            ax1.hist(data['residuals_admitted_Q1'], bins=30, edgecolor='black', alpha=0.7)
            ax1.set_title('Histogram of $\epsilon_1$')
            ax1.set_xlabel('Residual value')
            ax1.set_ylabel('Frequency')

            # Create a secondary y-axis for the KDE plot (Q1)
            ax2 = ax1.twinx()
            sns.kdeplot(data['residuals_admitted_Q1'], color="red", label="KDE", ax=ax2)
            ax2.set_ylabel('Density')
            ax2.legend(loc='upper left')

            # Annotate with normality result for Q1
            ax1.text(0.95, 0.95, f'K-S Test: {result_Q1}', 
                     transform=ax1.transAxes, horizontalalignment='right', verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Plot histogram for residuals on the second subplot (Q2)
            ax3.hist(data['residuals_admitted_Q2'], bins=30, edgecolor='black', alpha=0.7)
            ax3.set_title('Histogram of $\epsilon_2$')
            ax3.set_xlabel('Residual value')
            ax3.set_ylabel('Frequency')

            # Create a secondary y-axis for the KDE plot (Q2)
            ax4 = ax3.twinx()
            sns.kdeplot(data['residuals_admitted_Q2'], color="blue", label="KDE", ax=ax4)
            ax4.set_ylabel('Density')
            ax4.legend(loc='upper left')

            # Annotate with normality result for Q2
            ax3.text(0.95, 0.95, f'K-S Test: {result_Q2}', 
                     transform=ax3.transAxes, horizontalalignment='right', verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Display the plots
            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_residuals_normality.png'))
            plt.close()

            # Custom bin edges
            bin_edges_GPA = [0] + [1/num_bins + i * ((1-1/num_bins) - 1/num_bins) / (num_bins - 2) for i in range(num_bins - 1)] + [1]
            # Create bins and find the midpoints for GPA
            data['GPA_bin_edges'], bins_edges = pd.cut(data['percentile_GPA_applyQ1'], bins=bin_edges_GPA, retbins=True, right=False, include_lowest=True)
            # Calculate midpoints
            data['GPA_bin_midpoints'] = data['GPA_bin_edges'].apply(lambda x: (x.left + x.right) / 2)

            # Create equal interval bins for GPA and Score using .loc to avoid the warning
            data.loc[:, 'GPA_bin'] = pd.cut(data['percentile_GPA_applyQ1'], bins=num_bins, labels=np.arange(num_bins))

            # Filter out rows where Q2-related variables have NaN
            data_filtered = data.dropna(subset=['percentile_GPA_applyQ1', 'percentile_S_applyQ2', 'Y_s_GPA_Q2_predicted']).copy()

            num_bins = 20
            # Custom bin edges
            bin_edges_GPA = [0] + [1/num_bins + i * ((1-1/num_bins) - 1/num_bins) / (num_bins - 2) for i in range(num_bins - 1)] + [1]
            # Create bins and find the midpoints for GPA
            data_filtered['GPA_bin_edges_GPA_Q2'], bins_edges = pd.cut(data_filtered['percentile_GPA_applyQ1'], bins=bin_edges_GPA, retbins=True, right=False, include_lowest=True)
            # Calculate midpoints
            data_filtered['GPA_bin_midpoints_Q2'] = data_filtered['GPA_bin_edges_GPA_Q2'].apply(lambda x: (x.left + x.right) / 2)

            # Create bins and find the midpoints for S
            data_filtered['S_bin_edges_GPA_Q2'], bins_edges = pd.cut(data_filtered['percentile_S_applyQ2'], bins=bin_edges_GPA, retbins=True, right=False, include_lowest=True)
            # Calculate midpoints
            data_filtered['S_bin_midpoints_Q2'] = data_filtered['S_bin_edges_GPA_Q2'].apply(lambda x: (x.left + x.right) / 2)

            # Create equal interval bins for GPA and Score using .loc to avoid the warning
            data_filtered.loc[:, 'GPA_bin_Q2'] = pd.cut(data_filtered['percentile_GPA_applyQ1'], bins=num_bins, labels=np.arange(num_bins))
            data_filtered.loc[:, 'Score_bin_Q2'] = pd.cut(data_filtered['percentile_S_applyQ2'], bins=num_bins, labels=np.arange(num_bins))

            # Define common bins edges for all histograms: bins with equal intervals between the minimum and maximum values
            bin_edges = np.linspace(min(data[['Y_GPA_predicted', 'Y']].min()),
                                    max(data[['Y_GPA_predicted', 'Y']].max()), 
                                    21)  # 21 edges for 20 bars


            # Group by GPA and Score bins to calculate the PMF
            grouped = data_filtered.groupby(['GPA_bin_Q2', 'Score_bin_Q2']).size().reset_index(name='count')

            # Calculate the total counts for each GPA_bin
            gpa_bin_counts = grouped.groupby('GPA_bin_Q2')['count'].transform('sum')

            # Calculate the PMF normalized by each GPA_bin's total counts
            grouped['pmf'] = grouped['count'] / gpa_bin_counts

            # Pivot the table to get a grid representation
            grid_pmf = grouped.pivot(index='GPA_bin_Q2', columns='Score_bin_Q2', values='pmf')

            # Group by GPA and Score bins to calculate the mean Y_s_GPA_Q2_predicted
            grouped_values = data_filtered.groupby(['GPA_bin_Q2', 'Score_bin_Q2']).agg({
                'Y_s_GPA_Q2_predicted': 'mean'
            }).reset_index().rename(columns={'Y_s_GPA_Q2_predicted': 'mean_Y_s_GPA_Q2_predicted'})

            # For each GPA bin, compute a weighted sum of mean_Y_s_GPA_Q2_predicted using the PMFs
            def compute_weighted_average(gpa_bin):
                weights = grid_pmf.loc[gpa_bin].values
                y_values_grouped = grouped_values[grouped_values['GPA_bin_Q2'] == gpa_bin]['mean_Y_s_GPA_Q2_predicted'].values

                return np.dot(y_values_grouped, weights)

            data_filtered['Y_GPA_Q2'] = data_filtered['GPA_bin_Q2'].apply(compute_weighted_average)

            # Now, join/merge this back to the original 'data' DataFrame if needed
            data = data.merge(data_filtered[['Y_GPA_Q2']], left_index=True, right_index=True, how='left')

            # Create Y bins based on 'Y' value and 'bin_edges'
            data_filtered['Y_bin'] = pd.cut(data['Y'], bins=bin_edges, include_lowest=True)
            data_filtered['Y_GPA_Q2_bin'] = pd.cut(data['Y_GPA_Q2'], bins=bin_edges, include_lowest=True)

            # Filter data for background = 0 and background = 1
            data_bg0 = data[data['background'] == 0]
            data_bg1 = data[data['background'] == 1]

            # Filtered data considering only rows where Q2 > 0, for each background
            data_filtered_bg0 = data_filtered[data_filtered['background'] == 0]
            data_filtered_bg1 = data_filtered[data_filtered['background'] == 1]

            # Grouping and calculating standard deviation for background = 0
            std_Y_by_GPA_bg_0 = data_bg0.groupby('GPA_bin')['Y'].std()
            std_Y_by_GPA_Q2_bg_0 = data_filtered_bg0.groupby('GPA_bin_Q2')['Y'].std().sort_index()

            # Grouping and calculating standard deviation for background = 1
            std_Y_by_GPA_bg_1 = data_bg1.groupby('GPA_bin')['Y'].std()
            std_Y_by_GPA_Q2_bg_1 = data_filtered_bg1.groupby('GPA_bin_Q2')['Y'].std().sort_index()

            # ## P(Y | Q2, GPA) 

            # Function to calculate conditional probabilities
            def calculate_conditional_probabilities(data):
                counts_per_group = data.groupby(['GPA_bin_Q2', 'Score_bin_Q2']).size()
                total_counts_per_gpa_bin = data.groupby('GPA_bin_Q2').size()
                prob_s_given_gpa_q2 = counts_per_group.div(total_counts_per_gpa_bin, level='GPA_bin_Q2')
                prob_s_given_gpa_q2_df = prob_s_given_gpa_q2.reset_index(name='Conditional_Probability')
                return prob_s_given_gpa_q2_df.pivot('Score_bin_Q2', 'GPA_bin_Q2', 'Conditional_Probability')

            # Calculate matrices for background = 0 and background = 1
            prob_s_given_gpa_q2_bg0 = calculate_conditional_probabilities(data_filtered_bg0)
            prob_s_given_gpa_q2_bg1 = calculate_conditional_probabilities(data_filtered_bg1)

            # Plotting subfigures
            # Determine common scale for the color range
            vmin = min(prob_s_given_gpa_q2_bg0.min().min(), prob_s_given_gpa_q2_bg1.min().min())  # Minimum value for color scale
            vmax = max(prob_s_given_gpa_q2_bg0.max().max(), prob_s_given_gpa_q2_bg1.max().max())  # Maximum value for color scale

            fig, axs = plt.subplots(1, 2, figsize=(28, 10))

            # Plot for background = 0
            sns.heatmap(prob_s_given_gpa_q2_bg0, annot=True, fmt=".2f", cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[0])
            axs[0].set_title('PMF P[S | GPA, Q2 > 0, B=0]')
            axs[0].set_xlabel('GPA Bin')
            axs[0].set_ylabel('S Bin')
            axs[0].xaxis.tick_top()
            axs[0].xaxis.set_label_position('top')
            axs[0].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            # Plot for background = 1
            sns.heatmap(prob_s_given_gpa_q2_bg1, annot=True, fmt=".2f", cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[1])
            axs[1].set_title('PMF P[S | GPA, Q2 > 0, B=1]')
            axs[1].set_xlabel('GPA Bin')
            axs[1].set_ylabel('S Bin')
            axs[1].xaxis.tick_top()
            axs[1].xaxis.set_label_position('top')
            axs[1].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P[S | GPA, Q2 > 0,b]_PMF.png'))
            plt.close()


            # #### Calculate P[Y in y_bin | GPA(%) in gpa_bin, Q2>0, s(%) in S_bin]

            def calculate_probabilities_and_mapping(data_filtered,sigma_2):
                y_gpa_q2_bin_edges = data_filtered['Y_GPA_Q2_bin'].cat.categories
                results = pd.DataFrame(columns=['GPA_bin_Q2', 'Score_bin_Q2', 'Y_GPA_Q2_bin', 'P_Y_s_GPA_Q2_predicted'])

                score_bins = data_filtered['Score_bin_Q2'].dropna().unique()

                for (g, s), group in data_filtered.groupby(['GPA_bin_Q2', 'Score_bin_Q2']):
                    mu_2_1 = group['Y_s_GPA_Q2_predicted'].mean() # from equation (2)
                    sigma_2 = sigma_2 # global std

                    for idx, interval in enumerate(y_gpa_q2_bin_edges):
                        left_format = f"{interval.left:.2f}" if interval.left * 10 % 1 == 0 else f"{interval.left:.3f}"
                        right_format = f"{interval.right:.2f}" if interval.right * 10 % 1 == 0 else f"{interval.right:.3f}"
                        prob = norm.cdf((interval.right - mu_2_1) / sigma_2) - norm.cdf((interval.left - mu_2_1) / sigma_2)
                        results = results.append({
                            'GPA_bin_Q2': g,
                            'Score_bin_Q2': s,
                            'Y_GPA_Q2_bin': f"({left_format}, {right_format}]",
                            'Y_GPA_Q2_bin_index': idx,
                            'P_Y_s_GPA_Q2_predicted': prob
                        }, ignore_index=True)

                results.set_index(['GPA_bin_Q2', 'Score_bin_Q2', 'Y_GPA_Q2_bin_index','Y_GPA_Q2_bin'], inplace=True)

                # Extract unique mapping
                unique_mapping_df = results.reset_index()[['Y_GPA_Q2_bin', 'Y_GPA_Q2_bin_index']].drop_duplicates()

                return results['P_Y_s_GPA_Q2_predicted'].squeeze(), unique_mapping_df

            # Calculate probabilities and unique mappings for background = 0 and background = 1
            prob_y_given_gpa_s_q2_bg0, unique_mapping_bg0 = calculate_probabilities_and_mapping(data_filtered_bg0, sigma_2)
            prob_y_given_gpa_s_q2_bg1, unique_mapping_bg1 = calculate_probabilities_and_mapping(data_filtered_bg1, sigma_2)


            # $$P[Y \in Y_{\text {bin }} | GPA(\%) \in GPA(\%)_{\text {bin }}, Q2>0] = 
            #      \sum_s  P\left[Y\_{GPA}\_{Q2} \in Y_{\text {bin }} | GPA(\%) \in GPA(\%)_{\text {bin }}, Q2>0, s(\%) \in S(\%)_{\text {bin }}\right] * P\left[s(\%) \in S(\%)_{\text {bin }}| GPA(\%)_{\text {bin }}, Q2>0\right]$$

            # Function to convert interval string to pd.Interval object
            def convert_to_interval(interval_str):
                # Removing all non-numeric characters except the decimal point and comma
                clean_str = ''.join(char for char in interval_str if char.isdigit() or char in ['.', ','])
                left, right = clean_str.split(',')
                # Converting to float and creating an Interval object
                return pd.Interval(float(left), float(right), closed='right')

            unique_mapping_df = unique_mapping_bg0 # unique_mapping_bg1 also the same
            # Convert Y_GPA_Q2_bin in unique_mapping_df to pd.Interval
            unique_mapping_df['Y_GPA_Q2_bin'] = unique_mapping_df['Y_GPA_Q2_bin'].apply(convert_to_interval)
            unique_mapping_df['Y_bin_index'] = unique_mapping_df['Y_GPA_Q2_bin_index']
            unique_mapping_df['Y_bin'] = unique_mapping_df['Y_GPA_Q2_bin']
            unique_mapping_df = unique_mapping_df.drop(['Y_GPA_Q2_bin', 'Y_GPA_Q2_bin_index'], axis=1)


            # Merge data_filtered with unique_mapping_df
            data_filtered_bg1 = pd.merge(
                data_filtered_bg1,
                unique_mapping_df,
                on='Y_bin',
                how='left'
            )

            data_filtered_bg0 = pd.merge(
                data_filtered_bg0,
                unique_mapping_df,
                on='Y_bin',
                how='left'
            )

            # Create Y bins based on 'Y' value and 'bin_edges'
            data['Y_bin'] = pd.cut(data['Y'], bins=bin_edges, include_lowest=True)
            data = pd.merge(data, unique_mapping_df, on='Y_bin', how='left')

            data_bg1['Y_bin'] = pd.cut(data_bg1['Y'], bins=bin_edges, include_lowest=True)
            data_bg0['Y_bin'] = pd.cut(data_bg0['Y'], bins=bin_edges, include_lowest=True)
            # Merge data_filtered with unique_mapping_df
            data_bg1 = pd.merge(
                data_bg1,
                unique_mapping_df,
                on='Y_bin',
                how='left'
            )

            data_bg0 = pd.merge(
                data_bg0,
                unique_mapping_df,
                on='Y_bin',
                how='left'
            )


            # In[19]:


            if 'Y_GPA_Q2_bin' in prob_y_given_gpa_s_q2_bg0.index.names:
                prob_y_given_gpa_s_q2_bg0 = prob_y_given_gpa_s_q2_bg0.droplevel('Y_GPA_Q2_bin')
            if 'Y_GPA_Q2_bin' in prob_y_given_gpa_s_q2_bg1.index.names:
                prob_y_given_gpa_s_q2_bg1 = prob_y_given_gpa_s_q2_bg1.droplevel('Y_GPA_Q2_bin')


            # In[20]:


            gpa_bin_ranges_bg0 = data_filtered_bg0['GPA_bin_Q2'].unique()
            Y_bin_index = data['Y_bin_index'].unique()
            gpa_bin_ranges_bg1 = data_filtered_bg1['GPA_bin_Q2'].unique()

            def calculate_prob_y_given_gpa_q2(prob_s_given_gpa_q2, prob_y_given_gpa_s_q2, gpa_bin_ranges, Y_bin_index):
                # Initialize an empty DataFrame for storing the results
                prob_y_given_gpa_q2 = pd.DataFrame(columns=['GPA_bin_Q2', 'Y_bin_index', 'Conditional_Probability'])

                for gpa_bin in gpa_bin_ranges:
                    for y_bin in Y_bin_index:
                        total_prob = 0

                        # Extract probabilities for the current gpa_bin
                        if gpa_bin in prob_s_given_gpa_q2.columns:
                            states_probs = prob_s_given_gpa_q2[gpa_bin]

                            for s, prob_s in states_probs.iteritems():
                                # Get the conditional probability of Y given s and gpa_bin
                                if (gpa_bin, s, y_bin) in prob_y_given_gpa_s_q2.index:
                                    y_given_gpa_s_prob = prob_y_given_gpa_s_q2.loc[(gpa_bin, s, y_bin)]
                                    total_prob += y_given_gpa_s_prob * prob_s

                        prob_y_given_gpa_q2 = prob_y_given_gpa_q2.append({
                            'GPA_bin_Q2': gpa_bin,
                            'Y_bin_index': y_bin,
                            'Conditional_Probability': total_prob
                        }, ignore_index=True)

                matrix_p_y_gpa_q2 = prob_y_given_gpa_q2.pivot(index='Y_bin_index', columns='GPA_bin_Q2', values='Conditional_Probability')

            #     # Sum up the 'Conditional_Probability' by 'GPA_bin_Q2'
            #     prob_sums = prob_y_given_gpa_q2.groupby('GPA_bin_Q2')['Conditional_Probability'].sum().reset_index()
            #     prob_sums['Sum_Close_to_1'] = prob_sums['Conditional_Probability'].apply(lambda x: np.isclose(x, 1))
            #     print(prob_sums)

                return matrix_p_y_gpa_q2

            # Calculate for background = 0
            matrix_p_y_gpa_q2_bg0 = calculate_prob_y_given_gpa_q2(prob_s_given_gpa_q2_bg0, prob_y_given_gpa_s_q2_bg0, gpa_bin_ranges_bg0, Y_bin_index)

            # Calculate for background = 1
            matrix_p_y_gpa_q2_bg1 = calculate_prob_y_given_gpa_q2(prob_s_given_gpa_q2_bg1, prob_y_given_gpa_s_q2_bg1, gpa_bin_ranges_bg1, Y_bin_index)


            # In[21]:


            def plot_heatmap(matrix, bin_edges, bin_edges_GPA, background, ax):
                # Convert index and columns to strings for better labeling
                matrix.index = [str(interval) for interval in matrix.index]
                matrix.columns = [str(interval) for interval in matrix.columns]

                # Plotting
                cax = ax.imshow(matrix, cmap='Reds', aspect='auto', vmax=1, vmin=0)

                # Annotate the heatmap with the actual values
                for (i, j), val in np.ndenumerate(matrix.values):
                    if 0 < val < 1:  # Check if value is neither 0 nor 1
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]

                # Setting labels for the Y axis
                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)

                # Setting labels for the X axis
                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                # Move the X-axis labels to the top
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                # Title
                title = f'P(Y| Q2, GPA, B={background}): Extrapolation from Equation (2) and S Integration'
                ax.set_title(title)

                return cax

            # Create a figure with two subplots and use constrained_layout
            fig, axs = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

            # Plot heatmap for background = 0
            cax0 = plot_heatmap(matrix_p_y_gpa_q2_bg0, bin_edges, bin_edges_GPA, 0, axs[0])

            # Plot heatmap for background = 1
            cax1 = plot_heatmap(matrix_p_y_gpa_q2_bg1, bin_edges, bin_edges_GPA, 1, axs[1])

            # # Add color bar
            # fig.colorbar(cax1, ax=axs, orientation='vertical')
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_Modeled P(Y | Q2, GPA,b).png'))
            plt.close()
            


            # ## P(Y | GPA)

            # ### Equation (1) Predicted Y (Y_GPA_predicted) + Global Sigma_1

            def calculate_probabilities(data, sigma, bin_edges, bin_edges_GPA):
                gpa_bins = data['GPA_bin'].dropna().unique()
                results = pd.DataFrame(columns=['GPA_bin', 'Y_bin_index', 'Y_bin', 'P_Y_given_GPA'])

                for g in gpa_bins:
                    subset = data[data['GPA_bin'] == g]
                    mu_Y = subset['Y_GPA_predicted'].mean() # local mean

                    for i, (Y_lower, Y_upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                        if not np.isnan(sigma):
                            prob = norm.cdf((Y_upper - mu_Y) / sigma) - norm.cdf((Y_lower - mu_Y) / sigma)
                        else:
                            prob = np.nan
                        results = results.append({'GPA_bin': g, 'Y_bin_index': i, 'Y_bin': f'{Y_lower}-{Y_upper}', 'P_Y_given_GPA': prob}, ignore_index=True)

                return results.pivot(index='Y_bin_index', columns='GPA_bin', values='P_Y_given_GPA')

            y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
            x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]

            # Assuming sigma_1, bin_edges, and bin_edges_GPA are defined
            matrix_p_y_gpa_bg0 = calculate_probabilities(data_bg0, sigma_1, bin_edges, bin_edges_GPA)
            matrix_p_y_gpa_bg1 = calculate_probabilities(data_bg1, sigma_1, bin_edges, bin_edges_GPA)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [matrix_p_y_gpa_bg0, matrix_p_y_gpa_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='Blues', aspect='auto', vmax=1, vmin=0)
                ax.set_title(f'P(Y | GPA, {title}): Extrapolation from Equation (1)')
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
                ax.xaxis.set_tick_params(width=0.5)
                ax.yaxis.set_tick_params(width=0.5)
                ax.set_xlabel('GPA (%) Bin')
                ax.set_ylabel('Y Bin')

                for (i, j), val in np.ndenumerate(matrix):
                    if 0 < val < 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)

                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f'{edge:.2f}' for edge in bin_edges])
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([f'{100*edge:.0f}%' for edge in bin_edges_GPA], rotation=45)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_Modeled P(Y | GPA,b).png'))
            plt.close()


            # ## P(Q2 | GPA): probability that a student applies to Q2 given their GPA

            def calculate_prob_q2_given_gpa(data, bin_edges_GPA):
                grouped_q2_gpa = data.groupby('GPA_bin')['Applied_Q2'].agg(['count', 'sum']).reset_index()
                grouped_q2_gpa['P_Q2_given_GPA'] = grouped_q2_gpa['sum'] / grouped_q2_gpa['count']

                x_bin_labels = [f'{100*edge:.0f}%-{100*bin_edges_GPA[i+1]:.0f}%' for i, edge in enumerate(bin_edges_GPA[:-1])]
                return grouped_q2_gpa['P_Q2_given_GPA'], x_bin_labels

            # Assuming bin_edges_GPA and bin_edges are defined
            prob_q2_gpa_bg0, x_labels_bg0 = calculate_prob_q2_given_gpa(data_bg0, bin_edges_GPA)
            prob_q2_gpa_bg1, x_labels_bg1 = calculate_prob_q2_given_gpa(data_bg1, bin_edges_GPA)

            # Determine the maximum probability to set the same y scale
            max_prob = max(prob_q2_gpa_bg0.max(), prob_q2_gpa_bg1.max())

            # Number of Y bins
            num_y_bins = len(bin_edges) - 1

            # Creating matrices for both backgrounds
            matrix_p_q2_gpa_bg0 = pd.DataFrame([prob_q2_gpa_bg0.values] * num_y_bins)
            matrix_p_q2_gpa_bg1 = pd.DataFrame([prob_q2_gpa_bg1.values] * num_y_bins)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            # Common settings for both plots
            for ax, prob, x_labels, title in zip(axs, [prob_q2_gpa_bg0, prob_q2_gpa_bg1], [x_labels_bg0, x_labels_bg1], ['B=0', 'B=1']):
                ax.bar(np.arange(len(prob)), prob, color='grey')
                ax.set_title(f'PMF of P(Q2 | GPA) across GPA Bins ({title})')
                ax.set_xlabel('GPA (%) Bin')
                ax.set_ylabel('P(Q2 | GPA)')
                ax.set_xticks(np.arange(len(x_labels)))
                ax.set_xticklabels(x_labels, rotation=45)
                ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
                ax.set_ylim(0, max_prob + 0.05)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(Q2 | GPA,b).png'))
            plt.close()


            # ### Directly compute the observed LHS: P(Q2>0|Y,GPA)
            # Use the empirical PMF directly from data: counts the number of occurrences where $Q2>0$ and divides it by the total number of occurrences in each $(GPA, Y)$ cell

            def calculate_prob_Q2_above_GPA(data, unique_mapping_df, bin_edges, bin_edges_GPA):
                aboveGPA_data = data[data['gpa'] >= data['GPA_cutoff']]
                aboveGPA_data = unique_mapping_df.merge(aboveGPA_data, on='Y_bin_index', how='left')

                grouped_data = aboveGPA_data.groupby(['GPA_bin', 'Y_bin_index'])
                total_counts = grouped_data.size()
                count_Q2_gt_0 = aboveGPA_data[aboveGPA_data['Q2'] > 0].groupby(['GPA_bin', 'Y_bin_index']).size()
                prob_Q2_gt_0_given_GPA_Y = count_Q2_gt_0.div(total_counts)

                pivot_results_Q2 = prob_Q2_gt_0_given_GPA_Y.reset_index().pivot_table(
                    index='Y_bin_index', 
                    columns='GPA_bin',
                    values=0,  # Explicitly specifying the column name for values
                    dropna=False
                )
                return pivot_results_Q2

            # Assuming all necessary variables are defined
            pivot_results_Q2_bg0 = calculate_prob_Q2_above_GPA(data_bg0, unique_mapping_df, bin_edges, bin_edges_GPA)
            pivot_results_Q2_bg1 = calculate_prob_Q2_above_GPA(data_bg1, unique_mapping_df, bin_edges, bin_edges_GPA)

            # Save the results in new variables
            matrix_Q2_given_gpa_y_didata_bg0 = pivot_results_Q2_bg0
            matrix_Q2_given_gpa_y_didata_bg1 = pivot_results_Q2_bg1

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [matrix_Q2_given_gpa_y_didata_bg0, matrix_Q2_given_gpa_y_didata_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(f'P(Q2 > 0 | GPA >= GPA_cutoff, Y) from Local Approximation -- {title}')

                # Annotating the heatmap
                for (i, j), val in np.ndenumerate(matrix.values):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    else:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)

                # Setting labels for the Y axis as bin_edges
                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)

                # Setting labels for the X axis as bin_edges_GPA
                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(Q2 | GPA, Y)_localapprox.png'))
            plt.close()


            # ### Model predicted version

            def compute_matrix(matrix_p_y_gpa_q2, matrix_p_y_gpa, matrix_p_q2_gpa):
                computed_matrix = pd.DataFrame(np.zeros(matrix_p_y_gpa.shape))

                for i in range(computed_matrix.shape[0]):
                    for j in range(computed_matrix.shape[1]):
                        if matrix_p_y_gpa.iloc[i, j] != 0:  # Check if the denominator is not zero
                            value = (matrix_p_y_gpa_q2.iloc[i, j] * matrix_p_q2_gpa.iloc[i, j]) / matrix_p_y_gpa.iloc[i, j]
                            computed_matrix.iloc[i, j] = value

                return computed_matrix

            # Assuming matrices are already calculated for each background
            matrix_Q2_given_gpa_y_bg0 = compute_matrix(matrix_p_y_gpa_q2_bg0, matrix_p_y_gpa_bg0, matrix_p_q2_gpa_bg0)
            matrix_Q2_given_gpa_y_bg1 = compute_matrix(matrix_p_y_gpa_q2_bg1, matrix_p_y_gpa_bg1, matrix_p_q2_gpa_bg1)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [matrix_Q2_given_gpa_y_bg0, matrix_Q2_given_gpa_y_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(f'P(Q2 > 0 | GPA, Y): Bayes Rule and Extrapolation -- {title}')
                 # Annotating the heatmap
                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif val < 0:
                        ax.text(j, i, '<0', ha='center', va='center', color='blue', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)

                # Setting labels for the Y axis as bin_edges
                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)

                # Setting labels for the X axis as bin_edges_GPA
                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
                ax.xaxis.set_tick_params(width=0.5)
                ax.yaxis.set_tick_params(width=0.5)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

            # Adjust layout
            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(Q2 | GPA, Y)_extrapo.png'))
            plt.close()


            # ### Size Threshold version

            def calculate_size_matrix(data, unique_mapping_df, bin_edges, bin_edges_GPA):
                aboveGPA_data = data[data['gpa'] >= data['GPA_cutoff']]
                aboveGPA_data = unique_mapping_df.merge(aboveGPA_data, on='Y_bin_index', how='left')
                grouped_data = aboveGPA_data.groupby(['Y_bin_index', 'GPA_bin'])

                counts_per_group = grouped_data.size()
                counts_per_group_df = counts_per_group.reset_index(name='Size')
                count_matrix = counts_per_group_df.pivot('Y_bin_index', 'GPA_bin', 'Size')
                return count_matrix

            # Assuming bin_edges_GPA, bin_edges, GPA_cutoff, unique_mapping_df, data_bg0, and data_bg1 are defined
            count_matrix_bg0 = calculate_size_matrix(data_bg0, unique_mapping_df, bin_edges, bin_edges_GPA)
            count_matrix_bg1 = calculate_size_matrix(data_bg1, unique_mapping_df, bin_edges, bin_edges_GPA)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [count_matrix_bg0, count_matrix_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
                ax.set_title(f'Size of (GPA(%), Y) Cell for GPA >= GPA_cutoff -- {title}')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix.values):
                    if val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    else:
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center', color='black', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_size_gpa_gtcutoff_y.png'))
            plt.close()

            # Size 10
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            for ax, matrix, title in zip(axs, [count_matrix_bg0, count_matrix_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
                ax.set_title(f'Size of (GPA(%), Y) Cell with Threshold 10 -- {title}')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix.values):
                    if val < 10:
                        ax.text(j, i, '<10', ha='center', va='center', color='red', fontsize=8)
                    else:
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center', color='black', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_size_gpa_y_10.png'))
            plt.close()


            # Size 50
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            for ax, matrix, title in zip(axs, [count_matrix_bg0, count_matrix_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
                ax.set_title(f'Size of (GPA(%), Y) Cell with Threshold 50 -- {title}')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix.values):
                    if val < 50:
                        ax.text(j, i, '<50', ha='center', va='center', color='red', fontsize=8)
                    else:
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center', color='black', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_size_gpa_y_50.png'))
            plt.close()


            # ### Q2>0 Sub-sample


            def calculate_size_matrix_Q2(data_filtered, bin_edges, bin_edges_GPA):
                # Create a copy to avoid modifying the original DataFrame
                aboveGPA_data_Q2 = data_filtered[data_filtered['gpa'] >= data_filtered['GPA_cutoff']].copy()
                aboveGPA_data_Q2['Y_bin'] = pd.cut(aboveGPA_data_Q2['Y'], bins=bin_edges, include_lowest=True)

                grouped_data_filtered = aboveGPA_data_Q2.groupby(['Y_bin', 'GPA_bin_Q2'])

                counts_per_group = grouped_data_filtered.size()
                counts_per_group_df = counts_per_group.reset_index(name='Size')
                count_matrix_Q2 = counts_per_group_df.pivot('Y_bin', 'GPA_bin_Q2', 'Size')
                return count_matrix_Q2

            # Assuming bin_edges_GPA, bin_edges, GPA_cutoff, data_filtered_bg0, and data_filtered_bg1 are defined
            count_matrix_Q2_bg0 = calculate_size_matrix_Q2(data_filtered_bg0, bin_edges, bin_edges_GPA)
            count_matrix_Q2_bg1 = calculate_size_matrix_Q2(data_filtered_bg1, bin_edges, bin_edges_GPA)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [count_matrix_Q2_bg0, count_matrix_Q2_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
                ax.set_title(f'Size of (GPA(%), Y) Cell for Q2>0 & GPA > GPA_cutoff -- {title}')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix.values):
                    if val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    else:
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center', color='black', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_size_gpa_gtcutoff_y_Q2.png'))
            plt.close()


            # In[33]:


            def calculate_size_matrix_Q2(data_filtered, bin_edges, bin_edges_GPA):
                aboveGPA_data_Q2 = data_filtered[data_filtered['gpa'] >= data_filtered['GPA_cutoff']].copy()
                aboveGPA_data_Q2['Y_bin'] = pd.cut(aboveGPA_data_Q2['Y'], bins=bin_edges, include_lowest=True)
                grouped_data_filtered = aboveGPA_data_Q2.groupby(['Y_bin', 'GPA_bin_Q2'])

                counts_per_group = grouped_data_filtered.size()
                counts_per_group_df = counts_per_group.reset_index(name='Size')
                count_matrix_Q2 = counts_per_group_df.pivot('Y_bin', 'GPA_bin_Q2', 'Size')
                return count_matrix_Q2

            # Assuming bin_edges_GPA, bin_edges, GPA_cutoff, data_filtered_bg0, and data_filtered_bg1 are defined
            count_matrix_Q2_bg0 = calculate_size_matrix_Q2(data_filtered_bg0, bin_edges, bin_edges_GPA)
            count_matrix_Q2_bg1 = calculate_size_matrix_Q2(data_filtered_bg1, bin_edges, bin_edges_GPA)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [count_matrix_Q2_bg0, count_matrix_Q2_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
                ax.set_title(f'Size of (GPA(%), Y) Cell with Threshold 100 for Q2>0 & GPA > GPA_cutoff -- {title}')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix.values):
                    if val < 100:
                        ax.text(j, i, '<100', ha='center', va='center', color='red', fontsize=8)
                    else:
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center', color='black', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            plt.tight_layout
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_size_gpa_y_100_Q2.png'))
            plt.close()


            # #### Use values in matrix_Q2_given_gpa_y if size < size threshold; Use values in matrix_Q2_given_gpa_y_didata if size >= size threshold

            def apply_threshold_and_create_matrix(original_matrix, count_matrix, matrix_Q2_given_gpa_y, threshold=10):
                matrix_with_threshold = original_matrix.copy()
                for i in range(count_matrix.shape[0]):
                    for j in range(count_matrix.shape[1]):
                        if count_matrix.iloc[i, j] < threshold:
                            matrix_with_threshold.iat[i, j] = matrix_Q2_given_gpa_y.iat[i, j]
                return matrix_with_threshold

            matrix_Q2_given_gpa_y_Size10_bg0 = apply_threshold_and_create_matrix(matrix_Q2_given_gpa_y_didata_bg0, count_matrix_bg0, matrix_Q2_given_gpa_y_bg0)
            matrix_Q2_given_gpa_y_Size10_bg1 = apply_threshold_and_create_matrix(matrix_Q2_given_gpa_y_didata_bg1, count_matrix_bg1, matrix_Q2_given_gpa_y_bg1)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [matrix_Q2_given_gpa_y_Size10_bg0, matrix_Q2_given_gpa_y_Size10_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(f'P(Q2 > 0 | GPA, Y) with Size Threshold 10 -- {title}')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(Q2 | GPA, Y)_Size10.png'))
            plt.close()


            # In[35]:


            def apply_threshold_and_create_matrix(original_matrix, count_matrix, matrix_Q2_given_gpa_y, threshold=50):
                matrix_with_threshold = original_matrix.copy()
                for i in range(count_matrix.shape[0]):
                    for j in range(count_matrix.shape[1]):
                        if count_matrix.iloc[i, j] < threshold:
                            matrix_with_threshold.iat[i, j] = matrix_Q2_given_gpa_y.iat[i, j]
                return matrix_with_threshold

            matrix_Q2_given_gpa_y_Size50_bg0 = apply_threshold_and_create_matrix(matrix_Q2_given_gpa_y_didata_bg0, count_matrix_bg0, matrix_Q2_given_gpa_y_bg0)
            matrix_Q2_given_gpa_y_Size50_bg1 = apply_threshold_and_create_matrix(matrix_Q2_given_gpa_y_didata_bg1, count_matrix_bg1, matrix_Q2_given_gpa_y_bg1)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [matrix_Q2_given_gpa_y_Size50_bg0, matrix_Q2_given_gpa_y_Size50_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(f'P(Q2 > 0 | GPA, Y) with Size Threshold 50 -- {title}')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(Q2 | GPA, Y)_Size50.png'))
            plt.close()


            # ### $P(S > 0 \mid \text{GPA(%), Y, Q2}>0)$

            # #### 1. PMF from Direct Data
            # Without assuming normality, use the empirical probability mass function (PMF) directly from data. The empirical PMF approach counts the number of occurrences where 
            # S>0 and divides it by the total number of occurrences in each (GPA_bin_Q2, Y_bin) cell.

            # In[36]:


            def calculate_conditional_prob_S(data_filtered, unique_mapping_df, bin_edges, bin_edges_GPA):
                aboveGPA_data_Q2 = data_filtered[data_filtered['gpa'] >= data_filtered['GPA_cutoff']].copy()
                aboveGPA_data_Q2['Y_bin'] = pd.cut(aboveGPA_data_Q2['Y'], bins=bin_edges, include_lowest=True)

                grouped_data_filtered = aboveGPA_data_Q2.groupby(['GPA_bin_Q2', 'Y_bin'])
                total_counts = grouped_data_filtered.size()
                count_S_gt_0 = aboveGPA_data_Q2[aboveGPA_data_Q2['S'] > 0].groupby(['GPA_bin_Q2', 'Y_bin']).size()
                prob_S_gt_0_given_GPA_Y = count_S_gt_0.div(total_counts)

                pivot_results_S_Q2 = prob_S_gt_0_given_GPA_Y.reset_index().pivot_table(
                    index='Y_bin', 
                    columns='GPA_bin_Q2',
                    values=0,  # Explicitly specifying the column name for values
                    dropna=False
                )

                unique_mapping_df_str = unique_mapping_df
                unique_mapping_df_str['Y_bin'] = unique_mapping_df['Y_bin'].astype(str)
                # Convert 'Y_bin' intervals to string for mapping
                pivot_results_S_Q2.index = pivot_results_S_Q2.index.astype(str)
                mapping_dict = unique_mapping_df_str.set_index('Y_bin')['Y_bin_index'].to_dict()
                pivot_results_S_Q2.index = pivot_results_S_Q2.index.map(mapping_dict)
                return pivot_results_S_Q2

            # Assuming bin_edges_GPA, bin_edges, GPA_cutoff, unique_mapping_df, data_filtered_bg0, and data_filtered_bg1 are defined
            matrix_Sgt0_given_gpa_y_Q2_didata_bg0 = calculate_conditional_prob_S(data_filtered_bg0, unique_mapping_df, bin_edges, bin_edges_GPA)
            matrix_Sgt0_given_gpa_y_Q2_didata_bg1 = calculate_conditional_prob_S(data_filtered_bg1, unique_mapping_df, bin_edges, bin_edges_GPA)

            # Plotting subfigures
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Common settings for both plots
            for ax, matrix, title in zip(axs, [matrix_Sgt0_given_gpa_y_Q2_didata_bg0, matrix_Sgt0_given_gpa_y_Q2_didata_bg1], ['B=0', 'B=1']):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(f'P(S > 0 | GPA >= GPA_cutoff, Y, Q2 > 0) from Local Approximation -- {title}')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(S > 0 | GPA, Y, Q2)_localapprox.png'))
            plt.close()


            # #### 3. From Model using Bayes Rule 
            # $P(S>0\mid G P A(\%), Y, Q 2>0)=\frac{P(Y \mid S>0, G P A(\%), Q 2>0) \times P(S>0\mid G P A(\%), Q 2>0)}{P(Y \mid G P A(\%), Q 2>0)}$

            # ##### 3.1 $P(Y \mid S>0, G P A(\%), Q 2>0)$ from model: equation (2)
            # To get matrix_p_y_gpa_Sgt0_Q2

            # In[40]:

            def calculate_probabilities(data, bin_edges, sigma_2_global):
                # Filter data where S > 0 and Q2 > 0
                data_S_Q2 = data[(data['S'] > 0) & (data['Q2'] > 0)]
                gpa_bins = data_S_Q2['GPA_bin'].dropna().unique()

                results = pd.DataFrame(columns=['GPA_bin', 'Y_bin_index', 'Y_bin', 'P_Y_given_GPA_S_Q2'])
                for g in gpa_bins:
                    subset = data_S_Q2[data_S_Q2['GPA_bin'] == g]
                    mu_Y = subset['Y_s_GPA_Q2_predicted'].mean()

                    for i, (Y_lower, Y_upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                        if not np.isnan(sigma_2_global):
                            prob = norm.cdf((Y_upper - mu_Y) / sigma_2_global) - norm.cdf((Y_lower - mu_Y) / sigma_2_global)
                        else:
                            prob = np.nan
                        results = results.append({'GPA_bin': g, 'Y_bin_index': i, 'Y_bin': f'{Y_lower}-{Y_upper}', 'P_Y_given_GPA_S_Q2': prob}, ignore_index=True)
                return results.pivot(index='Y_bin_index', columns='GPA_bin', values='P_Y_given_GPA_S_Q2')

            def plot_heatmap(matrix, ax, title, bin_edges, bin_edges_GPA):
                cax = ax.imshow(matrix, cmap='Purples', aspect='auto', vmax=1, vmin=0)
                ax.set_title(title)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            # Calculate probabilities for background = 0 and background = 1
            matrix_p_y_gpa_Sgt0_Q2_bg0 = calculate_probabilities(data_bg0, bin_edges, sigma_2)
            matrix_p_y_gpa_Sgt0_Q2_bg1 = calculate_probabilities(data_bg1, bin_edges, sigma_2)

            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Plot for background = 0
            plot_heatmap(matrix_p_y_gpa_Sgt0_Q2_bg0, axs[0], 'P(Y | GPA, S > 0, Q2 > 0) from Extrapolation -- B=0', bin_edges, bin_edges_GPA)

            # Plot for background = 1
            plot_heatmap(matrix_p_y_gpa_Sgt0_Q2_bg1, axs[1], 'P(Y | GPA, S > 0, Q2 > 0) from Extrapolation -- B=1', bin_edges, bin_edges_GPA)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(Y | Q2, GPA,S>0)_model.png'))
            plt.close()


            # In[41]:


            # # Calculating and printing column sums
            # column_sums = matrix_p_y_gpa_Sgt0_Q2_bg1.sum(axis=0)
            # print("Column Sums:")
            # print(column_sums)


            # ##### 3.2 $P(S>0\mid G P A(\%), Q 2>0)$: matrix_p_s_gt0_gpa_Q2_model
            # $P(S>0\mid G P A, Q 2>0)$ from model: consider the following linear probability model applied to the $Q2>0$ sub-sample:
            #     \begin{equation}\label{eq4_LPM_S}
            #         \boldsymbol{1(S>0)=\underbrace{\alpha_4+\beta_4 \times G P A (\%)}_{P(S>0\mid G P A, Q 2>0) \textbf{ from model}} +\epsilon_4}
            #     \end{equation}

            # Function to fit a linear model and estimate probabilities
            def LPM_seperate_bg(data, bin_edges_GPA, subgroup):
                # Create a copy of the data to avoid modifying the original DataFrame
                data_copy = data.copy()

                # Convert S to binary (1 if S > 0, else 0)
                data_copy['S_binary'] = (data_copy['S'] > 0).astype(int)

                # Define the predictor and add a constant
                X = sm.add_constant(data_copy['percentile_GPA_applyQ1'])
                y = data_copy['S_binary']
                eq3_admitted_model = sm.OLS(y, X).fit()

                coefficients_eq3_admitted = eq3_admitted_model.params 
                p_values = eq3_admitted_model.pvalues

                # Format the equation string
                equation = f"Y = {coefficients_eq3_admitted['const']:.2f} "
                for variable in coefficients_eq3_admitted.index[1:]:
                    equation += f"+ ({coefficients_eq3_admitted[variable]:.2f})*{variable} "
                    equation += f"(p={p_values[variable]:.2g}) "

                print(f"\nEquation (3) for subgroup b={subgroup}:")
                print(equation)

                data_copy['Estimated_Probability'] = eq3_admitted_model.predict(X)

                # Grouping by GPA bins and calculating the average probabilities
                grouped_data = data_copy.groupby('GPA_bin_Q2')['Estimated_Probability'].mean().reset_index()

                return grouped_data, eq3_admitted_model

            def LPM_regressor_bg(data, bin_edges_GPA):
                # Create a copy of the data to avoid modifying the original DataFrame
                data_copy = data.copy()

                # Convert S to binary (1 if S > 0, else 0)
                data_copy['S_binary'] = (data_copy['S'] > 0).astype(int)

                # Define the predictor and add a constant
                X = sm.add_constant(data_copy[['percentile_GPA_applyQ1', 'background']])
                y = data_copy['S_binary']
                eq3_admitted_model = sm.OLS(y, X).fit()

                coefficients_eq3_admitted = eq3_admitted_model.params 
                p_values = eq3_admitted_model.pvalues

                # Format the equation string
                equation = f"Y = {coefficients_eq3_admitted['const']:.2f} "
                for variable in coefficients_eq3_admitted.index[1:]:
                    equation += f"+ ({coefficients_eq3_admitted[variable]:.2f})*{variable} "
                    equation += f"(p={p_values[variable]:.2g}) "

                print(f"\nEquation (3) with regressor b:")
                print(equation)

                data_copy['Estimated_Probability'] = eq3_admitted_model.predict(X)

                # Grouping by GPA bins and calculating the average probabilities
                grouped_data = data_copy.groupby('GPA_bin_Q2')['Estimated_Probability'].mean().reset_index()

                return grouped_data, eq3_admitted_model


            grouped_data_bg0, eq3_admitted_model_bg0 = LPM_seperate_bg(data_filtered_bg0, bin_edges_GPA,0)
            grouped_data_bg1, eq3_admitted_model_bg1 = LPM_seperate_bg(data_filtered_bg1, bin_edges_GPA,1)
            grouped_data, eq3_admitted_model = LPM_regressor_bg(data_filtered, bin_edges_GPA)

            def plot_estimated_probabilities(grouped_data, ax, bin_edges_GPA, title):
                ax.bar(grouped_data['GPA_bin_Q2'], grouped_data['Estimated_Probability'], color='grey')
                x_bin_labels = [f'{100*edge:.0f}%-{100*bin_edges_GPA[i+1]:.0f}%' for i, edge in enumerate(bin_edges_GPA[:-1])]
                ax.set_xticks(np.arange(len(x_bin_labels)))
                ax.set_xticklabels(x_bin_labels, rotation=45)
                ax.set_ylabel('P[S>0 | GPA, Q2>0]')
                ax.set_title(title)
                ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

            # Calculate the maximum probability value for the y-axis scale
            max_probability = max(grouped_data_bg0['Estimated_Probability'].max(), grouped_data_bg1['Estimated_Probability'].max())
            num_y_bins = len(bin_edges) - 1

            # Create subplots with the same y-axis scale
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            # Plot for background = 0
            plot_estimated_probabilities(grouped_data_bg0, axs[0], bin_edges_GPA, 'P[S>0 | GPA(%), Q2>0] from LPM -- B=0')
            axs[0].set_ylim(0, max_probability)  # Set y-axis limit

            # Plot for background = 1
            plot_estimated_probabilities(grouped_data_bg1, axs[1], bin_edges_GPA, 'P[S>0 | GPA(%), Q2>0] from LPM -- B=1')
            axs[1].set_ylim(0, max_probability)  # Set y-axis limit

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(S > 0 | GPA, Q2)_LPM.png'))
            plt.close()


            matrix_p_s_gt0_gpa_Q2_bg0 = pd.DataFrame([grouped_data_bg0['Estimated_Probability'].values] * num_y_bins)
            matrix_p_s_gt0_gpa_Q2_bg1 = pd.DataFrame([grouped_data_bg1['Estimated_Probability'].values] * num_y_bins)


            # ##### 3.3 $P(Y \mid G P A(\%), Q 2>0)$ from model: matrix_p_y_gpa_q2

            # ##### 3.4 BR on Model: $P(S>0 \mid G P A(\%), Y, Q 2>0)=\frac{P(Y \mid S>0, G P A(\%), Q 2>0) \times P(S>0 \mid G P A(\%), Q 2>0)}{P(Y \mid G P A(\%), Q 2>0)}$
            # matrix_p_y_gpa_Sgt0_Q2 * matrix_p_s_gt0_gpa_Q2 / matrix_p_y_gpa_q2

            # In[43]:


            # Function to calculate the computed matrix for P(S > 0 | GPA, Y, Q2>0)
            def calculate_computed_matrix(matrix_p_y_gpa_Sgt0_Q2, matrix_p_s_gt0_gpa_Q2, matrix_p_y_gpa_q2):
                computed_matrix = pd.DataFrame(np.zeros(matrix_p_y_gpa_Sgt0_Q2.shape))
                for i in range(computed_matrix.shape[0]):
                    for j in range(computed_matrix.shape[1]):
                        if matrix_p_y_gpa_q2.iloc[i, j] != 0:
                            value = (matrix_p_y_gpa_Sgt0_Q2.iloc[i, j] * matrix_p_s_gt0_gpa_Q2.iloc[i, j]) / matrix_p_y_gpa_q2.iloc[i, j]
                            computed_matrix.iloc[i, j] = value
                return computed_matrix

            computed_matrix_bg0 = calculate_computed_matrix(matrix_p_y_gpa_Sgt0_Q2_bg0, matrix_p_s_gt0_gpa_Q2_bg0, matrix_p_y_gpa_q2_bg0)
            computed_matrix_bg1 = calculate_computed_matrix(matrix_p_y_gpa_Sgt0_Q2_bg1, matrix_p_s_gt0_gpa_Q2_bg1, matrix_p_y_gpa_q2_bg1)

            # Function to plot a heatmap
            def plot_heatmap(matrix, ax, title, bin_edges, bin_edges_GPA):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(title)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Plot for background = 0
            plot_heatmap(computed_matrix_bg0, axs[0], 'P(S > 0 | GPA(%), Y, Q2>0) from Extrapolation -- B=0', bin_edges, bin_edges_GPA)

            # Plot for background = 1
            plot_heatmap(computed_matrix_bg1, axs[1], 'P(S > 0 | GPA(%), Y, Q2>0) from Extrapolation -- B=1', bin_edges, bin_edges_GPA)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(S > 0 | GPA, Y, Q2)_extrapo.png'))
            plt.close()

            # Save computed matrices under new names
            matrix_Sgt0_given_gpa_y_Q2_bg0 = computed_matrix_bg0
            matrix_Sgt0_given_gpa_y_Q2_bg1 = computed_matrix_bg1


            # #### 4. Threshold 100: above threshold: model; below threshold: data

            # In[44]:


            # Function to create a matrix with size threshold 100
            def apply_size_threshold(matrix_Sgt0_given_gpa_y_Q2_didata, count_matrix, matrix_Sgt0_given_gpa_y_Q2, threshold=100):
                matrix_with_threshold = matrix_Sgt0_given_gpa_y_Q2_didata.copy()
                for i in range(count_matrix.shape[0]):
                    for j in range(count_matrix.shape[1]):
                        if count_matrix.iloc[i, j] < threshold:
                            matrix_with_threshold.iat[i, j] = matrix_Sgt0_given_gpa_y_Q2.iat[i, j]
                return matrix_with_threshold

            matrix_Sgt0_given_gpa_y_Q2_Size100_bg0 = apply_size_threshold(matrix_Sgt0_given_gpa_y_Q2_didata_bg0, count_matrix_Q2_bg0, matrix_Sgt0_given_gpa_y_Q2_bg0)
            matrix_Sgt0_given_gpa_y_Q2_Size100_bg1 = apply_size_threshold(matrix_Sgt0_given_gpa_y_Q2_didata_bg1, count_matrix_Q2_bg1, matrix_Sgt0_given_gpa_y_Q2_bg1)

            # Function to plot a heatmap
            def plot_heatmap(matrix, ax, title, bin_edges, bin_edges_GPA):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(title)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')


            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Plot for background = 0
            plot_heatmap(matrix_Sgt0_given_gpa_y_Q2_Size100_bg0, axs[0], 'P(S > 0 | GPA(%), Y, Q2>0, B=0) with Size Threshold 100', bin_edges, bin_edges_GPA)

            # Plot for background = 1
            plot_heatmap(matrix_Sgt0_given_gpa_y_Q2_Size100_bg1, axs[1], 'P(S > 0 | GPA(%), Y, Q2>0, B=1) with Size Threshold 100', bin_edges, bin_edges_GPA)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(S > 0 | GPA, Y, Q2)_Size100.png'))
            plt.close()


            # ## $P(S>0, Q 2 >0 \mid G P A,Y)$
            # $$
            # P(S>0, Q 2 >0 \mid G P A,Y)=P(S>0 \mid G P A,Y, Q 2>0) \times P(Q 2>0 \mid G P A,Y)
            # $$


            # ### (2) P(S>0, Q 2 >0 | G P A,Y)_didata

            # Function to calculate the matrix for P(S > 0, Q2 > 0 | GPA, Y)
            def calculate_combined_probability(matrix_Sgt0_given_gpa_y, matrix_Q2_given_gpa_y):
                return matrix_Sgt0_given_gpa_y * matrix_Q2_given_gpa_y

            matrix_Sgt0_Q2_given_gpa_y_didata_bg0 = calculate_combined_probability(matrix_Sgt0_given_gpa_y_Q2_didata_bg0, matrix_Q2_given_gpa_y_didata_bg0)
            matrix_Sgt0_Q2_given_gpa_y_didata_bg1 = calculate_combined_probability(matrix_Sgt0_given_gpa_y_Q2_didata_bg1, matrix_Q2_given_gpa_y_didata_bg1)

            # Function to plot a heatmap
            def plot_heatmap(matrix, ax, title, bin_edges, bin_edges_GPA):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(title)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Plot for background = 0
            plot_heatmap(matrix_Sgt0_Q2_given_gpa_y_didata_bg0, axs[0], 'P(S > 0, Q2 > 0 | GPA, Y) from Local Approximation -- B=0', bin_edges, bin_edges_GPA)

            # Plot for background = 1
            plot_heatmap(matrix_Sgt0_Q2_given_gpa_y_didata_bg1, axs[1], 'P(S > 0, Q2 > 0 | GPA, Y) from Local Approximation -- B=1', bin_edges, bin_edges_GPA)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(S > 0, Q2 > 0 | GPA, Y)_localapprox.png'))
            plt.close()


            # ### (3) Model Predicted  P(S>0, Q 2 >0 | G P A,Y)

            # Function to calculate the matrix for P(S > 0, Q2 > 0 | GPA, Y)
            def calculate_combined_probability(matrix_Sgt0_given_gpa_y, matrix_Q2_given_gpa_y):
                return matrix_Sgt0_given_gpa_y * matrix_Q2_given_gpa_y

            matrix_Sgt0_Q2_given_gpa_y_bg0 = calculate_combined_probability(matrix_Sgt0_given_gpa_y_Q2_bg0, matrix_Q2_given_gpa_y_bg0)
            matrix_Sgt0_Q2_given_gpa_y_bg1 = calculate_combined_probability(matrix_Sgt0_given_gpa_y_Q2_bg1, matrix_Q2_given_gpa_y_bg1)

            # Function to plot a heatmap
            def plot_heatmap(matrix, ax, title, bin_edges, bin_edges_GPA):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(title)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Plot for background = 0
            plot_heatmap(matrix_Sgt0_Q2_given_gpa_y_bg0, axs[0], 'P(S > 0, Q2 > 0 | GPA, Y) from Extrapolation -- B=0', bin_edges, bin_edges_GPA)

            # Plot for background = 1
            plot_heatmap(matrix_Sgt0_Q2_given_gpa_y_bg1, axs[1], 'P(S > 0, Q2 > 0 | GPA, Y) from Extrapolation -- B=1', bin_edges, bin_edges_GPA)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(S > 0, Q2 > 0 | GPA, Y)_extrapo.png'))
            plt.close()


            # ### (4) P(S>0, Q 2 >0 | G P A,Y)_Size10

            # Function to calculate the matrix for P(S > 0, Q2 > 0 | GPA, Y)
            def calculate_combined_probability(matrix_Sgt0_given_gpa_y, matrix_Q2_given_gpa_y):
                return matrix_Sgt0_given_gpa_y * matrix_Q2_given_gpa_y

            matrix_Sgt0_Q2_given_gpa_y_Size10_bg0 = calculate_combined_probability(matrix_Sgt0_given_gpa_y_Q2_Size100_bg0, matrix_Q2_given_gpa_y_Size10_bg0)
            matrix_Sgt0_Q2_given_gpa_y_Size10_bg1 = calculate_combined_probability(matrix_Sgt0_given_gpa_y_Q2_Size100_bg1, matrix_Q2_given_gpa_y_Size10_bg1)

            # Function to plot a heatmap
            def plot_heatmap(matrix, ax, title, bin_edges, bin_edges_GPA):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(title)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Plot for background = 0
            plot_heatmap(matrix_Sgt0_Q2_given_gpa_y_Size10_bg0, axs[0], 'P(S > 0, Q2 > 0 | GPA, Y): Size 100 for P(S > 0 | GPA, Y, Q2 > 0); Size 10 for P(Q2 > 0 | GPA, Y) -- B=0', bin_edges, bin_edges_GPA)

            # Plot for background = 1
            plot_heatmap(matrix_Sgt0_Q2_given_gpa_y_Size10_bg1, axs[1], 'P(S > 0, Q2 > 0 | GPA, Y): Size 100 for P(S > 0 | GPA, Y, Q2 > 0); Size 10 for P(Q2 > 0 | GPA, Y) -- B=1', bin_edges, bin_edges_GPA)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(S > 0, Q2 > 0 | GPA, Y)_Size10.png'))
            plt.close()


            # ### (5) P(S>0, Q 2 >0 | G P A,Y)_Size50

            # Function to calculate the matrix for P(S > 0, Q2 > 0 | GPA, Y)
            def calculate_combined_probability(matrix_Sgt0_given_gpa_y, matrix_Q2_given_gpa_y):
                return matrix_Sgt0_given_gpa_y * matrix_Q2_given_gpa_y

            matrix_Sgt0_Q2_given_gpa_y_Size50_bg0 = calculate_combined_probability(matrix_Sgt0_given_gpa_y_Q2_Size100_bg0, matrix_Q2_given_gpa_y_Size50_bg0)
            matrix_Sgt0_Q2_given_gpa_y_Size50_bg1 = calculate_combined_probability(matrix_Sgt0_given_gpa_y_Q2_Size100_bg1, matrix_Q2_given_gpa_y_Size50_bg1)

            # Function to plot a heatmap
            def plot_heatmap(matrix, ax, title, bin_edges, bin_edges_GPA):
                cax = ax.imshow(matrix, cmap='Greys', aspect='auto', vmax=1, vmin=0)
                ax.set_title(title)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                for (i, j), val in np.ndenumerate(matrix):
                    if pd.isna(val):
                        ax.text(j, i, 'NaN', ha='center', va='center', color='red', fontsize=8)
                    elif val == 0:
                        ax.text(j, i, '0', ha='center', va='center', color='blue', fontsize=8)
                    elif 0 < val <= 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)
                    elif val > 1:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize=8)

                x_bin_labels = [f'{100*edge:.0f}%' for edge in bin_edges_GPA]
                x_ticks = np.arange(-0.5, len(x_bin_labels) - 0.5, 1)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_bin_labels, rotation=45)

                y_bin_labels = [f'{edge:.2f}' for edge in bin_edges]
                y_ticks = np.arange(-0.5, len(y_bin_labels) - 0.5, 1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_bin_labels)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='k')

                ax.set_xlabel('GPA(%) Bin')
                ax.set_ylabel('Y Bin')

            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            # Plot for background = 0
            plot_heatmap(matrix_Sgt0_Q2_given_gpa_y_Size50_bg0, axs[0], 'P(S > 0, Q2 > 0 | GPA, Y): Size 100 for P(S > 0 | GPA, Y, Q2 > 0); Size 50 for P(Q2 > 0 | GPA, Y) -- B=0', bin_edges, bin_edges_GPA)

            # Plot for background = 1
            plot_heatmap(matrix_Sgt0_Q2_given_gpa_y_Size50_bg1, axs[1], 'P(S > 0, Q2 > 0 | GPA, Y): Size 100 for P(S > 0 | GPA, Y, Q2 > 0); Size 50 for P(Q2 > 0 | GPA, Y) -- B=1', bin_edges, bin_edges_GPA)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_directory, f'Program_{program_id}_P(S > 0, Q2 > 0 | GPA, Y)_Size50.png'))
            plt.close()

            # Extract coefficients
            coefficients_eq1 = eq1_admitted_model.params
            coefficients_eq2 = eq2_admitted_model.params
            coefficients_eq3_bg0 = eq3_admitted_model_bg0.params
            coefficients_eq3_bg1 = eq3_admitted_model_bg1.params
            coefficients_eq3 = eq3_admitted_model.params

            # Create a DataFrame
            coefficients_table = pd.DataFrame({
                'Equtaion (1): Q1 Admitted': coefficients_eq1,
                'Equtaion (2): Q2 Admitted': coefficients_eq2,
                'Equtaion (3): Background 0': coefficients_eq3_bg0,
                'Equation (3): Background 1': coefficients_eq3_bg1,
                'Equation (3): Pool Background': coefficients_eq3
            }).round(2)
            
            excel_file_path = os.path.join(tables_directory, f'Program_{program_id}_coefficients_table.xlsx')
            coefficients_table.to_excel(excel_file_path)
            print(f"Coefficients Table saved to {excel_file_path}")

end_time = time.time()

elapsed_time_minutes = (end_time - start_time) / 60
print(f"Elapsed time: {elapsed_time_minutes:.2f} minutes")            