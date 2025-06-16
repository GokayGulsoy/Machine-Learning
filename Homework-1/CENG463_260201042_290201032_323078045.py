import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# function that computes count, missing percentage,
# and cardinality statistics which are common for
# both categorical and continuous features
def compute_common_statistics(df, column):
    count = len(df[column])
    num_of_missing_values = df[column].isna().sum()
    missing_percentage = (num_of_missing_values / count) * 100
    cardinality = len(pd.unique(df[column]))
    
    return count, missing_percentage ,cardinality


def create_DQ_report_for_categorical_features(dataset, categorical_columns, path_to_output_directory):
    # creating data quality report for categorical features
    df_categorical = dataset[categorical_columns]
    print("--First Five Entries of Dataframe representing categorical features--\n", df_categorical.head())
    DQR_categorical = pd.DataFrame(columns=['Feature', 'Count', '% Miss.', 'Card', 'Mode', 'Mode Freq.', 'Mode %', '2nd Mode', '2nd Mode Freq.', '2nd Mode %'])
        
    for categorical_column in df_categorical:    
        count, missing_percentage, cardinality = compute_common_statistics(df_categorical, categorical_column)
        mode = list(df_categorical[categorical_column].mode())[0]
        mode_frequency = len(df_categorical["GeneticRisk"][(df_categorical[categorical_column] == mode)])
        mode_percentage = (mode_frequency / count) * 100
        df_categorical_without_first_mode = df_categorical[categorical_column][(df_categorical[categorical_column] != mode)]
        second_mode = df_categorical_without_first_mode.mode()[0]
        second_mode_frequency = len(df_categorical[categorical_column][(df_categorical[categorical_column] == second_mode)])
        second_mode_percentage = (second_mode_frequency / count) * 100

        data_quality_report_entry = {'Feature': categorical_column, 'Count': count, '% Miss.': missing_percentage, 'Card': cardinality, 'Mode': mode, 'Mode Freq.': mode_frequency, 'Mode %': mode_percentage ,'2nd Mode': second_mode, '2nd Mode Freq.': second_mode_frequency, '2nd Mode %': second_mode_percentage}
        DQR_categorical.loc[len(DQR_categorical)] = data_quality_report_entry
  
    DQR_categorical.to_csv(path_to_output_directory + '\\output_DQR_Categorical.csv', index=False)
    
    
def create_DQ_report_for_continuous_features(dataset, continuous_columns, path_to_output_directory):
    # creating data quality report for continuous features
    df_continuous = dataset[continuous_columns]
    print("--First Five Entries of Dataframe representing continuous features--\n", df_continuous.head())
    DQR_continuous = pd.DataFrame(columns=['Feature', 'Count', '% Miss.', 'Card', 'Min.', '1st Qrt.', 'Mean', 'Median', '3rd Qrt.', 'Max.', 'Std. Dev.'])
   
    for continuous_column in df_continuous:
        count, missing_percentage, cardinality = compute_common_statistics(df_continuous, continuous_column)
        min_value = df_continuous[continuous_column].min()
        max_value = df_continuous[continuous_column].max()
        mean = df_continuous[continuous_column].mean()
        first_quartile, median, third_quartile = tuple(df_continuous[continuous_column].quantile([0.25, 0.50, 0.75]))
        standard_deviation = df_continuous[continuous_column].std()
        
        data_quality_report_entry = {'Feature': continuous_column, 'Count': count, '% Miss.': missing_percentage, 'Card': cardinality, 'Min.': min_value, '1st Qrt.': first_quartile, 'Mean': mean, 'Median': median, '3rd Qrt.': third_quartile, 'Max.': max_value, 'Std. Dev.': standard_deviation}
        DQR_continuous.loc[len(DQR_continuous)] = data_quality_report_entry
       
    DQR_continuous.to_csv(path_to_output_directory + '\\output_DQR_Continuous.csv', index=False)    


def generate_data_quality_reports(dataset, categorical_feature_columns, continuous_feature_columns, path_to_output_directory):
    create_DQ_report_for_categorical_features(dataset, categorical_feature_columns, path_to_output_directory)
    create_DQ_report_for_continuous_features(dataset, continuous_feature_columns, path_to_output_directory)
    
    
def create_visualizations_for_categorical_features(dataset, categorical_columns, path_to_output_directory):
     df_categorical = dataset[categorical_columns]
     target_feature = dataset["Diagnosis"]
     
     for categorical_column in df_categorical:
        if categorical_column == "Diagnosis":
             continue 
         
        grouped = dataset.groupby([categorical_column, target_feature]).size().unstack(fill_value=0)
        
        categories = grouped.index.tolist()
        diagnosis_0_counts = grouped[0].tolist()  
        diagnosis_1_counts = grouped[1].tolist()  

        x_values = np.arange(len(categories))
        width = 0.40  

        fig, axes = plt.subplots(figsize=(8,6))
        axes.bar(x_values - width/2, diagnosis_0_counts, width, label='Diagnosis 0')
        axes.bar(x_values + width/2, diagnosis_1_counts, width, label='Diagnosis 1')

        axes.set_xlabel(categorical_column)
        axes.set_ylabel('Count')
        axes.set_title(f'{categorical_column} vs Diagnosis')
        axes.set_xticks(x_values)
        axes.set_xticklabels(categories)
        axes.legend()

        fig_name = path_to_output_directory + "\\" + categorical_column + " vs Diagnosis.jpg"
        plt.savefig(fig_name)
     

def create_visualizations_for_continuous_features(dataset, continuous_columns, path_to_output_directory):
     df_continuous = dataset[continuous_columns]
     target_feature = "Diagnosis"
     
     for continuous_column in df_continuous:
         
        feature_values_when_target_0 = dataset[dataset[target_feature] == 0][continuous_column]
        feature_values_when_target_1 = dataset[dataset[target_feature] == 1][continuous_column]
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        axes[0].hist(feature_values_when_target_0, bins=15, color='Blue', edgecolor='black')
        axes[0].set_title(continuous_column + " vs Diagnosis 0")
        
        axes[1].hist(feature_values_when_target_1, bins=15, color='Red', edgecolor='black')
        axes[1].set_title(continuous_column + " vs Diagnosis 1")
        
        for ax in axes:    
            ax.set_xlabel(continuous_column)
            ax.set_ylabel('Count')    

        fig_name = path_to_output_directory + "\\" + continuous_column + " vs Diagnosis.jpg"
        plt.savefig(fig_name)
     

# function that generates visualizations 
# for both categorical and continuous features
def generate_plots(dataset, categorical_feature_columns, continuous_feature_columns, path_output_directory):
    create_visualizations_for_categorical_features(dataset, categorical_feature_columns, path_output_directory)
    create_visualizations_for_continuous_features(dataset, continuous_feature_columns, path_output_directory)
    
    
"""
 script can be executed as follows:
 python <script_name> <path_to_input.csv> <path_to_visualization_outputs>
"""
def run_analysis_script():
    if len(sys.argv) < 3:
        print("Two command line arguments must be provided !!")
    
    else: 
        print("------Analysis Started------")
        # taking command-line arguments
        path_to_csv_file = sys.argv[1]
        path_to_output_directory = sys.argv[2]
        
        # check if input.csv file exists 
        # if not raise an exception
        if not os.path.exists(path_to_csv_file):
            raise FileNotFoundError("Either input.csv does not exist or given path is wrong !!")
            
        # check if output directory exists
        # if not create output directory
        if not os.path.exists(path_to_output_directory):
            os.makedirs(path_to_output_directory)    
            
        # reading csv file to data frame
        dataset = pd.read_csv(path_to_csv_file)

        # creating two data frames, one for categorical
        # features and other for continuous features
        categorical_feature_columns = ["Gender","Smoking","GeneticRisk","CancerHistory","Diagnosis"]
        continuous_feature_columns = ["Age","BMI","PhysicalActivity","AlcoholIntake"]
        
        # generating data quality reports and plots to illustrate relationship
        # between each descriptive feature and target feature
        generate_data_quality_reports(dataset, categorical_feature_columns, continuous_feature_columns, path_to_output_directory)
        generate_plots(dataset, categorical_feature_columns, continuous_feature_columns, path_to_output_directory)
        print("------Analysis Completed Successfully------")
        
# run analysis
run_analysis_script()