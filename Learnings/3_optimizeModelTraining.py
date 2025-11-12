# OPTIMIZE MODEL TRAINING WITH AZURE MACHINE LEARNING

# 1. RUNNING A TRAINING SCRIPT WAS A COMMAND JOB IN AZURE ML
# You want your code to be scalable, repeatable and ready for automation 
# Notebooks are not ideal for this, rather scripts better fit production workloads as COMMAND JOBS

# 1.1 Converting a Notebook to a Script
# You will have to:
# a. Remove all nonessential code: example print(), df.describe() statements used to explore your data and variables - will also help reduce cost and compute time

# b. Refactor your code into functions: making your code easier to read and test it by using functions!
# Also create multiple smaller functions to be able to test parts of your code, e.g

# read and visualize the data
print("Reading data...")
df = pd.read_csv('diabetes.csv')
df.head()
# split data
print("Splitting data...")
X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# To:
def main(csv_file):
    # read data
    df = get_data(csv_file)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)
    
    return df

# function that splits the data
def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

# c. Test your script in the terminal: one simple way is to run the script in a terminal of Azure ML compute instance




# 1.2 Running a Script as a Command Job
# To run a script as a command job, you'll need to configure and submit the job - you use the COMMAND FUNCTION with Python SDK v2
#
from azure.ai.ml import command

# configure job - need to specify all these parameters
job = command(
    code="./src",                                                    # folder that includes the script to run
    command="python train.py",                                       # specifies which file to run
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",  # with necessary pacakges to be installed in compute before running command
    compute="aml-cluster",                                           # compute to use to run the command
    display_name="train-model",                                      # name of the individual job
    experiment_name="train-classification-model"                     # name of the experiment the job belongs to
    )

# Command Function and all Possible Parameters: https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml 

# Once job is configured, you can submit it, which will initiate the job and run the script:
# submit job
returned_job = ml_client.create_or_update(job)


# 1.3 Using Parameters in a Command Job
# You can increase flexibility of your scripts by using paramaeters - e.g you can want to use the same script to train a model  on different datasets / using various hyperparameter values
# You must use ARGPARSE LIBRARY to read arguments passed to the script and assign them to variables

# E.g this script reads an arguments named training_data 
# import libraries
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main(args):
    # read data
    df = get_data(args.training_data)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)
    
    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":

    # parse args
    args = parse_args()

    # run main function
    main(args)

# Each parameter you expect should be defined in the script, you can specify type of value you expect or set a default value.

# Then pass the parameter values to a script

python train.py --training_data diabetes.csv # or any other file you want, may be in AzureML workspace

# then when confirming the job:
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py --training_data diabetes.csv",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-model",
    experiment_name="train-classification-model"
    )




# 2. TRACK MODEL TRAINING WITH MLFLOW IN JOBS
# When you have created a machine learning model job, and created a training script, the script is used to retrain the model every month when new data is collected
# You will want to monitor the models performance over time, whether new data benefits the model or there is drift, etc
# You can use MLflow to track models as scripts. 
# MLFLOW is an open-source platform that helps you track model metrics and artifacts across platforms and is intergrated with AzureML
# You can run the training scripts locally or in the cloud, then review model metrics and artifacts in workspace to compare runs and decide next step




# 2.1 Track Metrics With MLflow
# MLFlow is designed to manage the complete machine learning lifecycle. There are two options to track ml jobs with MLFlow:
# a. Enable autologging using mlflow.autolog()
# b. Use logging function to track custom metrics using mlflow.log_*


# To setup MLFlow in environment
# mlflow and azure-mlflow pip packages need to be installed on the compute executing the script
# You can create an environment by referring to a YAML  file that describes the conda env: yml file:
name: mlflow-env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - pip:
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - mlflow
    - azureml-mlflow

# Autologging is supported by follwoing libraries: scikit-learn, tensorflow and keras, xgboost, lightgbm, spark, fastai, pytorch
# To enable autologging run script:
import mlflow
mlflow.autolog()


# Depending on the type of value you want to log, use MLFlow command to store the metric with the experiment run

# mlflow.log_param(): Log single key-value parameter. Use this function for an input parameter you want to log.
# mlflow.log_metric(): Log single key-value metric. Value must be a number. Use this function for any output you want to store with the run.
# mlflow.log_artifact(): Log a file. Use this function for any plot you want to log, save as image file first.



# 2.2 View Metrics and Evaluate Models 
# You can review metrics in Azure ML studio, OR retrieve runs and metrics with MLflow 

# To view the metrics through an intuitive user interface, you can:
# Open the Studio by navigating to https://ml.azure.com
# Find your experiment run and open it to view its details
# In the Details tab, all logged parameters are shown under Params
# Select the Metrics tab and select the metric you want to explore.
# Any plots that are logges as artifacts can be found under Images
# The model assest that can be used to register and deploy the model are stored in the models folder under Outputs + logs

# Retriving Metrics with MLflow in a Notebook
# Get all active experiments in the workspace using MLflow
experiments = mlflow.search_experiments(max_results=2)
for exp in experiments:
    print(exp.name)

# Retrieve archived experiments 
from mlflow.entities import ViewType

experiments = mlflow.search_experiments(view_type=ViewType.ALL)
for exp in experiments:
    print(exp.name)

# Retrieve a specific experiment
exp = mlflow.get_experiment_by_name(experiment_name)
print(exp)


# Retrieving Runs
# MLflow allows you to search for runs inside of any experiment, with experiment id or name. e.g
mlflow.search_runs(exp.experiment_id)

# You can use search_all_experiments=True if you want to search across all the experiments in the workspace.

# By default, experiments are ordered descending by start_time, which is the time the experiment was queued in Azure Machine Learning. However, you can change this default by using the parameter order_by.

# Search runs with MLflow:https://mlflow.org/docs/latest/search-runs.html






# 3. PERFORM HYPERPARAMETER TUNING WITH AZURE MACHINE LEARNING
# Hyperparameter tuning is accomplished by training the multiple models, using the same algorithm and training data
# but different hyperparameter values. In Azure ML, you can tune hyperparameters by submitting a script as a sweep job. 
# A sweep job will run a trial for each hyperparameter combination to be tested.

# 3.1 Defining a Search Space
# The set of hyperparameter values tried during hyperparameter tuning is known as the search space. The range of possible values that can be chosen depends on the type of hyperparameter

# Discrete Hyperparameters - You select the value from a particular finite set of discrete value possibilities. You define a seach space for a discrete parameter using a Choice from a list of explicit values, which you can define as a 
# Python list (Choice(values = [10,20,30])), a range (Choice(values = range(1,10))) or 
# An arbitrary set of comma-separated values (Choice(values=(30,50,100))).
# QUniform(min_value, max_value, q) : returns a value like round(Uniform(min_value,max_value)/q)*q
# QLogUniform(min_value, max_value, q) : returns a value like round(exp(Uniform(min_value,max_value) )/q)*q
# QNormal(mu, sigma, q) : returns a value like round(Normal(mu,sigma)/q)*q
# QLogNormal(mu, sigma, q) : returns a value like round(exp(Normal(mu,sigma))/q)*q

# Continuous Hyperparameters -  Some hyperparameters are continuous, resulting in an infinite number of possibilities. You can use the following distribution types:
# Uniform(min_value,max_value) : returns a value uniformly distributed between min_value and max_value
# LogUniform(min_value,max_value) : returns a value drawn according to exp(Uniform(min_value, max_value)) so that the logarithm of the return value is uniformly distributed
# Normal(mu,sigma) : returns a real value thats normally distributed with the mean mu and std sigma
# LogNormal(mu,sigma) : returns a value drawn according to exp(Normal(mu,sigma)) so that the logarithm of the return value is normally distributed

# Defining a Search Space - to define a search space for hyperparameter tuning, create a dictionary with the appropriate parameter expression for each named hyperparameter.
# For e.g. the following search space indicates that the batch_size hyperparameter can have values 16,32,64 and the learning_rate hyperparameter can have any value from a normal distribution with a mean of 10 and std of 3.

from azure.ai.ml.sweep import Choice, Normal
command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Normal(mu=10, sigma=3),)



# 3.2 Configuring a Sampling Method
# The specific values used in a hyperparamter tuning run/ sweep job, depend on the type of sampling used. There are three main methods of sampling methods:

# Grid Sampling: Tries every possible combination of parameters in the search space
# Can only be applied when all hyperparameters are discrete. E.g:
from azure.ai.ml.sweep import Choice
command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),
    learning_rate=Choice(values=[0.01, 0.1, 1.0]),)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "grid",
    ...
)

# Random Sampling: Randomly chooses values from the search space
# Value of hyperparameter can be a mix of discrete and continuous values. E.g:
# from azure.ai.ml.sweep import Normal, Uniform

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),   
    learning_rate=Normal(mu=10, sigma=3),)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "random",
    ...)

# Sobol: a variation of random sampling where it adds a seed to random sampling to make the results of the sweep job reproducible
# When you add a seed, the sweep job can be reproduced and the search space distribution is spread more evenly.E.g

from azure.ai.ml.sweep import RandomSamplingAlgorithm

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = RandomSamplingAlgorithm(seed=123, rule="sobol"),
    ...)


# Bayesian Sampling: Chooses new values based on previous results
# Chooses hyperparameter values based on the bayesian optimization, which tries to select parameter combinations that will result in improved performance from the previous selection.
# You can only use bayesian sampling with Choice, Uniform and Quniform parameter expressions.
# from azure.ai.ml.sweep import Uniform, Choice

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Uniform(min_value=0.05, max_value=0.1),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "bayesian",
    ...
)

# 3.3 Configuring Early Termination
# Hyperparameter tuning helps you fine-tune your model and select best performance, however, this can be a never-ending conquest, you have to consider time and expense of testing new hyperparameter values to find a model that may perform better. If a sweep job  does no result in significantly better model, you may want to stop the sweep job and used the best performance thus far.
# When configuring an Azure ML sweep job, you can set maximum number of trials
# Better you can use EARLY TERMINATION POLICY: Where you stop a sweep job when newer models dont produce significantly better results.

# When to use Early Termination Policy
# This depends on the search space and sampling method you are working with
# For example, grid sampling method over a discrete search space with max six trials/models may not need this
# Early termination policy would be beneficial when working with continuous hyperparameters in your search space, as they present an unlimited number of possible values to choose from. 
# So you will most likely use it when working with continuous hyperparameters and a random or bayesian sampling method

# Configuring Early Termination Policy
# There are two main parameters when you choose to use an early termination policy:
# evaluation_interval: specifies at which interval you want the policy to be evaluated, everytime the primary metric is logged for a trial counts as an interval
# delay_evaluation: specifies when to start evaluating the policy, this parameter allows for at least a minimum of trials to complete without an early termination policy affecting them

# To determine the EXTENT to which a model should perform better than previous trials, there are three options for early termination:

# Bandit Policy
# Uses a slack_factor (relative) or slack_amount (absolute), any new model must perform within the slack range of the best performing model 
# You can use a bandit policy to stop a trial if the target performance metric underperforms the best trial so far by a specified margin
# For example : 
from azure.ai.ml.sweep import BanditPolicy

sweep_job.early_termination = BanditPolicy(
    slack_amount = 0.2, 
    delay_evaluation = 5, 
    evaluation_interval = 1)

# If primary metric is accuracy, after 5 trials, the best model has accuracy of 0.9, any new model needs to perform better than 0.9-0.2 = 0.7. 
# If accuracy is higher than 0.7, sweep job continues, else if less, it terminates sweep job.


# Median Stopping Policy
# Uses the median of the averages of the primary metric, any new model must perform better than the median.
# It abandons trails where the target performance metric is worse than the median of the running averages of all trials. E.g:
from azure.ai.ml.sweep import MedianStoppingPolicy

sweep_job.early_termination = MedianStoppingPolicy(
    delay_evaluation = 5, 
    evaluation_interval = 1)
# When median accuracy is 0.82, sweep job will continue if accuracy is higher, if lower, the policy will stop the sweep job

# Truncation Selection Policy
# Uses a truncation_percentage, whuch is the percentage of the lowest performing trials, any new model must perform better than the lowest performing trials.
# Lowest performing X% of trials at each evaluation interval:
from azure.ai.ml.sweep import TruncationSelectionPolicy

sweep_job.early_termination = TruncationSelectionPolicy(
    evaluation_interval=1, 
    truncation_percentage=20, 
    delay_evaluation=4 )

# When accuracy is logged for the fifth trial, metric should not be in the worst 20% of the trials so far



# 3.4 Using a Sweep Job for Hyperparameter Tuning
# In Azure ML we tune hyperparameters by running a sweep job

# Training script for hyperparameter tuning
# To run a sweep job, you need to create a training script just the way you would do for any other training job, except that your script must:
# Include an argument for each hyperparameter you want to try
# Log the target performance metric with MLflow

# E.g of script that trains a logistic regression model using a regularization argument to set the regularization rate hyperparameter and logs the accuracy metric

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow

# get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

# load the training dataset
data = pd.read_csv("data.csv")

# separate features and labels, and split for training/validatiom
X = data[['feature1','feature2','feature3','feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
mlflow.log_metric("Accuracy", acc)

# Configuring and running the sweep job 
# To prepare the weep job, you must first create a base command job that specifies which script to run and defines the parameters used by the script:
from azure.ai.ml import command

# configure command job as base
job = command(
    code="./src",
    command="python train.py --regularization ${{inputs.reg_rate}}",
    inputs={
        "reg_rate": 0.01,
    },
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    )

# And you can then override your input parameters with your search space:

from azure.ai.ml.sweep import Choice

command_job_for_sweep = job(
    reg_rate=Choice(values=[0.01, 0.1, 1]),
)

# Finally, call sweep() on your command job to sweep over your search space:

from azure.ai.ml import MLClient

# apply the sweep parameter to obtain the sweep_job
sweep_job = command_job_for_sweep.sweep(
    compute="aml-cluster",
    sampling_algorithm="grid",
    primary_metric="Accuracy",
    goal="Maximize",
)

# set the name of the sweep job experiment
sweep_job.experiment_name="sweep-example"

# define the limits for this sweep
sweep_job.set_limits(max_total_trials=4, max_concurrent_trials=2, timeout=7200)

# submit the sweep
returned_sweep_job = ml_client.create_or_update(sweep_job)

# You can then monitor the sweep jobs in Azure ML studio, sweep job will initiate trials for each hyperparameter combination and log metrics


# 4. RUNNING PIPELINES IN AZURE ML

# In Azure ML, you can experiment in notebooks and train ml models by running scripts as jobs. In a large scale process you want to separate the overall process into individual tasks. You can group tasks together as pipelines. Pipelines are key to an effective MLOps solution in Azure. MLOPS

# You can create components of individual tasks, making it easier to reuse and share code, when combining components, you make a pipeline and run as a pipeline job.

# Pipelines contain steps related to the training of an ml model.



# 4.1 Creating Components

# Components allow you to create reusable scripts that can easily be shared across users within the same Azure ML workspace. These are used to build pipelines.

# Using a Component 
# You would use a component to:
# Build a pipeline
# Share ready-to-go code
# You want to create components when you are preparing your code for scale, ready for production
# You can also create a component within Azure ML to store code within the workspace, ideally created to perform specific action
# Components can be normalizing data, training ml model, evaluating model, etc


# Creating a Component
# A component consists of three parts:
# Metadata: includes components name, version, etc
# Interface: includes the expected input parameters (like a dataset or hyperparameter) and expected output(like metrics and artifacts)
# Command, code and environment: specifies how to run the code

# To create a component, you need two files:
# A script that contains the workflow you want to execute
# A YAML file to define the metadata, interface, and command, code, and environment of the component
# You can create the YAML file or use the command_component() function as a decorator to create the YAML file
# E.g Python script prep.py that prepares the data by removing missing valued and normalizing the data


# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# setup arg parser
parser = argparse.ArgumentParser()

# add arguments
parser.add_argument("--input_data", dest='input_data',
                    type=str)
parser.add_argument("--output_data", dest='output_data',
                    type=str)

# parse args
args = parser.parse_args()

# read the data
df = pd.read_csv(args.input_data)

# remove missing values
df = df.dropna()

# normalize the data    
scaler = MinMaxScaler()
num_cols = ['feature1','feature2','feature3','feature4']
df[num_cols] = scaler.fit_transform(df[num_cols])

# save the data as a csv
output_df = df.to_csv(
    (Path(args.output_data) / "prepped-data.csv"), 
    index = False)

# COMPONENT for the prep.py script, which is stored in the src folder;
from azure.ai.ml import load_component

parent_dir = ""
loaded_component_prep = load_component(source=parent_dir + "./prep.yml")

# Registering a Component
# To use a component in a pipeline, you need the script & YAML file, and to make them accessible to others, you need to register components to the Azure ML workspace:
prep = ml_client.components.create_or_update(prepare_data_component)


# 4.2 Creating a Pipeline

# In  Azure ML, a pipeline is a workflow of ML tasks in which each task is defined as a component.
# Components can be arranged sequentially or in parallel, enabling sophisticated flow logic to orchestrate ml operations.
# Each component can be run on specific compute target, making it possible to combine different types of processing as required to achieve an overall goal

# Pipeline jobs are run and each component is executed as a child job


# A pipeline is defined in a YAML file, which includes the job name, inputs, outputs and settings , you can create the YAML file or use the @pipeline function 

# The @pipeline() function builds  pipeline consisting of two sequential steps, represented by the two load components

from azure.ai.ml.dsl import pipeline

@pipeline
def pipeline_function_name(pipeline_job_input):
    # components !!
    prep_data = loaded_component_prep(input_data=pipeline_job_input)
    train_model = loaded_component_train(training_data=prep_data.outputs.output_data)

    return {
        "pipeline_job_transformed_data": prep_data.outputs.output_data,
        "pipeline_job_trained_model": train_model.outputs.model_output,
    }

# To pass a registered data asset as the pipeline job input, you can call the created function with a data asset input:

from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

pipeline_job = pipeline_function_name(
    Input(type=AssetTypes.URI_FILE, 
    path="azureml:data:1"
))


# The result of running the @pipeline() function is a YAML file that you can review by printing the pipeline_job object created:

print(pipeline_job)

# Output will be this yaml file, which includes the configuration of the pipeline and its components

display_name: pipeline_function_name
type: pipeline
inputs:
  pipeline_job_input:
    type: uri_file
    path: azureml:data:1
outputs:
  pipeline_job_transformed_data: null
  pipeline_job_trained_model: null
jobs:
  prep_data:
    type: command
    inputs:
      input_data:
        path: ${{parent.inputs.pipeline_job_input}}
    outputs:
      output_data: ${{parent.outputs.pipeline_job_transformed_data}}
  train_model:
    type: command
    inputs:
      input_data:
        path: ${{parent.outputs.pipeline_job_transformed_data}}
    outputs:
      output_model: ${{parent.outputs.pipeline_job_trained_model}}
tags: {}
properties: {}
settings: {}


# 4.3 Running a Pipeline 

# After you’ve used the function, you can edit the pipeline configurations by specifying which parameters you wnat to change and the new value.
# E.g.
# change the output mode
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"

# set pipeline level compute
pipeline_job.settings.default_compute = "aml-cluster"

# set pipeline level datastore
pipeline_job.settings.default_datastore = "workspaceblobstore"

# To review your changes:
print(pipeline_job)

# To run the pipeline job 
# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_job"
)


# To troubleshoot when there is an issue with the pipeline itself, you will find it in the outputs and logs of the pipeline job.
# When its an issue with the components, its will be in the outputs and logs of the failed component.

# Scheduling a Pipeline
# Pipelines are useful for automating the retraining of ml models, to automate the retraining, you can schedule a pipeline using the JobSchedule class to associate a schedule to a pipeline job

# One way to create a schedule is to create a time-based schedule using the RecurrenceTrigger class with the following parameters: 

# frequency : unit of time for when schedule should fire, min, hour, day, ect 
# Interval: number of frequency units to describe how often the schedule fires, should be integer. 

from azure.ai.ml.entities import RecurrenceTrigger

schedule_name = "run_every_minute"

recurrence_trigger = RecurrenceTrigger(
    frequency="minute",
    interval=1,)

# Then you need a pipeline_job to schedule a pipeline:
from azure.ai.ml.entities import JobSchedule

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
)

job_schedule = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule
).result()

# To delete a schedule, first you should disable it:

ml_client.schedules.begin_disable(name=schedule_name).result()
ml_client.schedules.begin_delete(name=schedule_name).result()
