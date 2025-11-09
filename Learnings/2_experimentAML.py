# EXPERIMENT WITH AZURE MACHINE LEARNING

# 1. FINDING THE BEST CLASSIFICATION MODEL WITH AUTOMATED MACHINE LEARNING
# Instead of having to test and evaluate multiple algorithms, you can automate it with automated machine learning (AutoML).
# You can create an AutoML experiment using the studio, CLI, or python SDK

# 1.1 Preprocess Data and Configure Featurization
# First you need to prepare your data, after you have collected it, you need to create a DATA ASSET in Azure ML,
# For AutoML to understand your data, you need to create a MLTable data asses that includes the schema of the data
# You can create a MLTabkle data asset when your data is stored in a folder together with a MLTable file
# When you have created the data asset, you can specify it as input with the following code:

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml:input-data-automl:1")

# AutoML applies scaling and normalization to numeric data automatically

# You can choose to have AutoML apply preprocessing transformations such as:
# a. Missing value imputation to eliminate nulls 
# b. Categorical encoding to convert categorical to numeric
# c. Dropping high-cardinality features such as record IDs
# d. Feature engineering 

# AutoML will perform featurization by default, you can disable it if you wwant.

# 1.2 Running an Automated ML Experiment
# AutoML algorithms chosen will depend on the task you specify, you can also restric some algorithms based on policy or anything

# Configuring AutoML
from azure.ai.ml import automl

# Configure the classification job
classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input, # AutoML needs a MLTable data asset as input
    target_column_name="Diabetic",
    primary_metric="accuracy", # needs to be sprcified for comparison
    n_cross_validations=5,
    enable_model_explainability=True
)

# To retrieve the list of metrics available when you want to train a classification model, you can use the ClassificationPrimaryMetrics function as shown here:
from azure.ai.ml.automl import ClassificationPrimaryMetrics
#from azure.ai.ml.automl import RegressionPrimaryMetrics
 
list(ClassificationPrimaryMetrics)

# To minimize cost and time spent on training, you can set limits to an AutoML experiment or job by using set_limit()
classification_job.set_limits(
    timeout_minutes=60, # after which AutoML experiment is terminated
    trial_timeout_minutes=20, # maximum number of minutes one trial can take
    max_trials=5, # muximum number of trials, or models that will be trained 
    enable_early_termination=True, # if experiment score isnt improving in the short term
)

# To save time, you can also run multiple trials in parallel.
# When you use a compute cluster, you can have as many parallel trials as you have nodes. 
# The maximum number of parallel trials is therefore related to the maximum number of nodes your compute cluster has. 
# If you want to set the maximum number of parallel trials to be less than the maximum number of nodes, you can use max_concurrent_trials.

# Submiting ans AutoML experiment
# submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)
# Then monitor it 
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)


# 1.3 Evaluating and Comparing Models
# In the overview page, you can review models, the input data asset and summary of the best model.

# When you gave enabled featurization for your AutoML experiment, data guardrails will be automatically be applied too.
# The three data guardlais supported for classification models are:
# - class balancing detection
# - missing feature imputation
# - high cardinality feature detection 

# Each of these guardraiuls will show one of three possible states:
# - PASSED: No problems deceted and no action required
# - DONE: Changes applied to your data and you should review changes AutoML has made
# - ALERTED: An issue was detected and could not be fixed, review your data to fix the issue.

# Next to guardrails, AutoML can also apply normilization and scaling techniques to each model that is trained, you can review technique in the list of models under Algorithm name
# For example, the algorithm name of a model listed may be MaxAbsScaler, LightGBM. MaxAbsScaler refers to a scaling technique where each feature is scaled by its maximum absolute value. 
# LightGBM refers to the classification algorithm used to train the model.




# 2. TRACKING MODEL TRAINING IN JUPYTER NOTEBOOKS WITH MLFLOW

# 2.1 Configure MLflow for model Tracking in Notebooks
# As a data scientist, you want tor model to be reproducible, be able to track and log it
# MLflow is an open-source library for tracking and managing ML experiments. 
# MLflow Tracking is a component of MLflow that logs everything about the model you are training, such as parameters, metrics and artifacts

# 2.1.1 Using Azure ML Notebooks with MLflow
# - in the workspace you can create notebooks and connect the notebooks to an Azure Machine Learning MANAGED COMPUTE INSTANCE where MLflow is already configured
# - to verify that the necessary packages are installed run:
pip show mlflow # the open source library
pip show azureml-mlflow # package comtaines the intergration code of Azure ML with MLflow

# 2.1.2 Usng MLflow on a local device 
# e.g notebooks in your local device, you can configure it by 
# 1. installing pacakes 
pip install mlflow
pip install azureml-mlflow
# 2. Then navigate to the Azure ML studio 
# 3. Select name of the workspace you are working on in the top right corner of the studio
# 4. Select view all properties in azure poertal
# 5. Copy the value of the MLflow tracking URI 
mlflow.set_tracking_uri = "MLFLOW-TRACKING-URI"


# 2.2 Training and Tracking Models in Notebooks
# When training and tracking models in notebooks, to group model training results, you will EXPERIMENTS
# Creating an MLflow experiment to goup runs: 
import mlflow

mlflow.set_experiment(experiment_name="heart-condition-classifier")

# If you dont create an experiment, MLflow will assume the default experiment and name it Default

# Logging results with MLflow 
# To start a run tracked by MLflow, you will use start_run() , you can - enable autologging, use custom logging

# Enable autologging
# If using a library that is supported bu autolog, then MLflow tells the framework you are using to log all the metrics, parameters, artifacts, models considered relevant
# E.g to enable for XGBoost - mlflow.xgboost.autolog()
# Notebook cell that trains and tracks a classification model using autologging
from xgboost import XGBClassifier

with mlflow.start_run():
    mlflow.xgboost.autolog()

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Custom logging 
# If you want t o log supplementary or sutom information that is not logged through autologging 

# Common custom logging:
#mlflow.log_param(): Logs a single key-value parameter. Use this function for an input parameter you want to log.
#mlflow.log_metric(): Logs a single key-value metric. Value must be a number. Use this function for any output you want to store with the run.
#mlflow.log_artifact(): Logs a file. Use this function for any plot you want to log, save as image file first.
#mlflow.log_model(): Logs a model. Use this function to create an MLflow model, which may include a custom signature, environment, and input examples.

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

with mlflow.start_run():
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)