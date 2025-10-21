# EXPLORE AND CONFIGURE THE AZURE MACHINE LEARNING WORKSPACE

# Please check setup.py steps to install python SDK

# 1. Azure Machine Learning Workspace Resources and Assets

# 1.1 Azure ML Workspace Resource and Assets
# Azure Subscription - Resource Group - Azure ML Service - Workspace
# Workspace is provisioned with other resources - storage account, AKV, application insights, container registry
# 

# Creating the Workspace usong Python SDK (Can also use Azure CLI, ARM template/ portal)
from azure.ai.ml.entities import Workspace
workspace_name = "myworkspace"

ws_basic = Workspace (
    name = workspace_name,
    location = "eastus",
    dispay_name = "Basic workspace-example",
    description = "This is a basic workspace created using Python SDK"
)
ml_client.workspaces.begin_create(ws_basic)

# Role Based Access Control (RBAC) - Assign roles to users to access workspace: Owner, Contributo, Reader, AzureML Data Scientist, AzureML Compute Operator


# 1.2 Azure ML Resources
# Resources are infrastruture you need to run an ML workflow
# a. The workspace
# b. Compute resources - compute instances, compute clusters, attached compute, kubernetes clusters, serverless compute etc
# c. Datastores - datastores contain the connection information to Azure data storage: workspaceartifactstore, workspaceworkingdirectory,worksspaceblobstore, workspaceblobstore, workspacefilestore


# 1.3 Azure ML Assets
# These are created and used at various stages of a project
# a. Models - sace as python pickle files.pkl
# b. Environments - make sure necessary dependencies are installed on the compute that executes the code
# c. Data -data assets refer to a specific file or folder, you can use them to access data every time, without having to provide authentication every time
# d. Components - a step in a pipeline, e.g normalization, training, testing, validation

# 1.4 Training Models in the Workspace
# you can use 
# a. Automated ML to explore algorithms and hyperparameters
# b. Running a Jupyter Notebook to develop code
# c. Running a Python Script as a job on compute - when code is production ready. There are different types of jobs:
#.    - Command job: single task, can be part of a pipeline
#.    - Sweep job: hyperparameter tuning when executing single script
#.    - Pipeline job: multiple steps in a workflow


# 2. Developer Tools for Workspace Interaction

# 2.1 Azure ML Studio
# Author: here you can create new jobs to train and track ML models - (Notebooks, Automated ML, Designer)
# Assets: here you can create and review assest you use when training models (Models, Environments, Data,Components, Pipelines, Jobs, Endpoints)
# Manage: here you can create and manage resources you need to train models ( Compute, Linked Services, Data Labeling)
# The studio is used to verify if pipelines ran successfully, or for quick experimentation or exploring past jobs. 
# For more repetitive tasks you would like to automate, the Azure CLI or Python SDK are better suited as these allow you to work on the code. 

# 2.2 Azure Python Software Development Kit (SDK)
# As a data scientist you will mostly work with the assses!
# Can connect to Jupyter notebooks or VS code 
# install with ----   pip install azure-ai-ml


# To connect to a workspace: 
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
DefaultAzureCredential(), subscription_id, resource_group, workspace)

# After authenticating client, call ML Client
# You call MLClient to create, retrieve, update, and delete resources in your workspace
from azure.ai.ml import command
# configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    experiment_name="train-model"
)
#connect to workspace and submit job
returned_job = ml_client.create_or_update(job)

# PYTHON SDK MLCLIENT CLASS REF DOC : https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.mlclient?view=azure-python

# 2.3 Azure CLI
# You can use Azure CLI to interact with the workspace and its assets - used mostly by engineers and andministrators to automate tasks
# It allows you to automate the creating and configuration of assets and resources to make it repeatable. 
# Ensuring consistency across environments, and incorporating ml config in devops workflows such as CI/CD pipelines

# Install Azure CLI: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

# Afer installing the Azure CLI, you have to install the Azure Machine Learning Extension : az extension add -n ml -y

# Then you can run help commang to get list of available commands: az ml -h

# List of commands to reference: https://learn.microsoft.com/en-us/cli/azure/ml

# E.g creating compute target
# az ml compute create --name aml-cluster --size STANDARD_DS3_v2 --min-instances 0 --max-instances 5 --type AmlCompute --resource-group my-resource-group --workspace-name my-workspace

# Or first create a yaml file (compute.yaml) with the compute configuration

# $schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json 
# name: aml-cluster
# type: amlcompute
# size: STANDARD_DS3_v2
# min_instances: 0
# max_instances: 5

#  and then run:

# az ml compute create --file compute.yml --resource-group my-resource-group --workspace-name my-workspace


# 3. Make Data Available in Azure ML



# 4. Compute Targets in Azure ML



# 5. Environments in Azure ML