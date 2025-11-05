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
# To access data in Azure ML,you can access data using Uniform Resource Identifiers (URIs)
# You can create datastores and data assets to make it easier to access data repeatedly without having to provide authentication details every time.

# 3.1 Understanding URIs
# URI references the location of your data, there are three common  types of URIs in Azure ML:
# a.Data stores in publicly of privately in an Azure Blob Storage or publicly available https(s) location - http(s)
# b.Data stores in an ADLS Gen2 -  abfs(s)
# c.Data stored in a datastore in your workspace - azureml://datastores/{datastore_name}/paths/{path_in_datastore}

# A datastore is a reference to an existing storage account on Azure. 
# Therefore, when you refer to data stored in a datastore, you may be referring to data being stored in an Azure Blob Storage
# or Azure Data Lake Storage. When you refer to the datastore however, you won't need to authenticate as the connection
# information stored with the datastore will be used by Azure Machine Learning.

# It's considered a best practice to avoid any sensitive data in your code, like authentication information. 
# Therefore, whenever possible, you should work with datastores and data assets in Azure Machine Learning. However, 
# during experimentation in notebooks, you may want to connect directly to a storage location to avoid unnecessary overhead.

# 3.2 Creating a Datastore
# Datastores encapsulate the connection information needed to access your data in Azure Storage.
# These provide easy-to-use URIs to data, securely store information needed to access the data, and can be reused across multiple experiments.
# When creating a datastore with an existing storage acc on Azure, you havee the choice between two different authentication methods:
# a. Credential basesd authentication - using account key, SAS token, or service principals
# b. Identity based authentication - using managed identity or user assigned identity

# Creation of datastores for most common types of Azure data sources: Azure Blob Storage , Azure File Share, Azure Data Lake Gen 2

# Using Buit-In Datastores: Each workspace is created with four built-in datastores, two connecting to Azure Storage Blob container
# and two connecting to Azure Storage File Share. 

# You can create a datastore using Azure CLI, user interface in Azure ML Studio or Python SDK. 
# Depending on the storage service you want to connect to, there are different options to authenticate

# E.G Azure Blob Storage using ACCOUNT KEY
blob_datastore = AzureBlobDatastore(
    name = "blob_example",
    description = " Datastore pointing to a blob container using account key authentication",
    account_name = "mytestblobstore",
    container_name = " data-container",
    credentials = AccountKeyCredentials(account_key="XXXXxxxXXxxXxx")
)
ml_client.create_or_update(blob_datastore)


# E.G Azure Blob Storage using SAS TOKEN
blob_datastore = AzureBlobDatastore(
name="blob_sas_example",
description="Datastore pointing to a blob container",
account_name="mytestblobstore",
container_name="data-container",
credentials=SasTokenConfiguration(
sas_token="?xx=XXXX-XX-XX&xx=xxxx&xxx=xxx&xx=xxxxxxxxxxx&xx=XXXX-XX-XXXXX:XX:XXX&xx=XXXX-XX-XXXXX:XX:XXX&xxx=xxxxx&xxx=XXxXXXxxxxxXXXXXXXxXxxxXXXXXxxXXXXXxXXXXxXXXxXXxXX"
),
)
ml_client.create_or_update(blob_datastore)


# Creating Datastore to Connect to other types of cloud storage  - https://learn.microsoft.com/en-us/azure/machine-learning/how-to-datastore


# 3.3 Creating a Data Asset
# To simplify getting access to the data you want to work with, you can use data assets. 
# Data assets are REFERENCES to where data is stored, how to get access and any other relevant metadata. 
# You can create data assests to get access to data in datastores, Azure storage servicesm public URLs or local device.

# Benefits of data assets:
# a. Can share and reuse data with other members of your team
# b. Can seamlessly access data during model training without worrying about connection strings or data paths
# c. Can version the metadata of the data asset

# Three types of data assets you can use:
# 1. URI file - points to a specific file 
# The supported paths are:
# a. Local: ./<path-to-file> - when you creaete a data asset from a local file, the file is uploaded to the default datastore of the workspace - workspaceblobstore
# b. Azure Blob Storage: wasbs://<account_name>.blob.core.windows.net/<container_name>/<folder>/<file>
# c. Azure Data Lake Storage Gen2: abfss://<file_system>@<account_name>.dfs.core.windows.net/<folder>/<file>
# d. Datastore: azureml://datastores/<datastore_name>/paths/<folder>/<file>

# URI file data asset example:
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<supported-path>'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FILE,
    description="<description>",
    name="<name>",
    version="<version>"
)

ml_client.data.create_or_update(my_data)

# When you parse the URI file data asset as input in an Azure Machine Learning job, you first need to read the data before you can work with it.
# Creating a Python script you want to run as a job, and you set the value of the input parameter input_data to be the URI file data asset 
# (which points to a CSV file). You can read the data by including the following code in your Python script:

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)
print(df.head(10))


# 2. URI folder - points to a folder
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
my_path = '<supported-path>'
my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="<description>",
    name="<name>",
    version='<version>'
)
ml_client.data.create_or_update(my_data)


# You can then read all CSV files in the folder and concatenate them, which you can do by including the following code in your Python script:
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

data_path = args.input_data
all_files = glob.glob(data_path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)



# 3. MLTable - points to a folder or a file, includes a schema to read as tabular data 
# MLTable data asset allows you to point to tabular data, you have to specify the schema definition to read the data
# You want to use an MLTable data asset when the schema of  your data is complex and changes frequently, instead
# of changing how to read the data in every script that uses the data, you only have to change it in the data itself.
# Automated ML, you hace to use an MLTable data asset as AML needs to know how to read the data
# To define the schema, you inlcude an MLTable file in the same folder as the data you want to read.
# This includes the  path pointing to the data you want to read and how to read it

#M LTable file yml example 
type: mltable

paths:
  - pattern: ./*.txt
transformations:
  - read_delimited:
      delimiter: ','
      encoding: ascii
      header: all_files_same_headers

#To create a MLTable data asses:
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<path-including-mltable-file>'

my_data = Data(
    path=my_path,
    type=AssetTypes.MLTABLE,
    description="<description>",
    name="<name>",
    version='<version>'
)

ml_client.data.create_or_update(my_data)

# Using it:
import argparse
import mltable
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

tbl = mltable.load(args.input_data)
df = tbl.to_pandas_dataframe()

print(df.head(10))

# Data Assets are most useful when executing ML tasks as Azure ML jobs, you can run a python script that takes inputs and generates outputs.
# A data asset can be parsed as both an input or output of an Azure ML job.



# 4. Compute Targets in Azure ML
# Compute targets are physical or virtual computers on which jobs are executed
# Cloud compute is mostly important on scaling the work you do locally on your machine

# 4.1 Choosing the Appropriate Compute Target 
# a. Compute Instances - ideal for development and testing (experimentation phase), single VM, Jupyter notebooks, VS code 
# b. Compute Clusters - ideal for training and batch inference at scale (production phase), multiple VMs, auto scaling, parallel processing
# c. Kubernetes Clusters - ideal for large scale training and real-time inference (production phase), multiple VMs, container orchestration, high availability
# d. Attached Compute - ideal for reusing existing compute resources, e.g on-premise servers, other cloud providers, Azure Databricks, Azure Synapse Analytics
# e. Serverless Compute - ideal for  on demand compute you can use for training jobs, fully managed. 

# Experimentation Phase - Compute Instances & Serverless Compute & Attached Compute
# Production Phase (Running jobs to train model / run pipeline batch jobs) - Compute Clusters, Azure ML Serverless Compute, Kubernetes, Attached Compute
# Deployment Phase (Deploying models for real-time inference) - Kubernetes Clusters, Containers, if its batch inference you can use compute targets



# 4.2 Compute Instances
# Using Python SDK to create a compute instance 
from azure.ai.ml.entities import ComputeInstance

ci_basic_name = "basic-ci-12345"
ci_basic = ComputeInstance(
    name=ci_basic_name, 
    size="STANDARD_DS3_v2"
)
ml_client.begin_create_or_update(ci_basic).result()

# Compute instance class expects following parameters: https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.computeinstance
# Compute instances also need a unique name across the region
# You can also create a compute instance by using a script too.With a script, you ensure that any necessary packages, tools, or 
# software is automatically installed on the compute and you can clone any repositories to the compute instance. 
# When you need to create compute instances for multiple users, using a script allows you to create a consistent development environment for everyone.


# To be allowed to work with the compute instance, it needs to be assigned to you as a user
# A compute instance can only be assigned to ONE USER, as it cant hanlde parallel workloads

# You want your compute instance to be running when you are working on it, and stopped when you are not using it,
# You can either do this yourself, set a schedule for it and also automatically shut doen when idle for a certain period of time - saving costs

# To use this compute instance, you can work with it through intergrated notebooks in Azure ML Studio, or connect to it using Visual Studio Code



# 4.3 Compute Clusters
# When running code in production, you use scripts instead of notebooks, you want to use compute teagrte that is scalable
# You can create compute clusters using Python SDK, Azure CLI or Azure ML Studio

#Python SDK example:
from azure.ai.ml.entities import AmlCompute

cluster_basic = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",
    size="STANDARD_DS3_v2",
    location="westus",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120,
    tier="low_priority",
)
ml_client.begin_create_or_update(cluster_basic).result()

# AmlCompute class reference doc: https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.amlcompute?view=azure-python

# There are three main parameters to consider when creating a compute cluster:
# a. size - the VM type to use for the nodes in the cluster, CPU/GPU? - https://learn.microsoft.com/en-us/azure/virtual-machines/sizes
# b. max instances - maximum number of nodes you can scale out to. Number of parallel workloads your compute cluster can handle. 
# c. tier - specifies whether vm are low priority of dedicated. Low priority are cheaper but can be evicted when Azure needs the capacity.


# Using Compute Cluster:
# a. When running a pipeline job you built in the Designer
# b. Running an Automated ML job - compyte clusters allows you to train multiple models in parallel which is common in AutoML
# c. Running a script as a job 

from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python diabetes-training.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="cpu-cluster",
    display_name="train-with-cluster",
    experiment_name="diabetes-training"
    )

# submit job
returned_job = ml_client.create_or_update(job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)




# 5. Environments in Azure ML
# Environnments list and store the necessary pacakges that you can reuse when running your code on different compute targets
# When you create an Azure Machine Learning workspace, curated environments are automatically created and made available to you. 
# Alternatively, you can create and manage your own custom environments and register them in the workspace. 
# Creating and registering custom environments makes it possible to define consistent, reusable runtime contexts for your 
# experiments - regardless of where the experiment script is run.

# Azure ML builds the environment definitions into Docker images and conda environments .
# Azure ML builds the environments on the Azure Container Registry associated with the workspace

# To list environments using python SDK:
envs = ml_client.environments.list()
for env in envs:
    print(env.name)

# To review details of a specific environment:
env = ml_client.environments.get(name="my-environment", version="1")
print(env)


# 5.1 Curated Environments
# Curated environmenyts are prebuilt by Azure ML team and maintained by them
# They use the AzureML prefix and are designed to provide scripts that are popular ml frameworks and tooling

# The following command allows you to retrive the description and tags of a curated env with PythonSDK:
env = ml_client.environments.get("AzureML-sklearn-0.24-ubuntu18.04-py37-cpu", version=44)
print(env. description, env.tags)

# Using a curated env
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-with-curated-environment",
    experiment_name="train-with-curated-environment"
)

# submit job
returned_job = ml_client.create_or_update(job)


# You can verify that a curated environment includes all necessary packages by reviewing its details
# When your job fails because packages needed are missing, you can review detailed error logs in the Outputs + logs
# A common erro message is the "ModuleNotFoundError", indicating that a package is missing from the environment


# 5.2 Custom Environments 
# When you need to create your own environemnts in Azure ML to list all necessary packages, libraries and dependencies
# to run your scripts you can create custome environments

# You can define an environments from a DOCKER IMAGE, A DOCKER BUILD WITH CONTEXT and a CONDA SPECIFICATION WITH DOCKER IMAGE

#a. Creating a custom envrionemnt from a Docker Image
# Docker images can be hosted in a public registry like Docker Hub (https://hub.docker.com/) or privately stored in and Azure Container Registry. 
# E.g publick Docker imahe that contains all necessary pacakges to train deep learning model with PyTorch - https://hub.docker.com/r/pytorch/pytorch

# To create an environments from a Docker image:
from azure.ai.ml.entities import Environment
env_docker_image = Environment(
    image="pytorch/pytorch:latest",
    name="public-docker-image-example",
    description="Environment created from a public Docker image.",
)
ml_client.environments.create_or_update(env_docker_image)


#You can also use Azure ML base images to create an env:
from azure.ai.ml.entities import Environment
env_docker_image = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    name="aml-docker-image-example",
    description="Environment created from a Azure ML Docker image.",
)
ml_client.environments.create_or_update(env_docker_image)


#b. Creating a custom envrionemnt with a Conda Specification File
# Evem though docker images contain all necessary pacakges when working with specific framework, you may need to inlude other packages
# E.g when you want to train the model with PyTorch and track model with MLflow
# you can add a conda spcification file to a docker image when creating an environment

# Conda specification file example (conda.yml)
name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.7
  - scikit-learn
  - pandas
  - numpy
  - matplotlib

# Then create the environment:
from azure.ai.ml.entities import Environment
env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./conda-env.yml",
    name="docker-image-plus-conda-example",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env_docker_conda)

# YOU CANNOT CREATE A CURATED ENV PREFIXED WITH AzureML


# Using an Environment
# To specify which environment you want to use to run your script, you reference an environment using the
#  <curated-environment-name>:<version> or <curated-environment-name>@latest syntax.
# The first time you use an environment in a job, it can take about 10-15 minutes to prepare the environment.
# the image of the environment is then hosted in the Azure Container Registry associated with your workspace, and 
# you can use it for another job with no build time.
