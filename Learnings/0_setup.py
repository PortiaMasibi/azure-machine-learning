
'''
Microsoft Azure SDK for Python - https://learn.microsoft.com/en-us/python/api/overview/azure/mgmt-compute-readme?view=azure-python

1. Prerequisites:
- Python 3.9+ is required to use this package. # python3 --version
- Azure subscription

2. Installing the Package
pip install azure-mgmt-compute
pip install azure-identity

3. Installing python SDK
pip install azure-ai-ml


4. Authenticating the Client
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)
'''



# # To connect to a workspace: 
# from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential

# # ml_client = MLClient(
# #     DefaultAzureCredential(), subscription_id, resource_group, workspace
# # )

# # After authenticating client, call ML Client
# from azure.ai.ml import command

# # configure job
# job = command(
#     code="./src",
#     command="python train.py",
#     environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
#     compute="aml-cluster",
#     experiment_name="train-model"
# )

# # connect to workspace and submit job
# returned_job = ml_client.create_or_update(job)

# # PYTHON SDK MLCLIENT : https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.mlclient?view=azure-python


