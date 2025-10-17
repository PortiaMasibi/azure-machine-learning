# EXPLORE AND CONFIGURE THE AZURE MACHINE LEARNING WORKSPACE

# Please check setup.py steps to install python SDK

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
