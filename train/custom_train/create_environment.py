import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from azure_connection.ml_client_setup import ml_client
from azure.ai.ml.entities import Environment, BuildContext

env_docker_context = Environment(
    build=BuildContext(path="./docker-context"),
    name="lightgbm-env",
    description="Environment created from a Docker context for lightgbm training.",
)
ml_client.environments.create_or_update(env_docker_context)