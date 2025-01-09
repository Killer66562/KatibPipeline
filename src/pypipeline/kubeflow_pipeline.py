import json
import time

from yaml import YAMLObject
import yaml

import kfp.components as comp
import kfp.dsl as dsl
from kfp.dsl import container_component, ContainerSpec, Input, Output, InputPath, OutputPath, Artifact, Dataset
from kfp import kubernetes


def get_spark_job_definition(dataset_path):
    """
    Read Spark Operator job manifest file and return the corresponding dictionary and
    add some randomness in the job name
    :return: dictionary defining the spark job
    """
    # Read manifest file
    with open('spark-job-python-10kprocess.yaml', "r") as stream:
        spark_job_manifest = yaml.safe_load(stream)

    # Add epoch time in the job name
    epoch = int(time.time())
    spark_job_manifest["metadata"]["name"] = spark_job_manifest["metadata"]["name"].format(epoch=epoch)
    return spark_job_manifest

#Get and concat dataset
@dsl.component(
  base_image="python:3.9",
  packages_to_install=["python-dotenv==1.0.1", "pandas==2.2.2", "numpy==2.0.0"]
)
def load_raw_datasets_from_nfs(
  diabetes_dataset: Output[Dataset]
):
  import pandas as pd
# fix csv path
  dataset_pd = pd.read_csv('/mnt/10kdataset.csv')
  dataset_pd.to_csv(diabetes_dataset.path)

@dsl.component(
  base_image="python:3.9"
)
def print_msg(msg: str) -> str:
    print(msg)
    return msg


@dsl.pipeline(
    name="Spark Operator job pipeline",
    description="Spark Operator job pipeline"
)
def spark_job_pipeline():

    load_raw_data_from_nfs_task = load_raw_datasets_from_nfs()
    load_raw_data_from_nfs_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
       load_raw_data_from_nfs_task,
       pvc_name='datasets-raw-pvc',
       mount_path="/mnt"
    )

    dataset_path = load_raw_data_from_nfs_task.outputs['diabetes_dataset']
    spark_job_definition = get_spark_job_definition(dataset_path=dataset_path)
    k8s_apply_op = comp.load_component_from_file("k8s-apply-component.yaml")
    spark_job_task = k8s_apply_op(object=json.dumps(spark_job_definition), dataset=dataset_path)

    # Fetch spark job name
    spark_job_name = spark_job_definition["metadata"]["name"]
    spark_job_namespace = spark_job_definition["metadata"]["namespace"]
    spark_job_task.set_caching_options(enable_caching=False)
    
    check_sparkapplication_status_op = comp.load_component_from_file("checkSparkapplication.yaml")
    check_sparkapplication_status_task = check_sparkapplication_status_op(name=spark_job_name, namespace=spark_job_namespace).after(spark_job_task)

    check_sparkapplication_status_task.set_caching_options(enable_caching=False)

    print_message_task = print_msg(msg=f"Job {spark_job_name} is completed.").after(check_sparkapplication_status_task)
    print_message_task.set_caching_options(enable_caching=False)


if __name__ == "__main__":
    # Compile the pipeline
    import kfp.compiler as compiler
    import logging
    logging.basicConfig(level=logging.INFO)
    pipeline_func = spark_job_pipeline
    pipeline_filename = "sparkJobPipeline.yaml"
    compiler.Compiler().compile(pipeline_func, pipeline_filename)
    logging.info(f"Generated pipeline file: {pipeline_filename}.")
