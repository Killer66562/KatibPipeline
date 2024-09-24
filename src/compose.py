from kfp import dsl, compiler
from kfp.dsl import Input, Output, Metrics, component


@component(base_image='python:3.10-slim')
def parse_input_json(
    json_file_path: str, 
    xgboost_input_metrics: Output[Metrics], 
    random_forest_input_metrics: Output[Metrics]
):
    import json

    def log_metric(metrics: Metrics, input_dict: dict):
        for key in input_dict:
            if key == "method":
                continue
            else:
                metrics.log_metric(key, input_dict.get(key))

    input_dict_arr: list[dict] = json.load(json_file_path)
    for input_dict in input_dict_arr:
        if input_dict["method"] == "xgboost":
            log_metric(xgboost_input_metrics, input_dict)
        elif input_dict["method"] == "random_forest":
            log_metric(random_forest_input_metrics, input_dict)
        else:
            continue

@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=[
        'kubeflow-katib==0.17.0'
    ]
)
def run_xgboost_katib_experiment(
    input_params_metrics: Input[Metrics], 
    best_params_metrics: Output[Metrics]
):
    from kubeflow.katib import KatibClient
    from kubernetes.client import V1ObjectMeta
    from kubeflow.katib import V1beta1Experiment
    from kubeflow.katib import V1beta1AlgorithmSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1FeasibleSpace
    from kubeflow.katib import V1beta1ExperimentSpec
    from kubeflow.katib import V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1ParameterSpec
    from kubeflow.katib import V1beta1TrialTemplate
    from kubeflow.katib import V1beta1TrialParameterSpec

    experiment_name = input_params_metrics.metadata.get("experiment_name")
    experiment_namespace = input_params_metrics.metadata.get("experiment_namespace")

    if experiment_name is None or experiment_namespace is None:
        raise ValueError("Both experiment_name and experiment namespace needs to be a string!")

    metadata = V1ObjectMeta(
        name=experiment_name, 
        namespace=experiment_namespace
    )

    algorithm_spec = V1beta1AlgorithmSpec(
        algorithm_name="random"
    )

    objective_spec = V1beta1ObjectiveSpec(
        type="maximize",
        goal= 0.99,
        objective_metric_name="accuracy",
    )

    learning_rate_min = input_params_metrics.metadata.get("learning_rate_min")
    learning_rate_max = input_params_metrics.metadata.get("learning_rate_max")
    learning_rate_step = input_params_metrics.metadata.get("learning_rate_step")

    if learning_rate_min is None or learning_rate_max is None or learning_rate_step is None:
        raise ValueError("All learning_rate_min, learning_rate_max and learning_rate_step cannot be null!")

    try:
        learning_rate_min = float(learning_rate_min)
        learning_rate_max = float(learning_rate_max)
        learning_rate_step = float(learning_rate_step)
    except ValueError:
        raise ValueError("All learning_rate_min, learning_rate_max and learning_rate_step needs to be a float!")

    n_estimators_min = input_params_metrics.metadata.get("n_estimators_min")
    n_estimators_max = input_params_metrics.metadata.get("n_estimators_max")
    n_estimators_step = input_params_metrics.metadata.get("n_estimators_step")

    if n_estimators_min is None or n_estimators_max is None or n_estimators_step is None:
        raise ValueError("All n_estimators_min, n_estimators_max and n_estimators_step cannot be null!")

    try:
        n_estimators_min = int(n_estimators_min)
        n_estimators_max = int(n_estimators_max)
        n_estimators_step = int(n_estimators_step)
    except ValueError:
        raise ValueError("All n_estimators_min, n_estimators_max and n_estimators_step needs to be a float!")

    parameters = [
        V1beta1ParameterSpec(
            name="lr",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(
                min=str(learning_rate_min),
                max=str(learning_rate_max), 
                step=str(learning_rate_step)
            ),
        ), 
        V1beta1ParameterSpec(
            name="ne",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(
                min=str(n_estimators_min),
                max=str(n_estimators_max), 
                step=str(n_estimators_step)
            ),
        )
    ]

    docker_image_name = input_params_metrics.metadata.get("docker_image_name")
    if docker_image_name is None:
        raise ValueError("Docker image name cannot be null!")

    random_state = input_params_metrics.metadata.get("random_state")
    if random_state is None:
        random_state = 42
    else:
        try:
            random_state = int(random_state)
        except ValueError:
            raise ValueError("Random state needs to be an int!")
        
    x_train_path = input_params_metrics.metadata.get("x_train_path")
    x_test_path = input_params_metrics.metadata.get("x_test_path")
    y_train_path = input_params_metrics.metadata.get("y_train_path")
    y_test_path = input_params_metrics.metadata.get("y_test_path")

    train_container = {
        "name": "training-container",
        "image": f"docker.io/{docker_image_name}",
        "command": [
            "python3",
            "/opt/xgboost/train.py",
            "--lr=${trialParameters.learningRate}",
            "--ne=${trialParameters.nEstimators}",
            f"--rs={random_state}",
            f"--esp=100000",
            f"--booster=gbtree",
            f"--x_train_path={x_train_path}",
            f"--x_test_path={x_test_path}",
            f"--y_train_path={y_train_path}",
            f"--y_test_path={y_test_path}",
            f"--save_model=false",
            f"--model_folder_path=models"
        ]
    }

    template_spec = {
        "containers": [
            train_container
        ],
        "restartPolicy": "Never"
    }

    volumes = []
    volumeMounts = []

    datasets_from_pvc = input_params_metrics.metadata.get("datasets_from_pvc")
    datasets_pvc_name = input_params_metrics.metadata.get("datasets_pvc_name")
    datasets_pvc_mount_path = input_params_metrics.metadata.get("datasets_pvc_mount_path")
    
    if datasets_from_pvc is True:
        volumes.append({
            "name": "datasets", 
            "persistentVolumeClaim": {
                "claimName": datasets_pvc_name
            }
        })
        volumeMounts.append({
            "name": "datasets", 
            "mountPath": datasets_pvc_mount_path
        })

    if save_model is True:
        volumes.append({
            "name": "models", 
            "persistentVolumeClaim": {
                "claimName": models_pvc_name
            }
        })
        volumeMounts.append({
            "name": "models", 
            "mountPath": "/opt/xgboost/models"
        })

    if datasets_from_pvc is True or save_model is True:
        train_container["volumeMounts"] = volumeMounts
        template_spec["volumes"] = volumes


    trial_spec={
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": template_spec
            }
        }
    }



    trial_template=V1beta1TrialTemplate(
        primary_container_name="training-container",
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="learningRate",
                description="Learning rate for the training model",
                reference="lr"
            ), 
            V1beta1TrialParameterSpec(
                name="nEstimators",
                description="N estimators for the training model",
                reference="ne"
            )
        ],
        trial_spec=trial_spec,
        retain=True
    )

    max_trial_counts = input_params_metrics.metadata.get("max_trial_counts")
    max_failed_trial_counts = input_params_metrics.metadata.get("max_failed_trial_counts")
    parallel_trial_counts = input_params_metrics.metadata.get("parallel_trial_counts")

    if max_failed_trial_counts is None or max_failed_trial_counts is None or parallel_trial_counts is None:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and parallel_trial_counts cannot be null!")
    
    try:
        max_trial_counts = int(max_trial_counts)
        max_failed_trial_counts = int(max_failed_trial_counts)
        parallel_trial_counts = int(parallel_trial_counts)
    except ValueError:
        raise ValueError("All max_trial_counts, max_failed_trial_counts and needs to be an int!")

    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=metadata,
        spec=V1beta1ExperimentSpec(
            max_trial_count=max_trial_counts,
            parallel_trial_count=parallel_trial_counts,
            max_failed_trial_count=max_failed_trial_counts,
            algorithm=algorithm_spec,
            objective=objective_spec,
            parameters=parameters,
            trial_template=trial_template,
        )
    )

    client_namespace = input_params_metrics.metadata.get("client_namespace")
    if client_namespace is None:
        raise ValueError("Client namespace cannot be null!")

    client = KatibClient(namespace=client_namespace)
    client.create_experiment(experiment=experiment)
    client.wait_for_experiment_condition(name=experiment_name, namespace=experiment_namespace, timeout=3600)

    result = client.get_optimal_hyperparameters(name=experiment_name, namespace=experiment_namespace).to_dict()

    best_params_list = result["parameter_assignments"]

    for params in best_params_list:
        name = params["name"]
        value = params["value"]
        best_params_metrics.log_metric(metric=name, value=value)