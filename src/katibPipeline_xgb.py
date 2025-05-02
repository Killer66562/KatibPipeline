from kfp import dsl, compiler
from kfp.dsl import Output, Metrics


@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=[
        'kubeflow-katib==0.17.0'
    ]
)
def create_katib_experiment_task(
    docker_image_name: str, 
    experiment_name: str, 
    experiment_namespace: str, 
    client_namespace: str, 
    max_trial_counts: int, 
    max_failed_trial_counts: int, 
    parallel_trial_counts: int,
    n_estimators: int,
    booster: str, 
    learning_rate_min: float, 
    learning_rate_max: float, 
    random_state: int, 
    early_stopping_rounds: int, 
    x_train_path: str, 
    x_test_path: str,
    y_train_path: str, 
    y_test_path: str, 
    datasets_from_pvc: bool,
    datasets_pvc_name: str, 
    datasets_pvc_mount_path: str, 
    models_pvc_name: str, 
    save_model: bool, 
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

    parameters = [
        V1beta1ParameterSpec(
            name="lr",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(
                min=str(learning_rate_min),
                max=str(learning_rate_max)
            ),
        )
    ]

    train_container = {
        "name": "training-container",
        "image": f"docker.io/{docker_image_name}",
        "command": [
            "python3",
            "/opt/xgboost/train.py",
            "--lr=${trialParameters.learningRate}",
            f"--ne={n_estimators}",
            f"--rs={random_state}",
            f"--esp=${early_stopping_rounds}",
            f"--booster={booster}",
            f"--x_train_path={x_train_path}",
            f"--x_test_path={x_test_path}",
            f"--y_train_path={y_train_path}",
            f"--y_test_path={y_test_path}",
            f"--save_model={save_model}",
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
            )
        ],
        trial_spec=trial_spec,
        retain=True
    )

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

    client = KatibClient(namespace=client_namespace)
    client.create_experiment(experiment=experiment)
    client.wait_for_experiment_condition(name=experiment_name, namespace=experiment_namespace, timeout=3600)

    result = client.get_optimal_hyperparameters(name=experiment_name, namespace=experiment_namespace).to_dict()

    best_params_list = result["parameter_assignments"]

    for params in best_params_list:
        name = params["name"]
        value = params["value"]
        best_params_metrics.log_metric(metric=name, value=value)
    

@dsl.pipeline
def katib_pipeline(
    experiment_name: str, 
    experiment_namespace: str = 'kubeflow-user-example-com', 
    client_namespace: str = 'kubeflow-user-example-com', 
    docker_image_name: str = "killer66562/xgboost-trainer:latest", 
    max_trial_counts: int = 10, 
    max_failed_trial_counts: int = 5, 
    parallel_trial_counts: int = 2,
    n_estimators: int = 2000,
    booster: str = 'gbtree', 
    learning_rate_min: float = 0.01, 
    learning_rate_max: float = 0.2, 
    random_state: int = 42, 
    early_stopping_rounds: int = 1000, 
    x_train_path: str = "/opt/xgboost/datasets/x_train.csv", 
    x_test_path: str = "/opt/xgboost/datasets/x_test.csv", 
    y_train_path: str = "/opt/xgboost/datasets/y_train.csv", 
    y_test_path: str = "/opt/xgboost/datasets/y_test.csv", 
    datasets_from_pvc: bool = False, 
    datasets_pvc_name: str = "datasets-pvc", 
    datasets_pvc_mount_path: str = "/opt/xgboost/datasets", 
    models_pvc_name: str = "models-pvc", 
    save_model: bool = False
):
    '''
    load_data_task = load_data()

    prepare_data_task = prepare_data(data_input=load_data_task.outputs['data_output'])
    '''
    
    create_katib_experiment_task(
        docker_image_name=docker_image_name, 
        experiment_name=experiment_name, 
        experiment_namespace=experiment_namespace,
        client_namespace=client_namespace,
        max_trial_counts=max_trial_counts,
        max_failed_trial_counts=max_failed_trial_counts,
        parallel_trial_counts=parallel_trial_counts,
        n_estimators=n_estimators, 
        booster=booster, 
        learning_rate_min=learning_rate_min,
        learning_rate_max=learning_rate_max, 
        random_state=random_state, 
        early_stopping_rounds=early_stopping_rounds, 
        x_train_path=x_train_path, 
        x_test_path=x_test_path, 
        y_train_path=y_train_path, 
        y_test_path=y_test_path,
        datasets_from_pvc=datasets_from_pvc, 
        datasets_pvc_name=datasets_pvc_name, 
        datasets_pvc_mount_path=datasets_pvc_mount_path, 
        models_pvc_name=models_pvc_name, 
        save_model=save_model
    )

compiler.Compiler().compile(katib_pipeline, 'katib_pipeline_xgboost.yaml')