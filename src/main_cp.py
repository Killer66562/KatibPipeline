from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset


'''
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2']
)
def load_data(data_output: Output[Dataset]):
    import pandas as pd
    
    url = "https://raw.githubusercontent.com/daniel88516/diabetes-data/main/10k.csv"
    df_data = pd.read_csv(url)
    
    df_data = df_data.drop(df_data[df_data['diabetes'] == 'No Info'].index)
    df_data = df_data[['gender','age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
    df_data = df_data.dropna(thresh=4)
    
    gender_map = {'Male': 0 , 'Female': 1  , 'Other': 2}
    df_data['gender'] = df_data['gender'].map(gender_map)
    df_data = df_data[df_data['gender'] != 2]
    df_data['age'] = df_data['age'].replace('No Info', df_data['age'].mean())
    df_data['bmi'] = df_data['bmi'].replace('No Info', df_data['bmi'].mean())
    df_data['HbA1c_level'] = df_data['HbA1c_level'].replace('No Info', df_data['HbA1c_level'].mean())
    df_data['blood_glucose_level'] = df_data['blood_glucose_level'].replace('No Info', df_data['blood_glucose_level'].mean())

    df_data.to_csv(data_output.path)

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.1']
)
def prepare_data(
    data_input: Input[Dataset], 
    x_train_output: Output[Dataset], x_test_output: Output[Dataset],
    y_train_output: Output[Dataset], y_test_output: Output[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df_data = pd.read_csv(data_input.path)

    x = df_data.drop(labels=['diabetes'], axis=1)
    y = df_data[['diabetes']]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    x_train_df = pd.DataFrame(x_train)
    x_test_df = pd.DataFrame(x_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    x_train_df.to_csv(x_train_output.path, index=False)
    x_test_df.to_csv(x_test_output.path, index=False)
    y_train_df.to_csv(y_train_output.path, index=False)
    y_test_df.to_csv(y_test_output.path, index=False)
'''
@dsl.component(
    base_image='python:3.10-slim', 
    packages_to_install=[
        'kubeflow-katib==0.17.0'
    ]
)
def create_katib_experiment_task(
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
    random_state_min: int, 
    random_state_max: int, 
    x_train_path: str, 
    x_test_path: str,
    y_train_path: str, 
    y_test_path: str, 
    datasets_from_pvc: bool,
    datasets_pvc_name: str, 
    datasets_pvc_mount_path: str, 
    models_pvc_name: str, 
    save_model: bool
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
        ),
        V1beta1ParameterSpec(
            name="rs",
            parameter_type="int",
            feasible_space=V1beta1FeasibleSpace(
                min=str(random_state_min),
                max=str(random_state_max)
            ),
        )
    ]

    train_container = {
        "name": "training-container",
        "image": "docker.io/killer66562/xgboost-trainer",
        "command": [
            "python3",
            "/opt/xgboost/train.py",
            "--lr=${trialParameters.learningRate}",
            f"--ne={n_estimators}",
            "--rs=${trialParameters.randomState}",
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
            "mountPath": "models"
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
                name="randomState",
                description="Random state for the training model",
                reference="rs"
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
    

@dsl.pipeline
def katib_pipeline(
    experiment_name: str, 
    experiment_namespace: str = 'kubeflow-user-example-com', 
    client_namespace: str = 'kubeflow-user-example-com', 
    max_trial_counts: int = 10, 
    max_failed_trial_counts: int = 5, 
    parallel_trial_counts: int = 2,
    n_estimators: int = 2000,
    booster: str = 'gbtree', 
    learning_rate_min: float = 0.01, 
    learning_rate_max: float = 0.2, 
    random_state_min: int = 1, 
    random_state_max: int = 100, 
    x_train_path: str = "datasets/x_train.csv", 
    x_test_path: str = "datasets/x_test.csv", 
    y_train_path: str = "datasets/y_train.csv", 
    y_test_path: str = "datasets/y_test.csv", 
    datasets_from_pvc: bool = False, 
    datasets_pvc_name: str = "datasets-pvc", 
    datasets_pvc_mount_path: str = "datasets", 
    models_pvc_name: str = "models-pvc", 
    save_model: bool = False
):
    '''
    load_data_task = load_data()

    prepare_data_task = prepare_data(data_input=load_data_task.outputs['data_output'])
    '''
    
    create_katib_experiment_task(
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
        random_state_min=random_state_min, 
        random_state_max=random_state_max, 
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

compiler.Compiler().compile(katib_pipeline, 'katib_pipeline_test.yaml')