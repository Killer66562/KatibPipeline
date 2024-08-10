from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset

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


@dsl.component(
    base_image='python:3.11-slim', 
    packages_to_install=[
        'kubeflow-katib==0.16.0', 
        'pandas==2.2.2', 
        'scikit-learn==1.5.1', 
        'xgboost==2.1.1', 
        'pandas==2.2.2', 
        'joblib==1.4.2'
    ]
)
def create_katib_experiment_task(
    experiment_name: str, 
    x_train: Input[Dataset], 
    y_train: Input[Dataset], 
    x_test: Input[Dataset],
    y_test: Input[Dataset],
    experiment_namespace: str, 
    client_namespace: str, 
    max_trial_counts: int, 
    max_failed_trial_counts: int, 
    parallel_trial_counts: int,
    n_estimators_min: int,
    n_estimators_max: int,
    learning_rate_min: float, 
    learning_rate_max: float, 
    cpu_per_trial: int
):
    import pandas as pd
    import kubeflow.katib as katib

    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    def objective(paramters):
        x_train_df = pd.read_csv(x_train.path)
        y_train_df = pd.read_csv(y_train.path)
        
        model = XGBClassifier(n_estimators=parameters['n_estimators'], learning_rate=parameters['learning_rate'])
        model.fit(x_train_df, y_train_df.values.ravel())

        x_test_df = pd.read_csv(x_test.path)
        y_test_df = pd.read_csv(y_test.path)
    
        y_pred = model.predict(x_test_df)
        accuracy = accuracy_score(y_test_df, y_pred)

        print(f"accuracy={accuracy}")

    parameters = {
        'n_estimators': katib.search.int(min=n_estimators_min, max=n_estimators_max), 
        'learning_rate': katib.search.double(min=learning_rate_min, max=learning_rate_max)
    }

    client = katib.KatibClient(namespace=client_namespace)
    client.tune(
        name=experiment_name, 
        namespace=experiment_namespace, 
        objective=objective, 
        parameters=parameters, 
        objective_metric_name='accuracy', 
        objective_type='maximize', 
        objective_goal=0.99, 
        max_trial_count=max_trial_counts, 
        max_failed_trial_count=max_failed_trial_counts, 
        parallel_trial_count=parallel_trial_counts, 
        resources_per_trial={"cpu": cpu_per_trial}
    )
    client.wait_for_experiment_condition(name=experiment_name)
    print(client.get_optimal_hyperparameters(experiment_name))

@dsl.pipeline
def katib_pipeline(
    experiment_name: str, 
    experiment_namespace: str = 'kubeflow-user-example-com', 
    client_namespace: str = 'kubeflow-user-example-com', 
    max_trial_counts: int = 6, 
    max_failed_trial_counts: int = 3, 
    parallel_trial_counts: int = 1,
    n_estimators_min: int = 1000,
    n_estimators_max: int = 2000,
    learning_rate_min: float = 0.01, 
    learning_rate_max: float = 0.2, 
    cpu_per_trial: int = 1
):
    load_data_task = load_data()

    prepare_data_task = prepare_data(data_input=load_data_task.outputs['data_output'])
    
    create_katib_experiment_task(
        experiment_name=experiment_name, 
        x_train=prepare_data_task.outputs['x_train_output'],
        x_test=prepare_data_task.outputs['x_test_output'],
        y_train=prepare_data_task.outputs['y_train_output'],
        y_test=prepare_data_task.outputs['y_test_output'],
        experiment_namespace=experiment_namespace,
        client_namespace=client_namespace,
        max_trial_counts=max_trial_counts,
        max_failed_trial_counts=max_failed_trial_counts,
        parallel_trial_counts=parallel_trial_counts,
        n_estimators_min=n_estimators_min,
        n_estimators_max=n_estimators_max,
        learning_rate_min=learning_rate_min,
        learning_rate_max=learning_rate_max,
        cpu_per_trial=cpu_per_trial
    )

compiler.Compiler().compile(katib_pipeline, 'katib_pipeline.yaml')