from prefect import flow, task
import subprocess


@task
def validate_data():
    subprocess.run(["python", "src/validation/validate_data.py"], check=True)


@task
def preprocess():
    subprocess.run(["python", "src/data/preprocess.py"], check=True)


@task
def train():
    subprocess.run(["python", "src/training/train.py"], check=True)


@task
def evaluate():
    subprocess.run(["python", "src/training/evaluate.py"], check=True)


@task
def register_model():
    subprocess.run(["python", "src/training/register_model.py"], check=True)


@flow(name="air-quality-training-pipeline")
def air_quality_training_pipeline():
    validate_data()
    preprocess()
    train()
    evaluate()
    register_model()


if __name__ == "__main__":
    air_quality_training_pipeline()