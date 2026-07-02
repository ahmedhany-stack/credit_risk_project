import os

project_structure = {
    "loan_Project": [
        ".github/workflows/ci.yml",

        "artifacts/model.pkl",
        "artifacts/preprocessor.pkl",
        "artifacts/metrics.json",

        "config/config.yaml",

        "data/raw/dataset.csv",
        "data/interim/",
        "data/processed/",

        "notebooks/01_eda.ipynb",
        "notebooks/02_feature_engineering.ipynb",
        "notebooks/03_model_experiments.ipynb",

        "src/components/data_ingestion.py",
        "src/components/data_validation.py",
        "src/components/data_transformation.py",
        "src/components/model_trainer.py",
        "src/components/model_evaluation.py",
        "src/components/model_pusher.py",

        "src/pipeline/training_pipeline.py",
        "src/pipeline/prediction_pipeline.py",

        "src/utils.py",
        "src/logger.py",
        "src/exception.py",
        "src/config.py",

        "app/main.py",
        "app/schemas.py",
        "app/predict.py",

        "tests/test_preprocessing.py",
        "tests/test_training.py",
        "tests/test_api.py",

        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "README.md",
        "setup.py",
        "pyproject.toml",
        ".gitignore",
        "LICENSE"
    ]
}


def create_structure(base_path, structure):
    for project, files in structure.items():
        project_path = os.path.join(base_path, project)

        for file_path in files:
            full_path = os.path.join(base_path, file_path)

            dir_name = os.path.dirname(full_path)

            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # create empty file if it's not a folder
            if not file_path.endswith("/"):
                with open(full_path, "w", encoding="utf-8") as f:
                    if file_path.endswith(".gitignore"):
                        f.write("__pycache__/\n*.pyc\n.env\n.venv/\n")
                    elif file_path.endswith("README.md"):
                        f.write("# Loan Project\n")
                    elif file_path.endswith("config.yaml"):
                        f.write("project: loan_project\n")
                    else:
                        f.write("")

    print("✅ Project structure created successfully!")


if __name__ == "__main__":
    create_structure(os.getcwd(), project_structure)