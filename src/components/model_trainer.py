import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.components.constants import DATA_PATH
from src.components.data_validation import load_data
from src.components.data_transformation import DataTransformation
from src.components.logger import logger


class ModelTrainer:

    def __init__(self):

        os.makedirs("artifacts", exist_ok=True)

        self.model_path = "artifacts/random_forest.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def train(self):

        try:

            logger.info("Loading dataset...")

            df = load_data(DATA_PATH)

            # =========================
            # FIXED: remove only target
            # =========================
            X = df.drop("credit_risk", axis=1)
            y = df["credit_risk"]

            logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

            # =========================
            # Train/Test split
            # =========================
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            print(X_train.columns)

            logger.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")

            # =========================
            # Transformation
            # =========================
            transformer = DataTransformation()

            logger.info("Fitting transformer...")
            X_train = transformer.fit_transform(X_train)

            logger.info("Transforming test data...")
            X_test = transformer.transform(X_test)

            # =========================
            # SAFETY CHECKS
            # =========================
            assert X_train.shape[0] > 0, "X_train is empty"
            assert y_train.shape[0] > 0, "y_train is empty"
            assert X_train.shape[0] == y_train.shape[0], "Mismatch X/y"

            logger.info(f"Final X_train shape: {X_train.shape}")

            # =========================
            # Model Training
            # =========================
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )

            logger.info("Training model...")

            model.fit(X_train, y_train)

            logger.info("Model training completed")

            # =========================
            # SAVE MODEL + PREPROCESSOR
            # =========================
            joblib.dump(model, self.model_path)
            joblib.dump(transformer, self.preprocessor_path)

            logger.info(f"Model saved at {self.model_path}")
            logger.info(f"Preprocessor saved at {self.preprocessor_path}")

            # =========================
            # FILE SIZE CHECK (CRITICAL)
            # =========================
            size = os.path.getsize(self.model_path)
            logger.info(f"Model file size: {size} bytes")

            if size < 1000:
                raise ValueError("❌ Model file is too small — training failed!")

            return model, X_test, y_test

        except Exception as e:

            logger.exception("Training failed")
            raise