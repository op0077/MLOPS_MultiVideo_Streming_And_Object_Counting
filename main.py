from src.Object_Counting import logger
from src.Object_Counting.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.Object_Counting.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.Object_Counting.pipeline.stage_03_training import ModelTrainingPipeline
from src.Object_Counting.pipeline.stage_04_evaluation import EvaluationPipeline
import os


STAGE_NAME = "Data Ingestion stage"
print(os.getcwd())

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f"*************************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Evaluation"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e