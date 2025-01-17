import sys
import os

# Add the 'utils' folder to Python's module search path
sys.path.append(os.path.abspath('../OBJECT_COUNTING_1/src'))

# from src.Object_Counting.config.configuration import ConfigurationManager
# from src.Object_Counting.components.training import Training
# from src.Object_Counting import logger

from Object_Counting.config.configuration import ConfigurationManager
from Object_Counting.components.training import Training
from Object_Counting import logger
STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e