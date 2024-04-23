import sys
import os

# Add the 'utils' folder to Python's module search path
sys.path.append(os.path.abspath('../OBJECT_COUNTING_1/src'))

# from src.Object_Counting.config.configuration import ConfigurationManager
# from src.Object_Counting.components.prepare_base_model import PrepareBaseModel
# from src.Object_Counting import logger
from Object_Counting.config.configuration import ConfigurationManager
from Object_Counting.components.prepare_base_model import PrepareBaseModel
from Object_Counting import logger

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()

if __name__ == "__main__":
    try:
        logger.info(f"*************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e