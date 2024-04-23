import sys
import os

# Add the 'utils' folder to Python's module search path
sys.path.append(os.path.abspath('../OBJECT_COUNTING_1/src'))

# from src.Object_Counting.config.configuration import ConfigurationManager
# from src.Object_Counting.components.evaluation import Evaluation
# from src.Object_Counting import logger

from Object_Counting.config.configuration import ConfigurationManager
from Object_Counting.components.evaluation import Evaluation
from Object_Counting import logger

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.export_model()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e