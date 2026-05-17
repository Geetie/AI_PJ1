from models.multihead import DigitsResnet101
from models.transformer import TransformerDigitsModel
from models.ctc import CTCModel
from config import config


def create_model(model_type=None):
    if model_type is None:
        model_type = config.model_type
    if model_type == 'transformer':
        return TransformerDigitsModel(config.class_num, config.num_heads)
    if model_type == 'ctc':
        return CTCModel(num_classes=config.class_num)
    return DigitsResnet101(config.class_num, config.num_heads)
