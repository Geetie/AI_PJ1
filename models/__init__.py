"""模型模块包"""
from config import config


def create_model(model_type=None):
    if model_type is None:
        model_type = config.model_type
    if model_type == 'fpn_multihead':
        from models.multihead import DigitsResnet101
        return DigitsResnet101(config.class_num, config.num_heads)
    elif model_type == 'transformer':
        from models.transformer import TransformerDigitsModel
        return TransformerDigitsModel(config.class_num, config.num_heads)
    elif model_type == 'ctc':
        from models.ctc import CTCModel
        return CTCModel(num_classes=config.class_num)
    else:
        from models.multihead import DigitsResnet101
        return DigitsResnet101(config.class_num, config.num_heads)
