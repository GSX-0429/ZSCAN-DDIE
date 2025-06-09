import warnings
from utils.Registry import Registry

MODELS = Registry('models')
STRUCTURAL = MODELS
TEXTUAL = MODELS
CLASSIFIERS = MODELS

#Build structural model
def build_structural(cfg):
    return STRUCTURAL.build(cfg)

#Build textual model
def build_textual(cfg):
    return TEXTUAL.build(cfg)

#Build classifier
def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)
