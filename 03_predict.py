from leaf import models
from leaf.inference import Predictor

# models.test()

pred = Predictor(config_name='flattened_leaves')
pred.predict(images_src='Output/ESWW0090023_12/crop', export_dst='export')