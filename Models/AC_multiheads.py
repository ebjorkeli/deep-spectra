from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, Conv1DTranspose, Flatten
from tensorflow.keras.models import Model

# Target tumor/healthy

# Adversial

class OutputHead(tf.keras.models.Model):
    def __init__(self):
