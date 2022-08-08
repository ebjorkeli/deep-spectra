import tensorflow as tf
from keras import backend as K
import numpy as np

def R_squared(target, prediction):
    err = tf.subtract(prediction, target)
    rest = tf.reduce_sum(tf.square(err))
    tot = tf.reduce_sum(tf.square(tf.subtract(target, tf.reduce_mean(target))))
    r2 = tf.subtract(1.0, tf.divide(rest, tot))
    return r2

def mean_squared(target, prediction):
    target = target.astype('float32')
    mse = tf.reduce_sum(tf.square(tf.subtract(target, prediction)))
    return mse

def mean_absolut(target, prediction):
    err = tf.subtract(prediction, target)
    mae = tf.reduce_sum(tf.abs(err))
    return mae

def logcosh(target, prediction):
    err = tf.subtract(prediction, target)
    lc = tf.log(tf.cosh(err))
    return lc

def binarycrossentropy(target, prediction):
    target = target.astype('float32')
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bce = BCE(target, prediction)
    return bce

def crossentropy(target, prediction):
    target = target.astype('float32')
    SCC = tf.keras.losses.SparseCategoricalCrossentropy()
    ce = SCC(target, prediction)
    return ce

def false_negative(target, prediction):
    target = target.astype('float32')
    FN = tf.keras.metrics.FalseNegatives()
    fn = FN(target, prediction)
    return fn

def recall(target, prediction):
    target = target.astype('float32')
    REC = tf.keras.metrics.Recall()
    rec = REC(target, prediction)
    return rec

def false_neg_rate(target, prediction):
    return 1 - recall(target, prediction)