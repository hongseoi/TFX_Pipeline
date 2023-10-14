import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from typing import Text

_TEXT_FEATURE_KEYS = [
    'title'
]

_LABEL_KEY = 'sentiment'

nltk.download('stopwords')
stopwords = stopwords.words("korean")


# Utility function for renaming the feature
def _transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'


import re

def tokenize_reviews(reviews):
    # 한국어 텍스트는 이미 소문자로 변환되어 있으므로 대문자로 변환할 필요가 없습니다.
    
    # 특수 문자 제거
    reviews = tf.strings.regex_replace(reviews, "[^가-힣a-zA-Z0-9-~$ ]", " ")
    
    # 불용어 제거 (한국어 불용어 목록을 stopwords_korean 리스트로 지정해야 함)
    reviews = tf.strings.regex_replace(reviews, r'\b(' + r'|'.join(stopwords) + r')\b\s*', " ")
    
    # 중복 공백 제거
    reviews = tf.strings.regex_replace(reviews, " +", " ")

    reviews = tf.strings.strip(reviews)

    return reviews



def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else 0.
  dense = tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
  return tf.where(tf.math.is_nan(dense), default_value, dense)


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in _TEXT_FEATURE_KEYS:
      outputs[_transformed_name(key)] = tokenize_reviews(_fill_in_missing(inputs[key]))

    outputs[_transformed_name(_LABEL_KEY)] = _fill_in_missing(inputs[_LABEL_KEY])

    return outputs