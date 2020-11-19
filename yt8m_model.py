import tensorflow as tf
import model_utils as utils
from configs import yt8m as yt8m_cfg

layers = tf.keras.layers


class YT8MModel(tf.keras.Model):
  ACT_FN_MAP = {
    "sigmoid": math.sigmoid,
    "relu6": tf.nn.relu6,
  }


def __init__(self,
             num_classes,
             num_frames,
             input_params: yt8m_cfg.YT8MModel,
             input_specs=layers.InputSpec(shape=[None, None, None]),
             **kwargs):
  """YT8M initialization function.
    Args:

      **kwargs: keyword arguments to be passed.
    """

  self._self_setattr_tracking = False
  self._config_dict = {
    'num_classes': num_classes,
    'num_frames': num_frames,
    'input_specs': input_specs,
    'iterations': input_params.iterations,
    'cluster_size': input_params.cluster_size,
    'hidden_size': input_params.hidden_size,
    'add_batch_norm': input_params.add_batch_norm,
    'sample_random_frames': input_params.sample_random_frames,
    'is_training': input_params.is_training,
    'activation': input_params.activation,
    'pooling_method': input_params.pooling_method,
  }
  self._num_classes = num_classes
  self._num_frames = num_frames
  self._input_specs = input_specs
  self._act_fn = self.ACT_FN_MAP.get(input_params.activation)

  inputs = tf.keras.Input(shape=self._input_specs)

  num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
  if input_params.sample_random_frames:
    model_input = utils.SampleRandomFrames(inputs, num_frames, input_params.iterations)
  else:
    model_input = utils.SampleRandomSequence(inputs, num_frames, input_params.iterations)

  max_frames = model_input.get_shape().as_list()[1]
  feature_size = model_input.get_shape().as_list()[2]
  reshaped_input = tf.reshape(model_input, [-1, feature_size])
  # TODO: summary
  # tf.compat.v1.summary.histogram("input_hist", reshaped_input)

  if input_params.add_batch_norm:
    reshaped_input = layer.BatchNormalization(name="input_bn",
                                              scale=True,
                                              center=True,
                                              is_training=input_params.is_training)(reshaped_input)

  cluster_weights = tf.Variable(
    tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size))(shape=[feature_size, input_params.cluster_size]),
    name="cluster_weights")

  # TODO: summary
  # tf.compat.v1.summary.histogram("cluster_weights", cluster_weights)
  activation = tf.linalg.matmul(reshaped_input, cluster_weights)

  if input_params.add_batch_norm:
    activation = layer.BatchNormalization(name="cluster_bn",
                                          scale=True,
                                          center=True,
                                          is_training=input_params.is_training)(activation)

  else:
    cluster_biases = tf.Variable(
      tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size))(shape=[input_params.cluster_size]),
      name="cluster_biases")
    # TODO: summary
    # tf.compat.v1.summary.histogram("cluster_biases", cluster_biases)
    activation += cluster_biases

  activation = self._act_fn(activation)
  # TODO: summary
  # tf.compat.v1.summary.histogram("cluster_output", activation)

  activation = tf.reshape(activation, [-1, max_frames, input_params.cluster_size])
  activation = utils.FramePooling(activation, input_params.pooling_method)

  hidden1_weights = tf.Variable(tf.random_normal_initializer(stddev=1 / math.sqrt(input_params.cluster_size))(
    shape=[input_params.cluster_size, input_params.hidden_size]),
                                name="hidden1_weights")

  # TODO: summary
  # tf.compat.v1.summary.histogram("hidden1_weights", hidden1_weights)
  activation = tf.linalg.matmul(activation, hidden1_weights)

  if input_params.add_batch_norm:
    activation = layer.BatchNormalization(name="hidden1_bn",
                                          scale=True,
                                          center=True,
                                          is_training=input_params.is_training)(activation)


  else:
    hidden1_biases = tf.Variable(tf.random_normal_initializer(stddev=0.01)(shape=[input_params.hidden_size]),
                                 name="hidden1_biases")

    # TODO: summary
    # tf.compat.v1.summary.histogram("hidden1_biases", hidden1_biases)
    activation += hidden1_biases

  activation = self._act_fn(activation)
  # TODO: summary
  # tf.compat.v1.summary.histogram("hidden1_output", activation)

  # TODO: last part of model
  # aggregated_model = getattr(video_level_models,
  #                            FLAGS.video_level_classifier_model)
  # return aggregated_model().create_model(model_input=activation,
  #                                        vocab_size=num_classes,
  #                                        **unused_params)

  super(YT8MModel, self).__init__(inputs=inputs, outputs=x, **kwargs)


@property
def checkpoint_items(self):
  """Returns a dictionary of items to be additionally checkpointed."""
  return dict(backbone=self.backbone)


def get_config(self):
  return self._config_dict


@classmethod
def from_config(cls, config, custom_objects=None):
  return cls(**config)