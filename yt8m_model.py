import tensorflow as tf
import model_utils as utils

layers = tf.keras.layers

''' WILL BE CALLED FROM THIS BUILDER FUNCTION!
@register_model_builder('video_classification')
def build_video_classification_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: video_classification_cfg.VideoClassificationModel,
    num_classes: int,
    l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds the video classification model."""
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  norm_activation_config = model_config.norm_activation
  model = video_classification_model.VideoClassificationModel(
      num_classes=num_classes,
      input_specs=input_specs,
      dropout_rate=model_config.dropout_rate,
      kernel_regularizer=l2_regularizer,
      add_head_batch_norm=model_config.add_head_batch_norm,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon)
  return model
  '''


class YT8MModel(tf.keras.Model):
    
    ACT_FN_MAP = {
      "sigmoid": math.sigmoid,
      "relu6": tf.nn.relu6,
    }

  def __init__(self,
               num_classes,
               num_frames,
               iterations,
               cluster_size,
               hidden_size, 
               add_batch_norm,
               sample_random_frames,
               is_training,
               activation,
               pooling_method,
               input_specs=layers.InputSpec(shape=[None, None, None]),
               dropout_rate=0.0,
               kernel_initializer='random_normal',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """YT8M initialization function.
      Args:
        num_classes: `int` number of classes in classification task.
        input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
        dropout_rate: `float` rate for dropout regularization.
        aggregate_endpoints: `bool` aggregate all end ponits or only use the
          final end point.
        kernel_initializer: kernel initializer for the dense layer.
        kernel_regularizer: tf.keras.regularizers.Regularizer object. Default to
          None.
        bias_regularizer: tf.keras.regularizers.Regularizer object. Default to
          None.
        **kwargs: keyword arguments to be passed.
      """

    self._self_setattr_tracking = False
    self._config_dict = {
      'num_classes': num_classes,
      'num_frames' : num_frames,
      'iterations' : iterations,
      'cluster_size' : cluster_size,
      'hidden_size' : hidden_size,
      'add_batch_norm' : add_batch_norm,
      'sample_random_frames' : sample_random_frames,
      'is_training' : is_training,
      'input_specs': input_specs,
      'dropout_rate': dropout_rate,
      'aggregate_endpoints': aggregate_endpoints,
      'kernel_initializer': kernel_initializer,
      'kernel_regularizer': kernel_regularizer,
      'bias_regularizer': bias_regularizer,
    }
    self._input_specs = input_specs
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._act_fn = self.ACT_FN_MAP.get(activation)


    inputs = tf.keras.Input(shape=self._input_specs)

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if sample_random_frames:
      model_input = utils.SampleRandomFrames(inputs, num_frames, iterations)
    else:
      model_input = utils.SampleRandomSequence(inputs, num_frames, iterations)

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    # TODO: summary
    # tf.compat.v1.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = layer.BatchNormalization(name="input_bn",
                                                scale=True,
                                                center=True,
                                                is_training=is_training)(reshaped_input)

    cluster_weights = tf.Variable(tf.random_normal_initializer(stddev=1/math.sqrt(feature_size))(shape=[feature_size, cluster_size]),
                                  name="cluster_weights")

    # TODO: summary
    # tf.compat.v1.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.linalg.matmul(reshaped_input, cluster_weights)

    if add_batch_norm:
      activation = layer.BatchNormalization(name="cluster_bn",
                                                scale=True,
                                                center=True,
                                                is_training=is_training)(activation)

    else:
      cluster_biases = tf.Variable(tf.random_normal_initializer(stddev=1/math.sqrt(feature_size))(shape=[cluster_size]),
                                  name="cluster_biases")
      # TODO: summary
      # tf.compat.v1.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    
    activation = self._act_fn(activation)
    # TODO: summary
    # tf.compat.v1.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, pooling_method)

    hidden1_weights = tf.Variable(tf.random_normal_initializer(stddev=1/math.sqrt(cluster_size))(shape=[cluster_size, hidden_size]),
                                  name="hidden1_weights")

    # TODO: summary
    # tf.compat.v1.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.linalg.matmul(activation, hidden1_weights)

    if add_batch_norm:
      activation = layer.BatchNormalization(name="hidden1_bn",
                                                scale=True,
                                                center=True,
                                                is_training=is_training)(activation)


    else:
      hidden1_biases = tf.Variable(tf.random_normal_initializer(stddev=0.01)(shape=[hidden_size]),
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
    #                                        vocab_size=vocab_size,
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