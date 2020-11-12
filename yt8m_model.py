import tensorflow as tf
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
      backbone=backbone,
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
  def __init__(self,
               num_classes,
               input_specs=layers.InputSpec(shape=[None, None, None, None, 3]),
               dropout_rate=0.0,
               aggregate_endpoints=False,
               kernel_initializer='random_uniform',
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

    inputs = tf.keras.Input(shape=input_specs.shape[1:])






    #reimplement in tf2







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