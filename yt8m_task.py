# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Video classification task definition."""
from absl import logging
import tensorflow as tf
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.beta.configs import video_classification as exp_cfg
from official.vision.beta.dataloaders import video_input
from official.vision.beta.modeling import factory_3d
from yt8m_model import YT8MModel


@task_factory.register_task_cls(exp_cfg.VideoClassificationTask)
class YT8MTask(base_task.Task):
  """A task for video classification."""

  def build_model(self):
    """Builds video classification model."""
    common_input_shape = [
        d1 if d1 == d2 else None
        for d1, d2 in zip(self.task_config.train_data.feature_shape,
                          self.task_config.validation_data.feature_shape)
    ]
    input_specs = tf.keras.layers.InputSpec(shape=[None] + common_input_shape)
    logging.info('Build model input %r', common_input_shape)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    # model = factory_3d.build_model(
    #     self.task_config.model.model_type,
    #     input_specs=input_specs,
    #     model_config=self.task_config.model,
    #     num_classes=self.task_config.train_data.num_classes,
    #     l2_regularizer=l2_regularizer)

    kernel_regularizer =
    bias_regularizer =

    model = YT8MModel(exp_cfg.DataConfig,
               input_specs=input_specs,
               dropout_rate=0.0,
               kernel_initializer='random_normal',
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer)
    return model

  def build_inputs(self, params: exp_cfg.DataConfig, input_context=None):
    """Builds classification input."""

    decoder = video_input.Decoder()
    decoder_fn = decoder.decode
    parser = video_input.Parser(input_params=params)
    postprocess_fn = video_input.PostBatchProcessor(params)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder_fn,
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=postprocess_fn)

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self, labels, model_outputs, aux_losses=None):
    """Sigmoid Cross Entropy (should be replaced by Keras implementation)
    Args:
      labels: labels.
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.
    Returns:
      The total loss tensor.
    """
    losses_config = self.task_config.losses

    return total_loss

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation.
       metric: mAP, gAP
    """
    if self.task_config.losses.one_hot:
      metrics = [
      ]
    else:
      metrics = [
      ]
    return metrics

  def process_metrics(self, metrics, labels, outputs):
    '''Prpcesses metrics'''
    return

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Does forward and backward.
    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.
    Returns:
      A dictionary of logs.
    """
    features, labels = inputs

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs =
      losses = self.build_losses(

      )


    #reimplement in tf2



    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def validation_step(self, inputs, model, metrics=None):
    """Validatation step.
    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.
    Returns:
      A dictionary of logs.
    """
    features, labels = inputs

    outputs = self.inference_step(features['image'], model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs) #TODO: check if necessary
    loss = self.build_losses(model_outputs=outputs, labels=labels,
                             aux_losses=model.losses)

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def inference_step(self, inputs, model):
    """Performs the forward step."""
    return model(inputs, training=False)