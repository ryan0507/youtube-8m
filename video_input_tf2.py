'''class YT8MFrameFeatureReader(BaseReader): --> reimplement in tf2
  Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  '''


from typing import Dict, Optional, Tuple

from absl import logging
import tensorflow as tf

from official.vision.beta.configs import video_classification as exp_cfg
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops_3d

IMAGE_KEY = 'image/encoded'
LABEL_KEY = 'clip/label/index'


def _process_image(image: tf.Tensor,
                   is_training: bool = True,
                   num_frames: int = 32,
                   stride: int = 1,
                   num_test_clips: int = 1,
                   min_resize: int = 224,
                   crop_size: int = 200,
                   zero_centering_image: bool = False,
                   seed: Optional[int] = None) -> tf.Tensor:
  """Processes a serialized image tensor.
  Args:
    image: Input Tensor of shape [timesteps] and type tf.string of serialized
      frames.
    is_training: Whether or not in training mode. If True, random sample, crop
      and left right flip is used.
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    zero_centering_image: If True, frames are normalized to values in [-1, 1].
      If False, values in [0, 1].
    seed: A deterministic seed to use when sampling.
  Returns:
    Processed frames. Tensor of shape
      [num_frames * num_test_clips, crop_size, crop_size, 3].
  """

  return


def _postprocess_image(image: tf.Tensor,
                       is_training: bool = True,
                       num_frames: int = 32,
                       num_test_clips: int = 1) -> tf.Tensor:
  """Processes a batched Tensor of frames.
  The same parameters used in process should be used here.
  Args:
    image: Input Tensor of shape [batch, timesteps, height, width, 3].
    is_training: Whether or not in training mode. If True, random sample, crop
      and left right flip is used.
    num_frames: Number of frames per subclip.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
  Returns:
    Processed frames. Tensor of shape
      [batch * num_test_clips, num_frames, height, width, 3].
  """

  return image


def _process_label(label: tf.Tensor,
                   one_hot_label: bool = True,
                   num_classes: Optional[int] = None) -> tf.Tensor:
  """Processes label Tensor."""

  return label


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(self, image_key: str = IMAGE_KEY, label_key: str = LABEL_KEY):


  def decode(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""

    return {}

class Parser(parser.Parser):
  """Parses a video and label dataset."""

  def __init__(self,
               input_params: exp_cfg.DataConfig,
               image_key: str = IMAGE_KEY,
               label_key: str = LABEL_KEY):

  def _parse_train_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for training."""
    # Process image and label.

    return {'image': image}, label

  def _parse_eval_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for evaluation."""


    return {'image': image}, label


class PostBatchProcessor(object):
  """Processes a video and label dataset which is batched."""

  def __init__(self, input_params: exp_cfg.DataConfig):

  def __call__(
      self,
      image: Dict[str, tf.Tensor],
      label: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses a single tf.Example into image and label tensors."""


    return {'image': image}, label