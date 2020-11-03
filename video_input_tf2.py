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


# def _process_image(image: tf.Tensor,
#                    is_training: bool = True,
#                    num_frames: int = 32,
#                    stride: int = 1,
#                    num_test_clips: int = 1,
#                    min_resize: int = 224,
#                    crop_size: int = 200,
#                    zero_centering_image: bool = False,
#                    seed: Optional[int] = None) -> tf.Tensor:
#   """Processes a serialized image tensor.
#   Args:
#     image: Input Tensor of shape [timesteps] and type tf.string of serialized
#       frames.
#     is_training: Whether or not in training mode. If True, random sample, crop
#       and left right flip is used.
#     num_frames: Number of frames per subclip.
#     stride: Temporal stride to sample frames.
#     num_test_clips: Number of test clips (1 by default). If more than 1, this
#       will sample multiple linearly spaced clips within each video at test time.
#       If 1, then a single clip in the middle of the video is sampled. The clips
#       are aggreagated in the batch dimension.
#     min_resize: Frames are resized so that min(height, width) is min_resize.
#     crop_size: Final size of the frame after cropping the resized frames. Both
#       height and width are the same.
#     zero_centering_image: If True, frames are normalized to values in [-1, 1].
#       If False, values in [0, 1].
#     seed: A deterministic seed to use when sampling.
#   Returns:
#     Processed frames. Tensor of shape
#       [num_frames * num_test_clips, crop_size, crop_size, 3].
#   """

#   return


def _postprocess_image(video_matrix,
                       num_frames,
                       contexts,
                       segment_labels,
                       segment_size,
                       num_classes) -> Dict[str, tf.Tensor]:

  """Processes a batched Tensor of frames.
  The same parameters used in process should be used here.
  Args:

  Returns:

  """

  if segment_labels:
    start_times = contexts["segment_start_times"].values
    # Here we assume all the segments that started at the same start time has
    # the same segment_size.
    uniq_start_times, seg_idxs = tf.unique(start_times,
                                           out_idx=tf.dtypes.int64)
    # TODO(zhengxu): Ensure the segment_sizes are all same.
    # Range gather matrix, e.g., [[0,1,2],[1,2,3]] for segment_size == 3.
    range_mtx = tf.expand_dims(uniq_start_times, axis=-1) + tf.expand_dims(
        tf.range(0, segment_size, dtype=tf.int64), axis=0)

    # Shape: [num_segment, segment_size, feature_dim].
    batch_video_matrix = tf.gather_nd(video_matrix,
                                      tf.expand_dims(range_mtx, axis=-1))
    num_segment = tf.shape(batch_video_matrix)[0]
    batch_video_ids = tf.reshape(tf.tile([contexts["id"]], [num_segment]),
                                 (num_segment,))
    batch_frames = tf.reshape(tf.tile([segment_size], [num_segment]),
                              (num_segment,))

    # For segment labels, all labels are not exhausively rated. So we only
    # evaluate the rated labels.

    ########### process label parts -> call _process_label() ###########
    # # Label indices for each segment, shape: [num_segment, 2].
    # label_indices = tf.stack([seg_idxs, contexts["segment_labels"].values],
    #                          axis=-1)
    # label_values = contexts["segment_scores"].values
    # sparse_labels = tf.sparse.SparseTensor(label_indices, label_values,
    #                                        (num_segment, num_classes))
    # batch_labels = tf.sparse.to_dense(sparse_labels, validate_indices=False)

    # sparse_label_weights = tf.sparse.SparseTensor(
    #     label_indices, tf.ones_like(label_values, dtype=tf.float32),
    #     (num_segment, num_classes))
    # batch_label_weights = tf.sparse.to_dense(sparse_label_weights,
    #                                          validate_indices=False)
    
  else:
    ########### process label parts -> call _process_label() ###########
    # # Process video-level labels.
    # label_indices = contexts["labels"].values
    # sparse_labels = tf.sparse.SparseTensor(
    #     tf.expand_dims(label_indices, axis=-1),
    #     tf.ones_like(contexts["labels"].values, dtype=tf.bool),
    #     (num_classes,))
    # labels = tf.sparse.to_dense(sparse_labels,
    #                             default_value=False,
    #                             validate_indices=False)


    # convert to batch format.
    batch_video_ids = tf.expand_dims(contexts["id"], 0)
    batch_video_matrix = tf.expand_dims(video_matrix, 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)
    batch_label_weights = None

  output_dict = {
      "video_ids": batch_video_ids,
      "video_matrix": batch_video_matrix,
      "labels": batch_labels,
      "num_frames": batch_frames,
  }
  if batch_label_weights is not None:
    output_dict["label_weights"] = batch_label_weights


  return output_dict


################ called at Parser ################
def _process_label(label: tf.Tensor,
                   one_hot_label: bool = True,
                   num_classes: Optional[int] = None) -> tf.Tensor:
  """Processes label Tensor."""

  return label


################ called at Parser ################
def get_video_matrix():



################ line 235 - 256 ################
class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(self,
               input_params: exp_cfg.DataConfig): # image_key: str = IMAGE_KEY, label_key: str = LABEL_KEY

    self._segment_labels = input_params.segment_labels
    self._feature_names = input_params.feature_names
    self._context_features = {
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    if self._segment_labels:
      self._context_features.update({
          # There is no need to read end-time given we always assume the segment
          # has the same size.
          "segment_labels": tf.io.VarLenFeature(tf.int64),
          "segment_start_times": tf.io.VarLenFeature(tf.int64),
          "segment_scores": tf.io.VarLenFeature(tf.float32)
      })
    else:
      self._context_features.update({"labels": tf.io.VarLenFeature(tf.int64)})
    
    self._sequence_features = {
        feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self._feature_names
    }

  def decode(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""

    # Read/parse frame/segment-level labels.
    contexts, features = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=self._context_features,
        sequence_features=self._sequence_features)

    return contexts, features

################ line 258 - 330 ################
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

################ line 331 - 347 ################
class PostBatchProcessor(object):
  """Processes a video and label dataset which is batched."""

  def __init__(self, input_params: exp_cfg.DataConfig):
    self._segment_labels = input_params.segment_labels
    self._segment_size = input_params.segment_size
    self._num_classes = input_params.num_classes


  def __call__(
      self, video_matrix, num_frames,
      contexts: Dict[str, tf.io.VarLenFeature(tf.int64)]) -> Dict[str, tf.Tensor]:
    ''' Partition frame-level feature matrix to segment-level feature matrix. '''

    output_dict = _postprocess_image(
      video_matrix=video_matrix,
      num_frames=num_frames,
      contexts=contexts,
      segment_labels=self._segment_labels,
      segment_size=self._segment_size,
      num_classes=self._num_classes)

    return output_dict


    # return {'image': image}, label