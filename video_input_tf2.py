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
import utils


from official.vision.beta.configs import video_classification as exp_cfg
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops_3d

IMAGE_KEY = 'image/encoded'
LABEL_KEY = 'clip/label/index'

def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be cast
      to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized


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

  # Validate parameters.
  if is_training and num_test_clips != 1:
      logging.warning(
          '`num_test_clips` %d is ignored since `is_training` is `True`.',
          num_test_clips)

  # Temporal sampler.
  if is_training:
      # Sample random clip.
      image = preprocess_ops_3d.sample_sequence(image, num_frames, True, stride,
                                                seed)
  elif num_test_clips > 1:
      # Sample linespace clips.
      image = preprocess_ops_3d.sample_linspace_sequence(image, num_test_clips,
                                                         num_frames, stride)
  else:
      # Sample middle clip.
      image = preprocess_ops_3d.sample_sequence(image, num_frames, False, stride)

  # Decode JPEG string to tf.uint8.
  image = preprocess_ops_3d.decode_jpeg(image, 3)

  # Resize images (resize happens only if necessary to save compute).
  image = preprocess_ops_3d.resize_smallest(image, min_resize)

  if is_training:
      # Standard image data augmentation: random crop and random flip.
      image = preprocess_ops_3d.crop_image(image, crop_size, crop_size, True,
                                           seed)
      image = preprocess_ops_3d.random_flip_left_right(image, seed)
  else:
      # Central crop of the frames.
      image = preprocess_ops_3d.crop_image(image, crop_size, crop_size, False)

  # Cast the frames in float32, normalizing according to zero_centering_image.
  return preprocess_ops_3d.normalize_image(image, zero_centering_image)


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
    ########## process label parts -> call _process_label() ###########
    # Process video-level labels.
    label_indices = contexts["labels"].values
    sparse_labels = tf.sparse.SparseTensor(
        tf.expand_dims(label_indices, axis=-1),
        tf.ones_like(contexts["labels"].values, dtype=tf.bool),
        (num_classes,))
    labels = tf.sparse.to_dense(sparse_labels,
                                default_value=False,
                                validate_indices=False)


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
                   serialized_exmaple: tf.Tensor,
                   feature_map: dict,
                   feature_names: tf.record,
                   seg_idxs:list,
                   one_hot_label: bool = True,
                   num_classes: Optional[int] = None,
                   ) -> tf.Tensor:
  """Processes label Tensor."""
  context_features = {
      "id" : tf.io.FixedLenFeature([], tf.string)
  }

  sequence_features = {
      feature_name: tf.io.FixedLenFeature([],dtype=tf.string)
      for feature_name in feature_names
  }
  contexts , features = tf.io.parse_single_sequence_example(
      serialized_exmaple,
      context_features=context_features,
      sequence_features= sequence_features
  )

  label_indices = contexts["labels"].values
  label_values = contexts["segment_scores"].values
  sparse_labels = tf.sparse.SparseTensor(tf.expand_dims(label_indices, axis = -1),
                                         tf.ones_like(contexts["labels"].values, dtype = tf.bool),
                                         (num_classes,))
  labels = tf.sparse.to_dense(sparse_labels, default_value= False, validate_indices= False)

  return labels

def _get_video_matrix(features, feature_size, max_frames,
                       max_quantized_value, min_quantized_value):
    """Decodes features from an input string and quantizes it.

       Args:
         features: raw feature values
         feature_size: length of each frame feature vector
         max_frames: number of frames (rows) in the output feature_matrix
         max_quantized_value: the maximum of the quantized value.
         min_quantized_value: the minimum of the quantized value.


       Returns:
         feature_matrix: matrix of all frame-features
         num_frames: number of frames in the sequence
       """
    decoded_features = tf.reshape(
        tf.cast(tf.io.decode_raw(features, tf.uint8), tf.float32), #tf.decode_raw -> tf.io.decode_raw
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features, max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

def _concat_features(features, feature_names, feature_sizes,
                    max_frames, max_quantized_value, min_quantized_value):
    '''loads (potentially) different types of features and concatenates them

        Args:
            features: raw feature values
            feature_names: list of feature names
            feature_sizes: list of features sizes
            max_frames:
            max_quantized_value:
            min_quantized_value:

        Returns:
            video_matrix: different features concatenated
            num_frames: the number of frames in the video
    '''

    num_features = len(feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(feature_names) == len(feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(feature_names), len(feature_sizes)))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
        feature_matrix, num_frames_in_this_feature = _get_video_matrix(
            features[feature_names[feature_index]],
            feature_sizes[feature_index], max_frames,
            max_quantized_value, min_quantized_value)
        if num_frames == -1:
            num_frames = num_frames_in_this_feature

        feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, max_frames)

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    return video_matrix, num_frames


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
    context, features = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=self._context_features,
        sequence_features=self._sequence_features)

    return context, features  #return entire contexts, sequences to be used in parser


class Parser(parser.Parser):
  """Parses a video and label dataset.
    takes the decoded raw tensors dict
    and parse them into a dictionary of tensors that can be consumed by the model.
    It will be executed after decoder.
  """

  def __init__(self,
               input_params: exp_cfg.DataConfig,
               max_quantized_value=2,
               min_quantized_value=-2,
               image_key: str = IMAGE_KEY,
               label_key: str = LABEL_KEY):
    self._num_frames = input_params.feature_shape[0]
    self._stride = input_params.temporal_stride
    self._num_test_clips = input_params.num_test_clips
    self._min_resize = input_params.min_image_size
    self._crop_size = input_params.feature_shape[1]
    self._one_hot_label = input_params.one_hot
    self._num_classes = input_params.num_classes
    self._image_key = image_key
    self._label_key = label_key
    self._dtype = tf.dtypes.as_dtype(input_params.dtype)

    self._segment_labels = input_params.segment_labels
    self._feature_names = input_params.feature_names
    self._feature_sizes = input_params.feature_sizes
    self._max_frames = input_params.max_frames
    self._max_quantized_value = max_quantized_value
    self._min_quantized_value = min_quantized_value



  def _parse_train_data(
      self, context, features) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:   # decoded_tensors: Dict[str, tf.Tensor]
    """Parses data for training."""
    video_matrix, num_frames = _concat_features(features, self._feature_names, self._feature_sizes,
                    self._max_frames, self.max_quantized_value, self.min_quantized_value)
    # image = _process_image(
    #     image=image,
    #     is_training=True,
    #     num_frames=self._num_frames,
    #     stride=self._stride,
    #     num_test_clips=self._num_test_clips,
    #     min_resize=self._min_resize,
    #     crop_size=self._crop_size)
    # image = tf.cast(image, dtype=self._dtype)

    return video_matrix, num_frames

  def _parse_eval_data(
          self, context, features) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:  # decoded_tensors: Dict[str, tf.Tensor]
    """Parses data for evaluation."""
    video_matrix, num_frames = _concat_features(features, self._feature_names, self._feature_sizes,
                    self._max_frames, self.max_quantized_value, self.min_quantized_value)
    # image = _process_image(
    #     image=image,
    #     is_training=False,
    #     num_frames=self._num_frames,
    #     stride=self._stride,
    #     num_test_clips=self._num_test_clips,
    #     min_resize=self._min_resize,
    #     crop_size=self._crop_size)
    # image = tf.cast(image, dtype=self._dtype)

    return video_matrix, num_frames



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