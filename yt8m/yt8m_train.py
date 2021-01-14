"""YT8M model training driver."""

from absl import app
from absl import flags
import gin

from official.core import train_utils
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.modeling import performance
from official.vision.beta.projects.yt8m.configs import yt8m
from official.vision.beta.projects.yt8m.tasks import yt8m_task

FLAGS = flags.FLAGS
flags.DEFINE_string(
  'train_dir',
  default=None,
  help='The directory where the training checkpoints'
       'are stored.')

def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  print(FLAGS.flag_values_dict()) #TODO: remove (for debug)
  params = train_utils.parse_configuration(FLAGS)
  params.task.train_data.input_path='gs://youtube8m-ml/2/frame/train/train*.tfrecord'
  params.task.validation_data.input_path='gs://youtube8m-ml/2/frame/test/test*.tfrecord'
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype,
                                           params.runtime.loss_scale)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
