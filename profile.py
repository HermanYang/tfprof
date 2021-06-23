# Copyright 2020 Cambricon
# =============================================================================
from analyze import Analyzer
from absl import logging
import ujson as json
import os
import shutil
import time
import threading
import collections

from tensorflow.python.client import timeline
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training import training_util

class ProfilerOptions(
    collections.namedtuple('ProfilerOptions', [
      'host_tracer_level', 'python_tracer_level', 'device_tracer_level',
      'delay_ms'
    ])):
  """Options for finer control over the profiler.

  Use `tfprof.profiler.ProfilerOptions` to control `tfprof.profiler.Profiler`
  behavior.

  Fields:
    host_tracer_level: Adjust CPU tracing level. Values are: 1 - critical info
    only, 2 - info, 3 - verbose. [default value is 2]
    python_tracer_level: Toggle tracing of Python function calls. Values are: 1
    - enabled, 0 - disabled [default value is 0]
    device_tracer_level: Adjust device (MLU/GPU) tracing level. Values are: 1 -
    enabled, 0 - disabled [default value is 1]
    delay_ms: Requests for all hosts to start profiling at a timestamp that is
      `delay_ms` away from the current time. `delay_ms` is in milliseconds. If
      zero, each host will start profiling immediately upon receiving the
      request. Default value is None, allowing the profiler guess the best
      value.

  """

  def __new__(cls,
        host_tracer_level=2,
        python_tracer_level=0,
        device_tracer_level=1,
        delay_ms=None):
    return super(ProfilerOptions,
          cls).__new__(cls, host_tracer_level, python_tracer_level,
                device_tracer_level, delay_ms)


class Profiler(object):
  """Profiler class for application activity profiling and tensorflow profiling.
  """

  # folder names
  _TIMELINE_FOLDER_NAME = "timeline"
  _STEP_STATS_FOLDER_NAME = "step_stats"

  # file names
  _ACTIVITY_STATS_COLLECTION_JSON_FILE_NAME = "activity_stats_collection.json"

  # build-in activities
  _SESSION_RUN_ACTIVITY = "Session Run"

  def __init__(self,  logdir='profile', options=ProfilerOptions()):

    self._set_options(options)

    # create folders to save profile data
    self._logdir = os.getenv(
      'TF_PROFILE_LOGDIR', logdir)
    if os.path.exists(self._logdir):
      shutil.rmtree(self._logdir)
    os.makedirs(self._logdir)
    self._timeline_folder = "{}/{}".format(
      self._logdir, Profiler._TIMELINE_FOLDER_NAME)
    self._step_stats_folder = "{}/{}".format(
      self._logdir, Profiler._STEP_STATS_FOLDER_NAME)
    os.mkdir(self._timeline_folder)
    os.mkdir(self._step_stats_folder)

    self._activity_stats_collection = {}
    # activity lut for internal used
    self._activity_stats_collection['lut'] = {}

    self._tf_stats_collection = {}

  def _is_python_tracer_enabled(self):
    return self._options.python_tracer_level > 0

  def _is_host_tracer_enabled(self):
    return self._options.host_tracer_level > 0

  def _is_device_tracer_enabled(self):
    return self._options.device_tracer_level > 0

  def _set_options(self, options):
    assert options.host_tracer_level in [
      0, 1, 2, 3], "invalid host_tracer_level {}, should be one of [0, 1, 2, 3]".format(options.host_tracer_level)
    assert options.device_tracer_level in [
      0, 1], "invalid device_tracer_level {}, should be one of [0, 1]".format(options.device_tracer_level)
    self._options = options
  
  def _get_options(self):
    return self._options

  def get_trace_level(self):
    """Function to get tensorflow trace level depends on Profiler trace level.
    """
    if self._options.host_tracer_level == 0 and self._options.device_tracer_level == 0:
      return config_pb2.RunOptions.NO_TRACE
    elif self._options.host_tracer_level == 1 and self._options.device_tracer_level == 0:
      return config_pb2.RunOptions.SOFTWARE_TRACE
    elif self._options.host_tracer_level == 2 and self._options.device_tracer_level == 0:
      return config_pb2.RunOptions.FULL_TRACE
    elif self._options.host_tracer_level == 0 and self._options.device_tracer_level == 1:
      return config_pb2.RunOptions.HARDWARE_TRACE
    elif self._options.host_tracer_level == 2 and self._options.device_tracer_level == 1:
      return config_pb2.RunOptions.FULL_TRACE
    else:
      raise ValueError(
        'unknown profile options {}'.format(str(self._options)))

  def get_run_metadata(self):
    """
      Function to create tensorflow RunMetadata
    """
    if not (self._is_host_tracer_enabled() or self._is_device_tracer_enabled()):
      return None
    else:
      if not self._tf_stats_collection:
        self._tf_stats_collection.update(
          {"current_step": -1, "step_stats": {}, "run_metadata": config_pb2.RunMetadata()})
      else:
        self._tf_stats_collection["run_metadata"] = config_pb2.RunMetadata()
      return self._tf_stats_collection["run_metadata"]

  def step_start(self):
    """
      Function to mark just before session run start
    """
    if not self._tf_stats_collection:
      self._tf_stats_collection.update(
        {"current_step": -1, "step_stats": {}, "run_metadata": None})

    session_run_activity_name = self._get_session_run_activity_name()
    self.activity_start(session_run_activity_name)

  def step_end(self, step, run_metadata=None):
    """Function to mark just after session run end

    Args:
      run_metadata: Tensorflow run_metadata in case didn't have chances to invoke get_run_metadata linking run_metadata before
    """
    # record session run activity
    session_run_activity_name = self._get_session_run_activity_name()
    self.activity_end(session_run_activity_name, step)

    if not (self._is_host_tracer_enabled() or self._is_device_tracer_enabled()):
      return

    self._tf_stats_collection["current_step"] = step
    if run_metadata is None:
      run_metadata = self._tf_stats_collection['run_metadata']
    assert run_metadata, 'Step {} run_metadata required'.format(step)
    self._save_session_run_metadata(run_metadata, step)

  def get_keras_profile_callback(self, start_step=None, end_step=None):
    """
      Function to create keras profiler callback
    """
    return KerasProfileCallback(self, start_step, end_step)

  def get_estimator_profile_hook(self, start_step=None, end_step=None):
    '''
      Function to create estimator profiler callback
    '''
    return ProfilerHook(self, start_step, end_step)

  def activity_start(self, activity_name):
    """Function to mark just before self defined activity start

    Args:
      activity_name: activity name as a string
      step: activity step associate with training or evalutation, if step =-1 means step agnostic activity
    """
    self._activity_stats_collection['lut'][activity_name] = {
      "start": time.time(), "end": time.time()}

  def activity_end(self, activity_name, step=-1):
    """Function to mark just after self defined activity start

    Args:
      activity_name: activity name as a string
      step: activity step associate with training or evalutation, if step =-1 means step agnostic activity
    """
    assert activity_name in self._activity_stats_collection['lut'], \
      'activity {} not start yet, make sure invoke activity_start first'.format(
      activity_name)

    self._activity_stats_collection['lut'][activity_name]['end'] = time.time(
    )
    if step not in self._activity_stats_collection:
      self._activity_stats_collection[step] = {}

    if activity_name in self._activity_stats_collection[step]:
      logging.warning('step {} duplicated'.format(step))

    self._activity_stats_collection[step][activity_name] = self._activity_stats_collection['lut'][activity_name]

  def get_slim_train_step_fn(self, start_step=None, end_step=None):
    """
      Function to get slim train step function clousure link to graph name
    """
    _start_step = start_step
    _end_step = end_step
    _profiler = self
    _options = self._get_options()

    # turn off profiling(to Profiler._PROFILE_APP_ACTIVITIES) if start step > 0
    if _start_step and _start_step > 0:
      self._set_options(ProfilerOptions(host_tracer_level=0, device_tracer_level=0))

    def train_step(sess, train_op, global_step, train_step_kwargs):
      """Function that takes a gradient step and specifies whether to stop.
      
      Args:
        sess: The current session.
        train_op: An `Operation` that evaluates the gradients and returns the total
        loss.
        global_step: A `Tensor` representing the global training step.
        train_step_kwargs: A dictionary of keyword arguments.
      
      Returns:
        The total loss and a boolean indicating whether or not to stop training.
      
      """

      run_options = config_pb2.RunOptions(
        trace_level=_profiler.get_trace_level())

      _profiler.step_start()
      start_time = time.time()
      run_metadata = config_pb2.RunMetadata()
      total_loss, np_global_step = sess.run([train_op, global_step],
                          options=run_options,
                          run_metadata=run_metadata)
      time_elapsed = time.time() - start_time
      _profiler.step_end(np_global_step, run_metadata)

      # turning on/off profiling according the start step and end step config
      if _start_step and (np_global_step >= (_start_step - 1) and (not _end_step or (_end_step and np_global_step < _end_step))):
        # turn on/restore profile level
        _profiler._set_options(_options)

      elif ((_start_step and np_global_step < (_start_step - 1)) or (_end_step and np_global_step >= _end_step)):
        # turn off
        _profiler._set_options(ProfilerOptions(host_tracer_level=0, device_tracer_level=0))

      if run_metadata is not None:
        if 'summary_writer' in train_step_kwargs:
          train_step_kwargs['summary_writer'].add_run_metadata(
            run_metadata, 'run_metadata-%d' % np_global_step)

      if 'should_log' in train_step_kwargs:
        if sess.run(train_step_kwargs['should_log']):
          logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                np_global_step, total_loss, time_elapsed)

      if "should_stop" in train_step_kwargs:
        should_stop = sess.run(train_step_kwargs["should_stop"])
      else:
        should_stop = False
      return total_loss, should_stop

    return train_step


  def finalize(self, analyze=True, batch_size=0, **kwargs):
    """Function generate profile summary

    Args:
      batch_size: batch size for training or evalutation
    """
    # pop activity internal data lut before json dump
    self._activity_stats_collection.pop('lut')
    activity_stats_collection_json = json.dumps(
      self._activity_stats_collection)
    activity_stats_collection_json_path = "{}/{}".format(
      self._logdir, Profiler._ACTIVITY_STATS_COLLECTION_JSON_FILE_NAME)
    with open(activity_stats_collection_json_path, "w+") as file:
      file.write(activity_stats_collection_json)

    if analyze:
      Analyzer(self._logdir, batch_size, self._activity_stats_collection,
           self._tf_stats_collection, **kwargs).generate_summary()

  def _get_session_run_activity_name(self):
    return Profiler._SESSION_RUN_ACTIVITY

  def _save_timeline(self, step, step_stats):
    with open(os.path.join(self._timeline_folder, "{}_{}".format("timeline", step)), 'w+') as file:
      file.write(timeline.Timeline(step_stats).generate_chrome_trace_format(
        show_dataflow=True, show_memory=True))

  def _save_step_stats(self, step, step_stats):
    with open(os.path.join(self._step_stats_folder, "{}_{}".format("stats", step)), 'w+') as file:
      file.write(step_stats.__str__())

  def _save_session_run_metadata(self, run_metadata, step):
    if not (self._is_host_tracer_enabled() or self._is_device_tracer_enabled()):
      return

    if step in self._tf_stats_collection["step_stats"]:
      # ignore duplicated steps
      logging.warning('step {} duplicated'.format(step))

    self._tf_stats_collection["step_stats"][step] = run_metadata.step_stats

    # saving step_stats and timeline
    step_stats_saving_thread = threading.Thread(
      target=Profiler._save_step_stats, args=(self, step, run_metadata.step_stats))
    timeline_saving_thread = threading.Thread(
      target=Profiler._save_timeline, args=(self, step, run_metadata.step_stats))
    step_stats_saving_thread.start()
    timeline_saving_thread.start()
    step_stats_saving_thread.join()
    timeline_saving_thread.join()

class KerasProfileCallback(Callback):
  def __init__(self, profiler, start_step=None, end_step=None):
    self._profiler = profiler
    self._epoch = 0
    self._batch_count_per_epoch = 0
    self._start_step = start_step
    self._end_step = end_step
    self._options = self._profiler._get_options()

  def on_epoch_begin(self, epoch, logs=None):
    self._epoch = epoch
    # turn off profiling if start step > 0
    if self._start_step and self._start_step > 0: 
      self._profiler._set_options(ProfilerOptions(host_tracer_level=0, device_tracer_level=0))

  def on_train_batch_begin(self, batch, logs=None):
    self._profiler.step_start()

  def on_train_batch_end(self, batch, logs=None):
    if self._batch_count_per_epoch < batch + 1:
      self._batch_count_per_epoch = batch + 1
    global_step = self._epoch * self._batch_count_per_epoch + batch
    self._profiler.step_end(global_step)
    if not self._start_step and not self._end_step:
      return
    # turning on/off profiling according the start step and end step config
    elif self._start_step and (global_step >= (self._start_step - 1) and (not self._end_step or (self._end_step and global_step < self._end_step))):
      self._profiler._set_options(self._options)
    elif ((self._start_step and global_step < (self._start_step - 1)) or (self._end_step and global_step >= self._end_step)):
      # turn off
      self._profiler._set_options(ProfilerOptions(host_tracer_level=0, device_tracer_level=0))


class ProfilerHook(session_run_hook.SessionRunHook):
  def __init__(self, profiler, start_step=None, end_step=None):
    self._profiler = profiler
    self._start_step = start_step
    self._end_step = end_step
    self._options = self._profiler._get_options()

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use ProfilerHook.")
    # turn off profiling(to Profiler._PROFILE_APP_ACTIVITIES) if start step > 0
    if self._start_step and self._start_step > 0:
      self._profiler._set_options(ProfilerOptions(host_tracer_level=0, device_tracer_level=0))

  def before_run(self, run_context):
    requests = {"global_step": self._global_step_tensor}
    opts = (
      config_pb2.RunOptions(trace_level=self._profiler.get_trace_level()))
    self._profiler.step_start()
    return SessionRunArgs(requests, options=opts)

  def after_run(self, run_context, run_values):
    global_step = run_values.results['global_step']
    self._profiler.step_end(global_step, run_values.run_metadata)
    if not self._start_step and not self._end_step:
      return
    # turning on/off profiling according the start step and end step config
    if self._start_step and (global_step >= (self._start_step - 1) and (not self._end_step or (self._end_step and global_step < self._end_step))):
      # turn on/restore profile level
      self._profiler._set_options(self._options)
    elif ((self._start_step and global_step < (self._start_step - 1)) or (self._end_step and global_step >= self._end_step)):
      # turn off
      self._profiler._set_options(ProfilerOptions(host_tracer_level=0, device_tracer_level=0))
