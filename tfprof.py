import ujson as json
import os
import shutil
import time
import threading
import collections
import numpy
import pandas
from absl import logging
from google.protobuf import text_format
from tensorflow.core.framework import step_stats_pb2
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

  def __init__(self,  logdir='logdir', options=ProfilerOptions()):

    self._set_options(options)

    # create folders to save profile data
    self._logdir = logdir
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
    self._step_stats_list = {}

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
      if not self._step_stats_list:
        self._step_stats_list.update(
          {"current_step": -1, "step_stats": {}, "run_metadata": config_pb2.RunMetadata()})
      else:
        self._step_stats_list["run_metadata"] = config_pb2.RunMetadata()
      return self._step_stats_list["run_metadata"]

  def step_start(self):
    """
      Function to mark just before session run start
    """
    if not self._step_stats_list:
      self._step_stats_list.update(
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

    self._step_stats_list["current_step"] = step
    if run_metadata is None:
      run_metadata = self._step_stats_list['run_metadata']
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

    self._activity_stats_collection['lut'][activity_name]['end'] = time.time()
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


  def finalize(self, batch_size=0, **kwargs):
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

    Analyzer(self._logdir, batch_size, self._activity_stats_collection,
           self._step_stats_list, **kwargs).generate_summary()

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

    if step in self._step_stats_list["step_stats"]:
      # ignore duplicated steps
      logging.warning('step {} duplicated'.format(step))

    self._step_stats_list["step_stats"][step] = run_metadata.step_stats

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

def microsecond_to_nanosecond(us):
    return us * 1000

class Analyzer(object):
    # build-in activities
    _SESSION_RUN_ACTIVITY = "Session Run"

    def __init__(self, logdir, batch_size, activity_stats_collection = None, step_stats_list=None):
        self._logdir = logdir
        self._batch_size = batch_size
        self._step_stats_list = {}
        self._activity_stats_collection = {}

        assert(os.path.exists(logdir))
        if activity_stats_collection:
            self._activity_stats_collection = activity_stats_collection
        else:
            activity_stats_collection_json_file_path = os.path.join(logdir, '{}.json'.format(Analyzer._ACTIVITY_STATS_COLLECTION_FILE_NAME))
            assert os.path.exists(activity_stats_collection_json_file_path),'path {} not exist'.format(activity_stats_collection_json_file_path)
            with open(activity_stats_collection_json_file_path) as file:
                self._activity_stats_collection = json.load(file)

        if step_stats_list:
            self._step_stats_list = step_stats_list
        else:
            step_stats_folder_path = os.path.join(logdir, "step_stats")
            assert os.path.exists(step_stats_folder_path), 'path {} not exist'.format(step_stats_folder_path)
            step_stats_file_names = os.listdir(step_stats_folder_path)
            for step_stats_file_name in step_stats_file_names:
                self._step_stats_list = {"step_stats":{}}
                step = step_stats_file_name.split("_")[-1]
                step_stats_file_path = os.path.join(step_stats_folder_path, step_stats_file_name)
                with open(step_stats_file_path) as step_stats_pbtxt:
                    self._step_stats_list["step_stats"][step] = text_format.Parse(step_stats_pbtxt.read(), step_stats_pb2.StepStats())
        
    def _get_session_run_activity_name(self):
        return Analyzer._SESSION_RUN_ACTIVITY

    def generate_overall_stats(self):
        activity_stats = {}
        for step in self._activity_stats_collection:
            activity_stats[step] = {}
            for activity_name in self._activity_stats_collection[step]:
                elapse = self._activity_stats_collection[step][activity_name]["end"] - self._activity_stats_collection[step][activity_name]["start"]
                activity_stats[step][activity_name] = {"elapse":elapse}
        activity_overall_stats = {}
        activity_step_times = {}
        activity_step_times[self._get_session_run_activity_name()] = []
        for step in activity_stats:
            for session_run_activity_name in activity_step_times:
                activity_step_times[session_run_activity_name].append(activity_stats[step][session_run_activity_name]["elapse"])

        for session_run_activity_name, step_times in activity_step_times.items():
            total_time = numpy.sum(numpy.asarray(step_times))

            throughput = "N/A"
            throughput_min = "N/A"
            throughput_mean = "N/A"
            throughput_median = "N/A"
            throughput_max = "N/A"
            throughput_99th_percentile = "N/A"

            the_99th_percentile = "N/A"
            latency_mean = "N/A"
            latency_median = "N/A"
            latency_min = "N/A"
            latency_max = "N/A"

            if self._batch_size > 0 and step_times:
                throughput = self._batch_size / numpy.asarray(step_times)
                throughput_min = numpy.min(throughput)
                throughput_mean = numpy.mean(throughput)
                throughput_median = numpy.median(throughput)
                throughput_max = numpy.max(throughput)
                throughput_99th_percentile = numpy.percentile(numpy.asarray(throughput), q=99, interpolation="lower")

                the_99th_percentile = numpy.percentile(numpy.asarray(step_times), q=99, interpolation="lower")
                latency_mean = numpy.mean(numpy.asarray(step_times))
                latency_median = numpy.median(numpy.asarray(step_times))
                latency_min= numpy.min(numpy.asarray(step_times))
                latency_max= numpy.max(numpy.asarray(step_times))

            overall_stats = {
                "batch_size": self._batch_size,
                "total_time": total_time,
                "throughput_min": throughput_min,
                "throughput_mean":throughput_mean,
                "throughput_median": throughput_median,
                "throughput_max": throughput_max,
                "throughput_99th_percentile":throughput_99th_percentile,
                "latency_99th_percentile": the_99th_percentile,
                "latency_mean": latency_mean,
                "latency_median": latency_median,
                "latency_min": latency_min,
                "latency_max": latency_max,
            }
            activity_overall_stats = overall_stats
        return activity_overall_stats

    def generate_json_summary(self, **kwargs):
        summary = {}
        for name, value in kwargs.items():
            summary[name] = value
        summary_json_file_path = "{}/{}.json".format(self._logdir, Analyzer._SUMMARIES_FILE_NAME)
        summary_json = json.dumps(summary, indent = 2)
        with open(summary_json_file_path,"w+") as f:
            f.write(summary_json)
    
    # trim dupliacted information in step_stats
    def trim_step_stats(step_stats):            
        trimmed_step_stats=step_stats_pb2.StepStats()
        collector_stats_list = []
        # found duplicated stats on host
        for dev_stats in step_stats.dev_stats:
            if dev_stats.device == "/host:CPU" or "stream:all" in dev_stats.device:
                continue
            elif "job" in dev_stats.device:
                collector_stats_list.append(dev_stats)
                continue
            else:
                _dev_stats = trimmed_step_stats.dev_stats.add() 
                _dev_stats.CopyFrom(dev_stats)
        # merge duplicated host traces
        _host_stats = trimmed_step_stats.dev_stats.add()
        _host_stats.device = "/host:CPU"
        for _dev_stats in collector_stats_list:
            for node_stats in _dev_stats.node_stats:
                _node_stats = _host_stats.node_stats.add()
                _node_stats.CopyFrom(node_stats)
        return trimmed_step_stats
    
    def extract_kernel_stats(trimmed_step_stats):
        # the patten is:
        # op_id:op_type#key1=value1,key2=value2#@@kernel_name
        def _parse_kernel_node_name(node_name):
            op_id = 'unknown'
            op_type = 'unknown'
            kernel_name = 'unknown'
            kernel_name_start_index = node_name.rfind('@@')
            if kernel_name_start_index > 0:
                kernel_name = node_name[kernel_name_start_index + 2:] # skip '@@'
                # remove kernel name info
                node_name = node_name[:kernel_name_start_index]
            op_id_end_index = node_name.find(':')
            if op_id_end_index > 0:
                op_id = node_name[:op_id_end_index]
                # remove op name info
                node_name = node_name[op_id_end_index + 1:] # skip ':'
            op_type_end_index = node_name.find('#')
            if op_type_end_index > 0:
                op_type = node_name[:op_type_end_index]
                # remove op type info
                node_name = node_name[op_type_end_index + 1:] # skip '#'
            return op_id, op_type, kernel_name

        kernel_stats_list = []
        for dev_stats in trimmed_step_stats.dev_stats:
            if "gpu" in dev_stats.device and "MemcpyDtoH" not in dev_stats.device and "MemcpyHtoD" not in dev_stats.device:
                kernel_stats_list.append(dev_stats)
        kernel_stats = pandas.DataFrame(columns = ['op_type', 'op_id', 'kernel_name', 'start_ns', 'elapse_ns', 'use_tensor_core'])
        for dev_stats in kernel_stats_list:
            for node_stats in dev_stats.node_stats:
                op_id, op_type, kernel_name = _parse_kernel_node_name(node_stats.node_name)
                start_ns =  microsecond_to_nanosecond(node_stats.all_start_micros)
                elapse_ns =  microsecond_to_nanosecond(node_stats.all_end_rel_micros)
                kernel_stats = kernel_stats.append({'op_type':op_type, 'op_id':op_id, 'kernel_name':kernel_name, 'start_ns':start_ns, 'elapse_ns':elapse_ns}, ignore_index=True)
        return kernel_stats

    def extract_host_op_stats(trimmed_step_stats):
        # [allocator_name total_memory peak_memory] op_id = op_type(input_op_name1, input_op_name2, ...)
        def _parse_op_timeline_label(timeline_label):
            op_id = 'unknown'
            op_type = 'unknown'
            allocation_info_end_index = timeline_label.find(']')
            if allocation_info_end_index > 0:
                timeline_label = timeline_label[allocation_info_end_index + 1:] # just ignore memory info
            op_id_end_index = timeline_label.find('=')
            if op_id_end_index > 0:
                op_id = timeline_label[:op_id_end_index]
                op_id = op_id.strip()
                #remove op_id info
                timeline_label = timeline_label[op_id_end_index + 1:] # skip '='
            op_type_end_index = timeline_label.find('(')
            if op_type_end_index > 0:
                op_type = timeline_label[:op_type_end_index]
                op_type = op_type.strip()
                # remove op_type info
                timeline_label = timeline_label[op_type_end_index + 1:] # skip '('
            return op_id, op_type
        host_stats = None
        for dev_stats in trimmed_step_stats.dev_stats:
            if dev_stats.device == '/host:CPU':
                host_stats = dev_stats
        if host_stats is None:
            return
        host_op_stats = pandas.DataFrame(columns = ['op_type', 'op_id', 'start_ns', 'elapse_ns'])
        for node_stats in host_stats.node_stats:
            op_id, op_type = _parse_op_timeline_label(node_stats.timeline_label)
            start_ns =  node_stats.all_start_nanos
            elapse_ns =  node_stats.all_end_rel_nanos
            host_op_stats = host_op_stats.append({'op_type':op_type, 'op_id':op_id, 'start_ns':start_ns, 'elapse_ns':elapse_ns}, ignore_index=True)
        return host_op_stats 

    def generate_summary(self):
        # e2e stats
        print(self.generate_overall_stats())

        # op stats
        op_stats = pandas.DataFrame(columns = ['step', 'op_type', 'op_id', 'start_ns', 'host_elapse_ns', 'devie_elapse_ns', 'is_kernen_launch', 'run_on'])
        for step, step_stats in self._step_stats_list["step_stats"].items():
            trimmed_step_stats = Analyzer.trim_step_stats(step_stats)
            host_op_stats = Analyzer.extract_host_op_stats(trimmed_step_stats)
            kernel_stats  = Analyzer.extract_kernel_stats(trimmed_step_stats)
            op_stats = pandas.merge(host_op_stats.rename(columns={'elapse_ns':'host_elapse_ns', 'start_ns':'host_start_ns'}), 
                kernel_stats.rename(columns={'elapse_ns':'kernel_elapse_ns', 'start_ns':'kernel_start_ns'}),
                how='left', on=['op_type', 'op_id'])
        print(op_stats)
        op_stats.to_json("op_stats.json")
        op_stats.to_excel("op_stats.xlsx")
        op_stats.to_csv("op_stats.csv")