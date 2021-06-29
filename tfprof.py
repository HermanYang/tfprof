import os
import shutil
import time
import threading
import collections
import pandas
from absl import logging
from google.protobuf import text_format
import pickle
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import step_stats_pb2
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import config_pb2

# tfprof output folder names
_TIMELINE_FOLDER_NAME = 'timeline'
_STEP_STATS_FOLDER_NAME = 'step_stats'
_PARTITION_GRAPH_FOLDER_NAME = 'partition_graphs'

def microsecond_to_nanosecond(us):
    return us * 1000

class ProfilerOptions(
    collections.namedtuple('ProfilerOptions', [
        'host_tracer_level', 'device_tracer_level'
    ])):
    '''Options for finer control over the profiler.

    Use `tfprof.profiler.ProfilerOptions` to control `tfprof.profiler.Profiler`
    behavior.

    Fields:
        host_tracer_level: Adjust CPU tracing level. Values are: 1 - enable, 0 - disable [default value is 1]
        device_tracer_level: Adjust device tracing level. Values are: 1 - enabled, 0 - disabled [default value is 1]
    '''

    def __new__(cls,
        host_tracer_level=1,
        device_tracer_level=1):
        return super(ProfilerOptions,
            cls).__new__(cls, host_tracer_level, device_tracer_level)


class Profiler(object):
    '''
    Profiler class for application activity profiling and tensorflow profiling.
    '''

    def __init__(self,  logdir='logdir', options=ProfilerOptions()):
        self._set_options(options)

        # create folders to save profile data
        self._logdir = logdir
        if os.path.exists(self._logdir):
          shutil.rmtree(self._logdir)
        os.makedirs(self._logdir)
        self._timeline_folder = '{}/{}'.format(
          self._logdir, _TIMELINE_FOLDER_NAME)
        self._step_stats_folder = '{}/{}'.format(
          self._logdir, _STEP_STATS_FOLDER_NAME)
        self._partition_graph_folder = '{}/{}'.format(
          self._logdir, _PARTITION_GRAPH_FOLDER_NAME)
        os.mkdir(self._timeline_folder)
        os.mkdir(self._step_stats_folder)
        os.mkdir(self._partition_graph_folder)

        self._session_run_latency = 0
        self._run_metadata = None
        self._raw_stats = {'session_run_latency':{}, 'step_stats': {}, 'partition_graphs':{} }


    def _is_host_tracer_enabled(self):
       return self._options.host_tracer_level > 0

    def _is_device_tracer_enabled(self):
        return self._options.device_tracer_level > 0

    def _set_options(self, options):
        assert options.host_tracer_level in [
            0, 1], 'invalid host_tracer_level {}, should be one of [0, 1]'.format(options.host_tracer_level)
        assert options.device_tracer_level in [
            0, 1], 'invalid device_tracer_level {}, should be one of [0, 1]'.format(options.device_tracer_level)
        self._options = options
  
    def _get_options(self):
        return self._options

    def get_trace_level(self):
        '''
        Function to get tensorflow trace level depends on Profiler trace level.
        '''
        if self._options.host_tracer_level == 0 and self._options.device_tracer_level == 0:
          return config_pb2.RunOptions.NO_TRACE
        elif self._options.host_tracer_level == 1 and self._options.device_tracer_level == 0:
          return config_pb2.RunOptions.SOFTWARE_TRACE
        elif self._options.host_tracer_level == 0 and self._options.device_tracer_level == 1:
          return config_pb2.RunOptions.HARDWARE_TRACE
        elif self._options.host_tracer_level == 1 and self._options.device_tracer_level == 1:
          return config_pb2.RunOptions.FULL_TRACE
        else:
          raise ValueError(
            'unknown profile options {}'.format(str(self._options)))

    def get_run_metadata(self):
        '''
          Function to create tensorflow RunMetadata
        '''
        if not (self._is_host_tracer_enabled() or self._is_device_tracer_enabled()):
          return None
        else:
          if not self._run_metadata:
            self._run_metadata = config_pb2.RunMetadata()
          return self._run_metadata

    def step_start(self):
        '''
        Function to mark just before session run start
        '''
        self._session_run_latency = time.time()

    def step_end(self, step, run_metadata=None):
        '''
        Function to mark just after session run end

        Args:
          run_metadata: Tensorflow run_metadata in case didn't have chances to invoke get_run_metadata linking run_metadata before
        '''
        # record session run latency
        self._session_run_latency = time.time() - self._session_run_latency
        self._raw_stats['session_run_latency'][step] = self._session_run_latency

        if not (self._is_host_tracer_enabled() or self._is_device_tracer_enabled()):
          return

        if run_metadata is None:
          run_metadata = self._run_metadata
        assert run_metadata, 'Step {} run_metadata required'.format(step)
        self._save_session_run_metadata(run_metadata, step)

    def _save_timeline(self, step):
        with open(os.path.join(self._timeline_folder, '{}_{}'.format('timeline', step)), 'w+') as file:
          file.write(timeline.Timeline(self._raw_stats['step_stat'][step]).generate_chrome_trace_format(
            show_dataflow=True, show_memory=True))

    def _save_step_stats(self, step):
        open(os.path.join(self._step_stats_folder,
            '{}_{}'.format('step_stats', step)), 'w+').write(text_format.MessageToString(self._raw_stats['step_stats'][step]))

    def _save_partition_graphs(self, step):
        index = 0
        for partition_graph in self._raw_stats['partition_graphs'][step]:
            open(os.path.join(self._partition_graph_folder,
                '{}_{}_{}'.format('parition_graph', step, index)), 'w+').write(text_format.MessageToString(partition_graph))
            index += 1

    def _save_session_run_metadata(self, run_metadata, step):
        if not (self._is_host_tracer_enabled() or self._is_device_tracer_enabled()):
            return

        if step in self._raw_stats['step_stats']:
            # ignore duplicated steps
            logging.warning('step {} duplicated'.format(step))

        self._raw_stats['step_stats'][step] = run_metadata.step_stats
        self._raw_stats['partition_graphs'][step] = [partition_graph for partition_graph in run_metadata.partition_graphs]

        # saving step_stats and timeline
        step_stats_saving_thread = threading.Thread(
            target=Profiler._save_step_stats, args=(self, step))
        timeline_saving_thread = threading.Thread(
            target=Profiler._save_timeline, args=(self, step))
        parition_graphs_saving_thread = threading.Thread(
            target=Profiler._save_partition_graphs, args=(self, step))

        step_stats_saving_thread.start()
        timeline_saving_thread.start()
        parition_graphs_saving_thread.start()
        step_stats_saving_thread.join()
        timeline_saving_thread.join()
        parition_graphs_saving_thread.join()

    def finalize(self, batch_size=0, **kwargs):
        '''
        Function generate profile summary

        Args:
          batch_size: batch size for training or evalutation
        '''
        pickle.dump(self._raw_stats, open('{}/{}.pickle'.format(self._logdir, 'raw_stats'), 'wb'))
        Analyzer(self._logdir, batch_size).generate_summary()

class Analyzer(object):
    # build-in activities

    def __init__(self, logdir, batch_size, raw_stats=None):
        self._logdir = logdir
        self._batch_size = batch_size
        self._raw_stats = {}

        if raw_stats:
            self._raw_stats = raw_stats
        else:
            self._raw_stats = self.import_raw_stats()

    def import_raw_stats(self):
        assert(os.path.exists(self._logdir))
        return pickle.load(open('{}/{}.pickle'.format(self._logdir, 'raw_stats'), 'rb'))

    # trim dupliacted information in step_stats
    def trim_step_stats(step_stats):            
        trimmed_step_stats=step_stats_pb2.StepStats()
        collector_stats_list = []
        # found duplicated stats on host
        for dev_stats in step_stats.dev_stats:
            if dev_stats.device == '/host:CPU' or 'stream:all' in dev_stats.device:
                continue
            elif 'job' in dev_stats.device:
                collector_stats_list.append(dev_stats)
                continue
            else:
                _dev_stats = trimmed_step_stats.dev_stats.add() 
                _dev_stats.CopyFrom(dev_stats)
        # merge duplicated host traces
        _host_stats = trimmed_step_stats.dev_stats.add()
        _host_stats.device = '/host:CPU'
        for _dev_stats in collector_stats_list:
            for node_stats in _dev_stats.node_stats:
                _node_stats = _host_stats.node_stats.add()
                _node_stats.CopyFrom(node_stats)
        return trimmed_step_stats

    # the patten is:
    # op_id:op_type#key1=value1,key2=value2#@@kernel_name
    def parse_kernel_node_name(node_name):
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
    
    def extract_kernel_stats(trimmed_step_stats):
        kernel_stats_list = []
        for dev_stats in trimmed_step_stats.dev_stats:
            if 'gpu' in dev_stats.device and 'MemcpyDtoH' not in dev_stats.device and 'MemcpyHtoD' not in dev_stats.device:
                kernel_stats_list.append(dev_stats)
        kernel_stats = pandas.DataFrame(columns=['op_id', 'op_type', 'kernel_name', 'start_ns', 'elapse_ns'])
        for dev_stats in kernel_stats_list:
            for node_stats in dev_stats.node_stats:
                op_id, op_type, kernel_name = Analyzer.parse_kernel_node_name(node_stats.node_name)
                start_ns =  microsecond_to_nanosecond(node_stats.all_start_micros)
                elapse_ns =  microsecond_to_nanosecond(node_stats.all_end_rel_micros)
                kernel_stats = kernel_stats.append({
                    'op_id':op_id,
                    'op_type':op_type,
                    'kernel_name':kernel_name,
                    'start_ns':start_ns,
                    'elapse_ns':elapse_ns},
                    ignore_index=True)
        return kernel_stats

    # [allocator_name total_memory peak_memory] op_id = op_type(input_op_name1, input_op_name2, ...)
    def parse_op_timeline_label(timeline_label):
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

    def extract_op_stats(trimmed_step_stats):
        op_stats = pandas.DataFrame(columns = ['op_id', 'op_type', 'start_ns', 'elapse_ns'])
        host_stats = None
        for dev_stats in trimmed_step_stats.dev_stats:
            if dev_stats.device == '/host:CPU':
                host_stats = dev_stats
        if host_stats is None:
            return
        for node_stats in host_stats.node_stats:
            op_id, op_type = Analyzer.parse_op_timeline_label(node_stats.timeline_label)
            start_ns =  node_stats.all_start_nanos
            elapse_ns =  node_stats.all_end_rel_nanos
            op_stats = op_stats.append({
                'op_id':op_id,
                'op_type':op_type,
                'start_ns':start_ns,
                'elapse_ns':elapse_ns},
                ignore_index=True)
        return op_stats

    def get_tensor_id(op_id, index):
        return '{}:{}'.format(op_id, index)

    def extract_tensor_stats(trimmed_step_stats):
        tensor_stats = pandas.DataFrame(columns=['tensor_id', 'op_id', 'data_type', 'shape', 'theoretical_bytes', 'allocated_bytes'])
        # [allocator_name total_memory peak_memory] op_id = op_type(input_op_name1, input_op_name2, ...)
        host_stats = None
        for dev_stats in trimmed_step_stats.dev_stats:
            if dev_stats.device == '/host:CPU':
                host_stats = dev_stats
        if host_stats is None:
            return
        for node_stats in host_stats.node_stats:
            for output in node_stats.output:
                op_id, _ = Analyzer.parse_op_timeline_label(node_stats.timeline_label)
                dtype_name = dtypes.as_dtype(output.tensor_description.dtype).name
                slot = output.slot
                shape = [dim.size for dim in output.tensor_description.shape.dim]
                tensor_id=Analyzer.get_tensor_id(op_id, slot)
                theoretical_bytes = output.tensor_description.allocation_description.requested_bytes
                allocated_bytes = output.tensor_description.allocation_description.requested_bytes
                tensor_stats = tensor_stats.append({
                    'tensor_id':tensor_id,
                    'op_id':op_id,
                    'data_type':dtype_name, 
                    'shape': shape, 
                    'theoretical_bytes':theoretical_bytes,
                    'allocated_bytes':allocated_bytes},
                    ignore_index=True)
        return tensor_stats


    def parse_attr_value(value):
        if value.HasField('list'):
            return [i for i in value.list.i]
        elif value.HasField('shape'):
            return [dim.size for dim in value.shape.dim]
        elif value.HasField('i'):
            return value.i
        elif value.HasField('s'):
            return value.s
        elif value.HasField('b'):
            return value.b
        elif value.HasField('f'):
            return value.f
        elif value.HasField('type'):
            return dtypes.as_dtype(value.type).name
        else:
            raise 'unknown value: {}'.format(value)

    def parse_attrs(attrs):
        attr_dict = dict()
        for key, value in attrs.items():
            if value.HasField('tensor') or value.HasField('func') or value.HasField('placeholder'):
                continue
            attr_dict[key] = Analyzer.parse_attr_value(value)
        return attr_dict

    def extract_execution_graph_stats(partition_graphs):
        execution_graph_stats = pandas.DataFrame()
        slot_lut = dict()
        for partition_graph in partition_graphs:
            for node in partition_graph.node:
                op_id = node.name
                op_type = node.op
                input_tensor_ids = [input if ':' in input else '{}:0'.format(input) for input in node.input]
                data = {
                    'op_id':op_id,
                    'op_type':op_type,
                    'input_tensor_ids':input_tensor_ids,
                    'output_tensor_ids':[],
                    }
                attrs = Analyzer.parse_attrs(node.attr)
                data.update(attrs)
                execution_graph_stats = execution_graph_stats.append(data,
                    ignore_index=True)
                # construct slot_lut for output_tensor_ids finding
                for input_tensor_id in input_tensor_ids:
                    [op_id, slot] = input_tensor_id.split(':')
                    slot_lut.setdefault(slot, set()).add(op_id)
        #update output_tensor_id base on input_tensor_id
        for _, item in execution_graph_stats.iterrows():
            for slot in slot_lut:
                if(item.op_id in slot_lut[slot]):
                    item.output_tensor_ids.append(Analyzer.get_tensor_id(item.op_id, slot))
        return execution_graph_stats
        
    def extract_e2e_stats(session_run_latency, batch_size = 0):
        e2e_stats = pandas.DataFrame(columns = ['step', 'batch_size', 'latency', 'throughput'])
        for step, latency in session_run_latency.items():
            throughput = batch_size / latency
            e2e_stats = e2e_stats.append({'step':step, 'batch_size':batch_size, 'latency':latency, 'throughput':throughput}, ignore_index=True)
        return e2e_stats.set_index('step')

    def generate_summary(self):
        # e2e stats
        e2e_summary= Analyzer.extract_e2e_stats(self._raw_stats['session_run_latency'], self._batch_size)

        # perf stats
        perf_stats = {}
        for step, step_stats in self._raw_stats['step_stats'].items():
            trimmed_step_stats = Analyzer.trim_step_stats(step_stats)
            op_stats = Analyzer.extract_op_stats(trimmed_step_stats)
            tensor_stats = Analyzer.extract_tensor_stats(trimmed_step_stats)
            kernel_stats  = Analyzer.extract_kernel_stats(trimmed_step_stats)
            perf_stats[step] = {'op_stats':op_stats, 'tensor_stats':tensor_stats, 'kernel_stats':kernel_stats}

        # execution graph stats
        graph_stats = {}
        for step, partition_graphs in self._raw_stats['partition_graphs'].items():
            exectuion_graph_stats = Analyzer.extract_execution_graph_stats(partition_graphs)
            graph_stats[step] =exectuion_graph_stats

        # merge perf stats info and execution graph stats
        perf_summary = pandas.DataFrame()
        tensor_summary = pandas.DataFrame()
        for step in perf_stats:
            op_stats = perf_stats[step]['op_stats']
            tensor_stats = perf_stats[step]['tensor_stats']
            tensor_stats['step'] = step
            tensor_summary = tensor_summary.append(tensor_stats).set_index(['step', 'tensor_id'])
            kernel_stats = perf_stats[step]['kernel_stats']
            exectuion_graph_stats = graph_stats[step]
            op_stats= pandas.merge(op_stats.rename(columns={'elapse_ns':'host_elapse_ns', 'start_ns':'host_start_ns'}), 
                kernel_stats.rename(columns={'elapse_ns':'kernel_elapse_ns', 'start_ns':'kernel_start_ns'}),
                how='left', on=['op_id','op_type'])
            op_stats= pandas.merge(op_stats, exectuion_graph_stats, how='inner', on=['op_id', 'op_type'])
            op_stats['step'] = step
            perf_summary = perf_summary.append(op_stats)
        perf_summary = perf_summary.set_index(['step', 'op_id', 'kernel_name'])

        # dumps 
        e2e_summary.to_json(os.path.join(self._logdir, 'e2e_summary.json'))
        e2e_summary.to_excel(os.path.join(self._logdir, 'e2e_summary.xlsx'))
        perf_summary.to_json(os.path.join(self._logdir, 'perf_summary.json'))
        perf_summary.to_excel(os.path.join(self._logdir, 'perf_summary.xlsx'))
        tensor_summary.to_json(os.path.join(self._logdir, 'tensor_summary.json'))
        tensor_summary.to_excel(os.path.join(self._logdir, 'tensor_summary.xlsx'))