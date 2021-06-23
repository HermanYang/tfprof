from google.protobuf import text_format
from tensorflow.core.framework import step_stats_pb2
from multiprocessing import Process, Manager
import os
import ujson as json
import numpy
import cxxfilt
import pandas

class Analyzer(object):
    # file names 
    _ACTIVITY_SUMMARY_FILE_NAME = "activity"
    _E2E_SUMMARY_FILE_NAME = "e2e"
    _OP_DETAILS_SUMMARY_FILE_NAME = "op_details"
    _HOST_OPS_SUMMARY_FILE_NAME = "host_ops"
    _DEVICE_OPS_SUMMARY_FILE_NAME = "device_ops"
    _HOST_AND_DEVICE_OPS_SUMMARY_FILE_NAME = "host_and_device_ops"
    _MEMCPY_SUMMARY_FILE_NAME = "memcpy" 

    _SUMMARIES_FILE_NAME = "summary"
    _ACTIVITY_STATS_COLLECTION_FILE_NAME = "activity_stats_collection"

    # build-in activities
    _SESSION_RUN_ACTIVITY = "Session Run"

    def _remove_all_summary_files(self):
        file_name_list = [Analyzer._ACTIVITY_SUMMARY_FILE_NAME, 
        Analyzer._E2E_SUMMARY_FILE_NAME, Analyzer._OP_DETAILS_SUMMARY_FILE_NAME, Analyzer._HOST_OPS_SUMMARY_FILE_NAME, 
        Analyzer._DEVICE_OPS_SUMMARY_FILE_NAME, Analyzer._HOST_AND_DEVICE_OPS_SUMMARY_FILE_NAME, Analyzer._MEMCPY_SUMMARY_FILE_NAME, 
        Analyzer._SUMMARIES_FILE_NAME]
        for file_name in file_name_list:
            file_path = "{}/{}.md".format(self._profile_folder_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            file_path = "{}/{}.csv".format(self._profile_folder_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            file_path = "{}/{}.xlsx".format(self._profile_folder_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

    def __init__(self, profile_folder_path, batch_size, activity_stats_collection = None, tf_stats_collection=None, **kwargs):
        self._profile_folder_path = profile_folder_path
        self._batch_size = batch_size
        self._kwargs = kwargs
        self._tf_stats_collection = {}
        self._activity_stats_collection = {}

        self._remove_all_summary_files()
        assert(os.path.exists(profile_folder_path))
        if activity_stats_collection:
            self._activity_stats_collection = activity_stats_collection
        else:
            activity_stats_collection_json_file_path = os.path.join(profile_folder_path, '{}.json'.format(Analyzer._ACTIVITY_STATS_COLLECTION_FILE_NAME))
            assert os.path.exists(activity_stats_collection_json_file_path),'path {} not exist'.format(activity_stats_collection_json_file_path)
            with open(activity_stats_collection_json_file_path) as file:
                self._activity_stats_collection = json.load(file)

        if tf_stats_collection:
            self._tf_stats_collection = tf_stats_collection
        else:
            step_stats_folder_path = os.path.join(profile_folder_path, "step_stats")
            assert os.path.exists(step_stats_folder_path), 'path {} not exist'.format(step_stats_folder_path)
            step_stats_file_names = os.listdir(step_stats_folder_path)
            for step_stats_file_name in step_stats_file_names:
                self._tf_stats_collection = {"step_stats":{}}
                step = step_stats_file_name.split("_")[-1]
                step_stats_file_path = os.path.join(step_stats_folder_path, step_stats_file_name)
                with open(step_stats_file_path) as step_stats_pbtxt:
                    self._tf_stats_collection["step_stats"][step] = text_format.Parse(step_stats_pbtxt.read(), step_stats_pb2.StepStats())
        
    def _get_session_run_activity_name(self):
        return Analyzer._SESSION_RUN_ACTIVITY

    def _generate_activity_stats(self):
        activity_stats = {}
        for step in self._activity_stats_collection:
            activity_stats[step] = {}
            for activity_name in self._activity_stats_collection[step]:
                elapse = self._activity_stats_collection[step][activity_name]["end"] - self._activity_stats_collection[step][activity_name]["start"]
                activity_stats[step][activity_name] = {"elapse":elapse}
        return activity_stats

    def generate_op_stats(self):
        def _get_op_memory_stats(node_stats):
            memory_consumption = 0
            for memory in node_stats.memory:
                memory_consumption += memory.total_bytes
            return memory_consumption

        def _get_op_outputs_stats(node_stats):
            outputs = {}
            dtype_to_dtypename = ["UNKNOWN", "FLOAT", "DOUBLE", "INT32", "UINT8", \
                                "INT16", "INT8", "STRING", "COMPLEX64", "COMPLEX", "INT64", \
                                "BOOL", "QINT8", "QINT32", "BFLOAT16", "QINT16", "QUINT16" \
                                "UIN16", "COMPLEX128", "HALF", "RESOURCE", \
                                "VARIANT", "UINT32", "UINT64"]
            for output in node_stats.output:
                dtype = output.tensor_description.dtype
                dtypename = "Unknown"
                shape = []
                slot = output.slot
                for dim in output.tensor_description.shape.dim:
                    shape.append(dim.size)
                if dtype < len(dtype_to_dtypename):
                    dtypename = dtype_to_dtypename[dtype]
                outputs[slot] = {"data_type": dtypename, "shape": shape}
            return outputs

        def _get_op_type(node_stats):
            label = node_stats.timeline_label
            op_type = "Unknown"
            if len(label) > 0: 
                label = label.split("=")
                assert(len(label) == 2)
                label = label[1].strip()
                op_type = label[:label.find("(")]
            return op_type

        op_stats = {}
        accelerator = None

        if not self._tf_stats_collection:
            return op_stats

        for step, stats in self._tf_stats_collection["step_stats"].items():
            op_stats[step] = {}
            for dev_stats in stats.dev_stats:
                # collected by StatsCollector
                if "job" in dev_stats.device:
                    machine = dev_stats.device.split("/")[1].split(":")[1]
                    if machine not in op_stats[step]:
                        op_stats[step][machine] = {}
                    if "GPU" in dev_stats.device:
                        device = "GPU"
                        assert(accelerator == None )
                        accelerator = "GPU"
                    elif "CPU" in dev_stats.device:
                        device = "CPU"
                    else:
                        device = "Unknown Device"
                    if device not in op_stats[step][machine]:
                        op_stats[step][machine][device] = {}
                    for node_stats in dev_stats.node_stats:
                        op_type = _get_op_type(node_stats)
                        memory_consumption = _get_op_memory_stats(node_stats)
                        outputs = _get_op_outputs_stats(node_stats)
                        node_name = node_stats.node_name
                        # only record last run in loop
                        op_stats[step][machine][device][node_name] = {"op_type": op_type, "start_micros": node_stats.all_start_micros, "host_elapse": node_stats.all_end_rel_micros, "memory_consumption": memory_consumption, "outputs":outputs}

        if accelerator is None:
            return

        device_stats = self.generate_device_stats(accelerator)
        for step in op_stats:
            for machine in op_stats[step]:
                for node_name, node_stats in op_stats[step][machine][accelerator].items():
                    if node_name in device_stats[step]["kernel_stats"]:
                        op_stats[step][machine][accelerator][node_name]["kernel_stats"] = device_stats[step]['kernel_stats'][node_name]
        return op_stats
        
    def generate_device_stats(self, accelerator):
        def _generate_gpu_device_stats():
            # timeline_label: "256 bytes"
            def _gpu_memcpy_get_io(node_stats):
                label = node_stats.timeline_label
                return int(label.split(" ")[0])
        
            def _gpu_memcpy_get_tensor_name(node_stats):
                node_name = node_stats.node_name
                tensor_name = node_name.split("::")[0]
                if tensor_name.startswith("edge"):
                    return tensor_name
                else:
                    return "no_tensor"

            # node_name: "edge_714_resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm::MemcpyDtoH"
            def _gpu_memcpy_get_type(node_stats):
                node_name = node_stats.node_name
                memcpy_type = node_name.split("::")[-1]
                if "DtoH" in memcpy_type:
                    return "DtoH"
                elif "HtoD" in memcpy_type:
                    return "HtoD"
                else: 
                    raise ValueError("Memory copy type '{}' is unknown".format(memcpy_type))
        
            # node_name: "resnet_v2_50/block4/unit_2/bottleneck_v2/conv2/kernel/Regularizer/l2_regularizer/L2Loss:L2Loss#id=2592#::_ZN3cub18DeviceReduceKernelINS_18DeviceReducePolicyIfiNS_3SumEE9Policy600ENS_22TransformInputIteratorIfN10tensorflow10squareHalfIfEEPflEES9_iS2_EEvT0_T1_T2_NS_13GridEvenShareISD_EET3_"
            def _gpu_kernel_get_op_type(node_stats):
                node_name = node_stats.node_name
                op_type = node_name.split("/")
                if len(op_type[-1].split(":")) < 2: return "unknown"
                op_type = op_type[-1].split(":")[1]
                op_type = op_type.split("#")[0]
                return op_type

            # node_name: "resnet_v2_50/block4/unit_2/bottleneck_v2/conv2/kernel/Regularizer/l2_regularizer/L2Loss:L2Loss#id=2592#::_ZN3cub18DeviceReduceKernelINS_18DeviceReducePolicyIfiNS_3SumEE9Policy600ENS_22TransformInputIteratorIfN10tensorflow10squareHalfIfEEPflEES9_iS2_EEvT0_T1_T2_NS_13GridEvenShareISD_EET3_"
            def _gpu_kernel_get_kernel_name(node_stats):
                node_name = node_stats.node_name
                if len(node_name.split("::")) < 2: return "unknown"
                return cxxfilt.demangle(node_name.split("::")[1])

            # node_name: "resnet_v2_50/block4/unit_2/bottleneck_v2/conv2/kernel/Regularizer/l2_regularizer/L2Loss:L2Loss#id=2592#::_ZN3cub18DeviceReduceKernelINS_18DeviceReducePolicyIfiNS_3SumEE9Policy600ENS_22TransformInputIteratorIfN10tensorflow10squareHalfIfEEPflEES9_iS2_EEvT0_T1_T2_NS_13GridEvenShareISD_EET3_"
            def _gpu_kernel_get_node_name(node_stats):
                node_name = node_stats.node_name
                return node_name.split(":")[0]

            device_stats = {}
            for step, stats in self._tf_stats_collection["step_stats"].items():
                device_stats[step] = {}
                device_stats[step]["kernel_stats"] = {}
                device_stats[step]["memcpy_stats"] = [] 
                for dev_stats in stats.dev_stats:
                    if "gpu" in dev_stats.device and ("MemcpyDtoH" in dev_stats.device or "MemcpyHtoD" in dev_stats.device):
                        # Memcpy 
                        for node_stats in dev_stats.node_stats:
                            io_bytes = _gpu_memcpy_get_io(node_stats)
                            memcpy_type = _gpu_memcpy_get_type(node_stats)
                            tensor_name = _gpu_memcpy_get_tensor_name(node_stats)
                            device_stats[step]["memcpy_stats"].append({"tensor_name":tensor_name, "start_micros":node_stats.all_start_micros, "elapse":node_stats.all_end_rel_micros, "memcpy_type":memcpy_type, "io_bytes":io_bytes })
                    elif "gpu" in dev_stats.device:
                        # Kernel
                        for node_stats in dev_stats.node_stats:
                            op_type = _gpu_kernel_get_op_type(node_stats)
                            kernel_name = _gpu_kernel_get_kernel_name(node_stats)
                            node_name = _gpu_kernel_get_node_name(node_stats)
                            if node_name not in device_stats[step]["kernel_stats"]:
                                device_stats[step]["kernel_stats"][node_name] = []
                            device_stats[step]["kernel_stats"][node_name].append({"op_type":op_type, "start_micros":node_stats.all_start_micros, "elapse":node_stats.all_end_rel_micros, "kernel_name":kernel_name})
            return device_stats

        def _generate_mlu_device_stats():
            # node_name: "gradients/resnet_v2_50/block1/unit_2/bottleneck_v2/conv1/QuantConvolutionWrapper_grad/Shape:Const"
            def _mlu_kernel_get_op_type(node_stats):
                return node_stats.node_name.split(":")[-1]

            # node_name: "gradients/resnet_v2_50/block1/unit_2/bottleneck_v2/conv1/QuantConvolutionWrapper_grad/Shape:Const"
            def _mlu_kernel_get_node_name(node_stats):
                return node_stats.node_name.split(":")[0]

            device_stats = {}
            for step, stats in self._tf_stats_collection["step_stats"].items():
                device_stats[step] = {}
                device_stats[step]["kernel_stats"] = {}
                device_stats[step]["memcpy_stats"] = [] 
                for dev_stats in stats.dev_stats:
                    if "MLU" in dev_stats.device and "Kernel" in dev_stats.device:
                        # Kernel
                        for node_stats in dev_stats.node_stats:
                            op_type = _mlu_kernel_get_op_type(node_stats)
                            kernel_name = "MLU Kernel"
                            node_name = _mlu_kernel_get_node_name(node_stats)
                            if node_name not in device_stats[step]["kernel_stats"]:
                                device_stats[step]["kernel_stats"][node_name] = []
                            device_stats[step]["kernel_stats"][node_name].append({"op_type":op_type, "start_micros":node_stats.all_start_micros, "elapse":node_stats.all_end_rel_micros, "kernel_name":kernel_name})
            return device_stats

        if accelerator == "GPU":
            return _generate_gpu_device_stats()
        elif accelerator == "MLU":
            return _generate_mlu_device_stats()
        else:
            return {}

    def generate_overall_stats(self, activity_stats):
        activity_overall_stats = {}
        activity_step_times = {}
        # ignore first step
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

    def _generate_activity_summary(self, activity_stats):
        if not activity_stats:
            return
        activity_stats_pandas = pandas.DataFrame(columns=['Step', 'Activity', 'Elapse(s)'])
        for step, _ in sorted(activity_stats.items(), key=lambda items:items[0]):
            for activity_name in activity_stats[step]:
                activity_stats_pandas = activity_stats_pandas.append({'Step':step, 'Activity':activity_name, 'Elapse(s)':activity_stats[step][activity_name]["elapse"]}, ignore_index = True)
        #csv
        summary_csv_file_path = "{}/{}.csv".format(self._profile_folder_path, Analyzer._ACTIVITY_SUMMARY_FILE_NAME)
        activity_stats_pandas.to_csv(summary_csv_file_path, index = False, sep = ',')
        #xlsx
        summary_xlsx_file_path = "{}/{}.xlsx".format(self._profile_folder_path, Analyzer._ACTIVITY_SUMMARY_FILE_NAME)
        activity_stats_pandas.to_excel(summary_xlsx_file_path, index = False)


    def generate_e2e_stats(self, activity_stats, op_stats, op_summary_stats):
        # graph execution start from CPU device and end at CPU device anyhow
        def _get_graph_execution_duration(op_stats):
            MICROS_TO_SECOND = 1000000
            start_micros = -1
            end_micros = -1
            for machine in op_stats:
                for device in op_stats[machine]:
                    for node_name, op_node in op_stats[machine][device].items():
                        if start_micros < 0 or op_node["start_micros"] < start_micros:
                            start_micros = op_node["start_micros"]
                        end_micros_tmp = op_node["start_micros"] + op_node["host_elapse"]
                        if end_micros_tmp > end_micros:
                            end_micros = end_micros_tmp
            return float(end_micros - start_micros + 1) / MICROS_TO_SECOND

        def _get_device_memcpy_sum(memcpy_stats):
            sum = 0
            for memcpy in memcpy_stats:
                sum += int(memcpy['elapse'])
            return sum

        e2e_stats = {}
        if not activity_stats:
            return e2e_stats

        step_elaspe=[]
        session_run_activity_name = self._get_session_run_activity_name()
        for step in activity_stats:
            e2e = activity_stats[step][session_run_activity_name]["elapse"]
            step_elaspe.append(e2e)
            throughput = 'N/A'
            device_op_elapse = 'N/A'
            memcpy_elapse = 'N/A'
            host_op_elapse = 'N/A'
            if self._batch_size > 0:
                throughput = self._batch_size / e2e
            e2e_stats[step] = {'batch_size':self._batch_size, 'e2e':e2e, 'throughput':throughput, 'host_op_elapse':host_op_elapse, 'device_op_elapse': device_op_elapse, 'memcpy_elapse': memcpy_elapse }

        if not op_stats:
            return e2e_stats

        step_elaspe=[]
        for step in op_stats:
            e2e = _get_graph_execution_duration(op_stats[step])
            step_elaspe.append(e2e)
            throughput = 'N/A'
            if self._batch_size > 0:
                throughput = self._batch_size / e2e 
            e2e_stats[step]['e2e'] = e2e
            e2e_stats[step]['throughput'] = throughput
        return e2e_stats

    def _generate_e2e_summary(self, e2e_stats):
        if not e2e_stats:
            return
        e2e_stats_pandas = pandas.DataFrame(columns=['Step', 'E2E(s)', 'Throughput(fps)', 'Host OP Elapse(s)', 'Device Elapse(s)', 'Memcpy Elapse(s)'])
        for step, stats in sorted(e2e_stats.items(), key=lambda item: item[0]):
            e2e_stats_pandas = e2e_stats_pandas.append({'Step':step, 'E2E(s)':stats['e2e'], 'Throughput(fps)':stats["throughput"], 'Host OP Elapse(s)':stats["host_op_elapse"], 'Device Elapse(s)':stats["device_op_elapse"], 'Memcpy Elapse(s)':stats["memcpy_elapse"]}, ignore_index = True)
        #csv
        summary_csv_file_path = "{}/{}.csv".format(self._profile_folder_path, Analyzer._E2E_SUMMARY_FILE_NAME)
        e2e_stats_pandas.to_csv(summary_csv_file_path, index = False, sep = ',')
        #xlsx
        summary_xlsx_file_path = "{}/{}.xlsx".format(self._profile_folder_path, Analyzer._E2E_SUMMARY_FILE_NAME)
        e2e_stats_pandas.to_excel(summary_xlsx_file_path, index = False)


    def _generate_host_and_device_ops_summary(self, op_summary_stats):
        if not op_summary_stats:
            return
        op_summary_stats_pandas = pandas.DataFrame(columns=['Step', 'Type', 'Device', 'Host Elapse(us)', 'Device Elapse(us)'])
        for step, stats in sorted(op_summary_stats.items(), key=lambda item: item[0]) :
            for op_type, device_summary in sorted(stats.items(), key=lambda item: item[0]):
                for device, op_summary in sorted(device_summary.items(), key=lambda item: item[0]):
                    op_summary_stats_pandas = op_summary_stats_pandas.append({'Step':step, 'Type':op_type, 'Device':device, 'Host Elapse(us)':op_summary["host_elapse"], 'Device Elapse(us)':op_summary["device_elapse"], 'Count':op_summary["count"], 'Total Memory(bytes)':op_summary["memory_consumption"]}, ignore_index = True)
        #csv
        summary_csv_file_path = "{}/{}.csv".format(self._profile_folder_path, Analyzer._HOST_AND_DEVICE_OPS_SUMMARY_FILE_NAME)
        op_summary_stats_pandas.to_csv(summary_csv_file_path, index = False, sep = ',')
        #xlsx
        summary_xlsx_file_path = "{}/{}.xlsx".format(self._profile_folder_path, Analyzer._HOST_AND_DEVICE_OPS_SUMMARY_FILE_NAME)
        op_summary_stats_pandas.to_excel(summary_xlsx_file_path, index = False)

    def _generate_host_ops_summary(self, host_op_summary_stats):
        if not host_op_summary_stats:
            return
        host_op_summary_stats_pandas = pandas.DataFrame(columns=['Step', 'Type', 'Host Elapse(us)', 'Count', 'Total Memory(bytes)'])
        for step, stats in sorted(host_op_summary_stats.items(), key=lambda item: item[0]):
            for op_type, op_summary in sorted(stats.items(), key=lambda item: item[0]):
                host_op_summary_stats_pandas = host_op_summary_stats_pandas.append({'Step':step, 'Type':op_type, 'Host Elapse(us)':op_summary["host_elapse"], 'Count':op_summary["count"], 'Total Memory(bytes)':op_summary["memory_consumption"]}, ignore_index = True)
        #csv
        summary_csv_file_path = "{}/{}.csv".format(self._profile_folder_path, Analyzer._HOST_OPS_SUMMARY_FILE_NAME)
        host_op_summary_stats_pandas.to_csv(summary_csv_file_path, index = False, sep = ',')
        #xlsx
        summary_xlsx_file_path = "{}/{}.xlsx".format(self._profile_folder_path, Analyzer._HOST_OPS_SUMMARY_FILE_NAME)
        host_op_summary_stats_pandas.to_excel(summary_xlsx_file_path, index = False)

    def _generate_op_details_summary(self, op_stats):
        def _get_device_compute_sum(kernel_stats):
            sum = 0
            for op_kernel in kernel_stats:
                sum += op_kernel['elapse']
            return sum

        if not op_stats:
            return
        op_stats_pandas = pandas.DataFrame(columns=['Step','Type','Name','Data Info','Device','Host Elapse(us)','Device Elapse(us)','Memory(bytes)'])
        for step, stats in sorted(op_stats.items(), key=lambda item: item[0]):
            for machine in stats:
                for device in stats[machine]:
                    for node_name in stats[machine][device]:
                        node = stats[machine][device][node_name]
                        device_elapse = 0
                        if 'kernel_stats' in node:
                            device_elapse = _get_device_compute_sum(node['kernel_stats'])
                        data_info = "N/A"
                        if "outputs" in node: 
                            data_info_list = []
                            for slot, output in node["outputs"].items():
                                data_info_list.append("{}{}".format(output["data_type"], output["shape"]))
                            data_info = ",".join(data_info_list)
                        op_stats_pandas = op_stats_pandas.append({'Step':step, 'Type':node["op_type"], 'Name':node_name, 'Data Info':data_info, 'Device':device, 'Host Elapse(us)':node["host_elapse"], 'Device Elapse(us)':device_elapse, 'Memory(bytes)':node["memory_consumption"]}, ignore_index = True)
        #csv
        summary_csv_file_path = "{}/{}.csv".format(self._profile_folder_path, Analyzer._OP_DETAILS_SUMMARY_FILE_NAME)
        op_stats_pandas.to_csv(summary_csv_file_path, index = False, sep = ',')
        #xlsx
        summary_xlsx_file_path = "{}/{}.xlsx".format(self._profile_folder_path, Analyzer._OP_DETAILS_SUMMARY_FILE_NAME)
        op_stats_pandas.to_excel(summary_xlsx_file_path, index = False)
                                        
    def _generate_device_ops_summary(self, device_op_summary_stats):
        if not device_op_summary_stats:
            return
        device_op_summary_stats_pandas =  pandas.DataFrame(columns=['Step', 'Type', 'Host Elapse(us)', 'Device Elapse(us)', 'Count', 'Memory Consumption(bytes)'])
        for step, stats in sorted(device_op_summary_stats.items(), key=lambda item: item[0]):
            for op_type, op_summary in sorted(stats.items(), key=lambda item: item[0]):
                device_op_summary_stats_pandas = device_op_summary_stats_pandas.append({'Step':step, 'Type':op_type, 'Host Elapse(us)':op_summary["host_elapse"], 'Device Elapse(us)':op_summary["device_elapse"], 'Count':op_summary["count"], 'Memory Consumption(bytes)':op_summary["memory_consumption"]}, ignore_index = True)
        #csv
        summary_csv_file_path = "{}/{}.csv".format(self._profile_folder_path, Analyzer._DEVICE_OPS_SUMMARY_FILE_NAME)
        device_op_summary_stats_pandas.to_csv(summary_csv_file_path, index = False, sep = ',')
        #xlsx
        summary_xlsx_file_path = "{}/{}.xlsx".format(self._profile_folder_path, Analyzer._DEVICE_OPS_SUMMARY_FILE_NAME)
        device_op_summary_stats_pandas.to_excel(summary_xlsx_file_path, index = False)

    def _generate_host_and_device_op_summary_stats(self, op_stats):
        def _sum_duration(nodes):
            elapse = 0
            for node in nodes:
                elapse = elapse + node["elapse"]
            return elapse

        op_summary_stats = {}
        if not op_stats:
            return op_summary_stats

        for step in op_stats:
            op_summary_stats[step] = {}
            for machine in op_stats[step]:
                for device in op_stats[step][machine]:
                    for node_name in op_stats[step][machine][device]:
                        op_nodes = op_stats[step][machine][device][node_name]
                        kernel_stats = []
                        if 'kernel_stats' in op_stats[step][machine][device][node_name]:
                            kernel_stats = op_stats[step][machine][device][node_name]["kernel_stats"]
                        op_type = op_nodes["op_type"]
                        host_elapse = op_nodes['host_elapse']
                        device_elapse = _sum_duration(kernel_stats)
                        memory_consumption = op_nodes['memory_consumption']
                        if op_type not in op_summary_stats[step]:
                            op_summary_stats[step][op_type] = {}
                        
                        if device not in op_summary_stats[step][op_type]:
                            op_summary_stats[step][op_type][device] = {"host_elapse":0, "device_elapse":0, "count":0, "memory_consumption":0}
                        op_summary_stats[step][op_type][device]["host_elapse"] = op_summary_stats[step][op_type][device]["host_elapse"] + host_elapse
                        op_summary_stats[step][op_type][device]["device_elapse"] = op_summary_stats[step][op_type][device]["device_elapse"] + device_elapse
                        op_summary_stats[step][op_type][device]["count"] = op_summary_stats[step][op_type][device]["count"] + 1
                        op_summary_stats[step][op_type][device]["memory_consumption"] = op_summary_stats[step][op_type][device]["memory_consumption"] + memory_consumption
        return op_summary_stats

    def _generate_host_op_summary_stats(self, op_summary_stats):
        host_op_summary_stats = {}
        for step in op_summary_stats:
            host_op_summary_stats[step] = {}
            for op_type, op_summary in op_summary_stats[step].items():
                for device in op_summary:
                    if device == 'CPU':
                        host_op_summary_stats[step][op_type] = op_summary_stats[step][op_type][device]
        return host_op_summary_stats

    def generate_device_op_summary_stats(self, op_summary_stats):
        device_op_summary_stats = {}
        for step in op_summary_stats:
            device_op_summary_stats[step] = {}
            for op_type, op_summary in op_summary_stats[step].items():
                for device in op_summary:
                    if device != "CPU":
                        device_op_summary_stats[step][op_type] = op_summary_stats[step][op_type][device]
        return device_op_summary_stats

    def generate_memcpy_summary_stats(self, device_stats):
        memcpy_summary_stats = {}
        for step in device_stats:
            memcpy_summary_stats[step] = {"HtoD":{"elapse":0, "io_bytes":0, "count":0}, "DtoH":{"elapse":0, "io_bytes":0, "count":0}}
            memcpy_stats = device_stats[step]["memcpy_stats"]
            for stats in memcpy_stats:
                memcpy_summary_stats[step][stats["memcpy_type"]]["elapse"] += stats["elapse"]
                memcpy_summary_stats[step][stats["memcpy_type"]]["io_bytes"] += stats["io_bytes"]
                memcpy_summary_stats[step][stats["memcpy_type"]]["count"] += 1
        return memcpy_summary_stats

    def _generate_memcpy_summary(self, memcpy_summary_stats):
        if not memcpy_summary_stats:
                return
        memcpy_summary_stats_pandas = pandas.DataFrame(columns = ['Step', 'Type', 'Elapse(us)', 'IO(bytes)', 'Count'])
        for step, stats in sorted(memcpy_summary_stats.items(), key=lambda item: item[0]):
            for type, device_memory_summary in stats.items():
                memcpy_summary_stats_pandas = memcpy_summary_stats_pandas.append({'Step':step, 'Type':type, 'Elapse(us)':device_memory_summary["elapse"], 'IO(bytes)':device_memory_summary["io_bytes"], 'Count':device_memory_summary["count"]}, ignore_index = True)
        #csv
        summary_csv_file_path = "{}/{}.csv".format(self._profile_folder_path, Analyzer._MEMCPY_SUMMARY_FILE_NAME)
        memcpy_summary_stats_pandas.to_csv(summary_csv_file_path, index = False, sep = ',')
        #xlsx
        summary_xlsx_file_path = "{}/{}.xlsx".format(self._profile_folder_path, Analyzer._MEMCPY_SUMMARY_FILE_NAME)
        memcpy_summary_stats_pandas.to_excel(summary_xlsx_file_path, index = False)

    def generate_json_summary(self, **kwargs):
        summary = {}
        for name, value in kwargs.items():
            summary[name] = value
        summary_json_file_path = "{}/{}.json".format(self._profile_folder_path, Analyzer._SUMMARIES_FILE_NAME)
        summary_json = json.dumps(summary, indent = 2)
        with open(summary_json_file_path,"w+") as f:
            f.write(summary_json)
    
    # def generate_summary(self):
    #     # generate activity summary
    #     activity_stats = self._generate_activity_stats()
    #     self._generate_activity_summary(activity_stats)

    #     # generate overall stats
    #     overall_stats = self.generate_overall_stats(activity_stats)

    #     # generate op stats
    #     op_stats = self.generate_op_stats()
    #     self._generate_op_details_summary(op_stats)

    #     # generate op summary
    #     op_summary_stats = self._generate_host_and_device_op_summary_stats(op_stats)
    #     self._generate_host_and_device_ops_summary(op_summary_stats)
    #     host_op_summary_stats = self._generate_host_op_summary_stats(op_summary_stats)
    #     self._generate_host_ops_summary(host_op_summary_stats)
    #     device_op_summary_stats = self.generate_device_op_summary_stats(op_summary_stats)
    #     self._generate_device_ops_summary(device_op_summary_stats)

    #     # generate e2e summary
    #     e2e_stats = self.generate_e2e_stats(activity_stats=activity_stats, op_stats=op_stats, op_summary_stats=op_summary_stats)
    #     self._generate_e2e_summary(e2e_stats=e2e_stats)

    #     # dump all stats in json format
    #     self.generate_json_summary(activity_stats=activity_stats, op_stats=op_stats, e2e_stats=e2e_stats, op_summary_stats=op_summary_stats,
    #                            host_op_summary_stats=host_op_summary_stats, device_op_summary_stats=device_op_summary_stats, overall_stats=overall_stats, **self._kwargs)
    def generate_summary(self):
        pass


