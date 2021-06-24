# tfprof

## Usage

1. Create Profiler

    ```python
    from tfprof import Profiler, ProfilerOptions
    profiler = Profiler(options=ProfilerOptions(host_tracer_level=2, device_tracer_level=1))
    ```

    profiler 性能采集 level 由 host_tracer_level 和 device_tracer_level 控制，默认 host_tracer_level=2、device_tracer_level=0

2. Integation

    ```python
    # for low level api
    run_options = tf.compat.v1.RunOptions(trace_level = profiler.get_trace_level())
    profiler.step_start()
    sess.run(..., options=run_options, run_metadata=profiler.get_run_metadata())
    profiler.step_end(step)

    # for keras api
    # not support keras distribution strategy mode, it's tensorflow bug
    run_options = tf.compat.v1.RunOptions(trace_level = profiler.get_trace_level())
    model.compile(..., options=run_options, run_metadata=profiler.get_run_metadata())
    model.fit(..., callbacks = [profiler.get_keras_profile_callback(start_step=..., end_step=...)])

    # for slim api
    slim.learning.train(..., train_step_fn=profiler.get_slim_train_step_fn(start_step=..., end_step=...)

    # for estimator api
    image_classifier = tf.estimator.Estimator(...)
    image_classifier.train(input_fn=..., hooks=[profiler.get_estimator_profile_hook(start_step=..., end_step=...)])
    ```

3. Finalize

    ```python
    profiler.finalize(batch_size = your-batch-szie) 
    ```

## ProfilerOptions

- host_tracer_level=0, device_tracer_level=0：关闭 host tracer，关闭 device tracer，会输出端到端性能数据；
- host_tracer_level=1, device_tracer_level=0: host tracer 采集 host 侧 Critical 性能信息；
- host_tracer_level=2, device_tracer_level=0: host tracer 在采集 host 侧 Critical 和 Info 等级的性能信息，详情参见 tensorflow/core/profiler/lib/traceme.h 对 host 侧性能信息等级定义；
- host_tracer_level=0, device_tracer_level=1: 只开启 device tracer，可以采集 device 端的性能数据和 mlu 运行时的性能数据；
- host_tracer_level=2, device_tracer_level=1: 开启 host tracer 和 device tracer 并采集全部性能数据；

## Outputs

ToDo(Herman):add outputs description