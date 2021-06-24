# tfprof

## Usage

1. Create Profiler

    ```python
    from tfprof import Profiler, ProfilerOptions
    profiler = Profiler(options=ProfilerOptions(host_tracer_level=2, device_tracer_level=1))
    ```

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

## Outputs

ToDo(Herman):add outputs description