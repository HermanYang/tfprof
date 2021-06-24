def TFLowLevelApi_TrainAndInference():
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    # Construct LeNet5 Model
    x = tf.placeholder('float', [None, 784])
    y_ = tf.placeholder('float', [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
    bias1 = tf.Variable(tf.truncated_normal([6]))
    conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding='SAME')
    h_conv1 = tf.nn.sigmoid(conv1 + bias1)
    maxPool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
    bias2 = tf.Variable(tf.truncated_normal([16]))
    conv2 = tf.nn.conv2d(maxPool2, filter2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2 = tf.nn.sigmoid(conv2 + bias2)
    maxPool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]))
    bias3 = tf.Variable(tf.truncated_normal([120]))
    conv3 = tf.nn.conv2d(maxPool3, filter3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3 = tf.nn.sigmoid(conv3 + bias3)
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]))
    b_fc1 = tf.Variable(tf.truncated_normal([80]))
    h_pool2_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    W_fc2 = tf.Variable(tf.truncated_normal([80, 10]))
    b_fc2 = tf.Variable(tf.truncated_normal([10]))
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    corrent_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    from tfprof import Profiler, ProfilerOptions

    profiler = Profiler(options=ProfilerOptions(host_tracer_level=2, device_tracer_level=1))
    run_options = tf.compat.v1.RunOptions(trace_level = profiler.get_trace_level())

    # Train
    mnist_data_set = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch_size = 256
    for step in range(100):
        # 取训练数据
        batch_xs, batch_ys = mnist_data_set.train.next_batch(batch_size)
        if step == 10:
            profiler.step_start()
        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys}, options=run_options, run_metadata=profiler.get_run_metadata())
        if step == 10:
            profiler.step_end(step)
        print("Step: {}, Accuracy: {}, Loss: {}".format(step, acc, loss))

    profiler.finalize(batch_size=batch_size)

TFLowLevelApi_TrainAndInference()