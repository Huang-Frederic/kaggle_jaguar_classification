import tensorflow as tf
import keras
import numpy as np

def my_softmax(x: keras.KerasTensor):
    x_max = keras.ops.max(x, 1, keepdims=True)
    negative_or_null_x = x - x_max
    exp_x = keras.ops.exp(negative_or_null_x)
    return exp_x / keras.ops.sum(exp_x, axis=1, keepdims=True)

def my_categorical_cross_entropy(y_true: keras.KerasTensor, y_pred: keras.KerasTensor):
    loss = - keras.ops.mean(
        keras.ops.sum(y_true * keras.ops.log(y_pred + tf.constant(1e-7, dtype=tf.float32)), axis=1, keepdims=True)
    )
    return loss

def create_really_small_mlp():
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation=keras.activations.tanh),
        keras.layers.Dense(10, activation=my_softmax)
    ])
    return model

def create_small_mlp():
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation=keras.activations.tanh),
        keras.layers.Dense(16, activation=keras.activations.tanh),
        keras.layers.Dense(10, activation=my_softmax)
    ])
    return model


def create_deep_mlp():
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        *[keras.layers.Dense(32, activation=keras.activations.tanh) for _ in range(25)],
        keras.layers.Dense(16, activation=keras.activations.tanh),
        keras.layers.Dense(10, activation=my_softmax)
    ])
    return model

def create_small_conv_net():
    model = keras.models.Sequential([# 28, 28, 1
        keras.layers.Conv2D(4, (3, 3), (1, 1), padding='valid', activation=keras.activations.tanh), # 26,26,4
        keras.layers.Conv2D(4, (3, 3), (1, 1), padding='valid', activation=keras.activations.tanh), # 24,24,4
        keras.layers.MaxPooling2D((2, 2)),#12,12,4

        keras.layers.Conv2D(8, (3, 3), (1, 1), padding='valid', activation=keras.activations.tanh), # 10,10,8
        keras.layers.Conv2D(8, (3, 3), (1, 1), padding='valid', activation=keras.activations.tanh), # 8,8,8
        keras.layers.MaxPooling2D((2, 2)), # 4,4,8

        keras.layers.Flatten(),
        keras.layers.Dense(16, activation=keras.activations.tanh),
        keras.layers.Dense(10, activation=my_softmax)
    ])
    return model

def create_small_residual_dense_net(input_shape):
    input_tensor = keras.Input(input_shape)
    hidden_tensor = keras.layers.Flatten()(input_tensor)
    hidden_tensor = keras.layers.Dense(16, activation=keras.activations.tanh)(hidden_tensor)
    previous_tensor = hidden_tensor
    hidden_tensor = keras.layers.Dense(16, activation=keras.activations.tanh)(hidden_tensor)
    hidden_tensor = keras.layers.Add()([previous_tensor, hidden_tensor])
    previous_tensor = hidden_tensor
    hidden_tensor = keras.layers.Dense(16, activation=keras.activations.tanh)(hidden_tensor)
    hidden_tensor = keras.layers.Add()([previous_tensor, hidden_tensor])
    previous_tensor = hidden_tensor
    hidden_tensor = keras.layers.Dense(16, activation=keras.activations.tanh)(hidden_tensor)
    hidden_tensor = keras.layers.Add()([previous_tensor, hidden_tensor])
    previous_tensor = hidden_tensor
    hidden_tensor = keras.layers.Dense(16, activation=keras.activations.tanh)(hidden_tensor)
    hidden_tensor = keras.layers.Add()([previous_tensor, hidden_tensor])
    output_tensor = keras.layers.Dense(10, activation=my_softmax)(hidden_tensor)

    model = keras.Model(input_tensor, output_tensor)
    return model

@tf.function
def gradient_step(model,
                  optimizer,
                  batch_x_train,
                  batch_y_train,
                  loss_function,
                  learning_rate,
                  V,
                  S,
                  gamma_V,
                  gamma_S,
                  t_EMA):
    W = model.trainable_variables
    with tf.GradientTape() as tape:
        batch_y_pred = model(batch_x_train)
        loss = loss_function(batch_y_train, batch_y_pred)
    accuracy = keras.metrics.categorical_accuracy(batch_y_train, batch_y_pred)
    grads = tape.gradient(loss, W)

    for (i, grad) in enumerate(grads):
        w = W[i]
        if optimizer == "SGD":
            w.assign(w - learning_rate * grad)
        elif optimizer == "SGD_with_momentum":
            v = V[i]
            v.assign(gamma_V * v - learning_rate * grad)
            w.assign(w + v)
        elif optimizer == "SGD_with_EMA_momentum":
            v = V[i]
            v.assign(gamma_V * v + (1.0 - gamma_V) * grad)
            w.assign(w - learning_rate * v)
        elif optimizer == "RMS_Prop":
            s = S[i]
            s.assign(gamma_S * s + (1.0 - gamma_S) * (grad ** 2.0))
            w.assign(w - learning_rate * grad / keras.ops.sqrt(s + 1e-7))
        elif optimizer == "Adam":
            v = V[i]
            s = S[i]
            v.assign(gamma_V * v + (1.0 - gamma_V) * grad)
            s.assign(gamma_S * s + (1.0 - gamma_S) * (grad ** 2.0))
            v_hat = v / (1.0 - gamma_V ** (t_EMA + 1.0))
            s_hat = s / (1.0 - gamma_S ** (t_EMA + 1.0))
            w.assign(w - learning_rate * v_hat / keras.ops.sqrt(s_hat + 1e-7))
        elif optimizer == "SignSGD":
            w.assign(w - learning_rate * keras.ops.sign(grad))

    return loss, accuracy

def run_experiment(
        dataset,
        seeds,
        model_init_func,
        loss_function,
        optimizer, # 'SGD', 'SGD_with_momentum', 'SGD_with_EMA_momentum', 'RMS_Prop', 'Adam', 'SignSGD'
        experiment_name,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 128,
        gamma_V: float = 0.9,
        gamma_S: float = 0.99,
):
    keras.optimizers.Adam()
    for seed in seeds:
        (x_train, y_train), (x_test, y_test) = dataset
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        model = model_init_func()
        _ = model.predict(x_train[0:batch_size])
        real_experiment_name = f"{experiment_name}_{seed}"

        model.summary()
        model.save(f'./models/{real_experiment_name}.keras')

        keras.utils.set_random_seed(seed)
        train_sample_count = len(x_train)
        test_sample_count = len(x_test)

        tb_log_writer_train = tf.summary.create_file_writer(f'./logs/{real_experiment_name}/train')
        tb_log_writer_test = tf.summary.create_file_writer(f'./logs/{real_experiment_name}/val')

        S = None
        V = None
        if optimizer == "SGD_with_momentum" or optimizer == "SGD_with_EMA_momentum" or optimizer == "Adam":
            V = [tf.Variable(tf.zeros_like(w)) for w in model.trainable_variables]
        if optimizer == "RMS_Prop" or optimizer == "Adam":
            S = [tf.Variable(tf.zeros_like(w)) for w in model.trainable_variables]

        steps_per_epoch = train_sample_count // batch_size
        gradient_steps_count = tf.constant(0, tf.float32)
        # gradient_steps_count = 0
        for epoch in range(epochs):
            dataset_indices = np.arange(0, train_sample_count)
            np.random.shuffle(dataset_indices)
            epoch_x_train = x_train[dataset_indices]
            epoch_y_train = y_train[dataset_indices]

            epoch_loss = 0
            epoch_accuracy = 0
            for step in range(steps_per_epoch):
                batch_x_train = epoch_x_train[step * batch_size: (step + 1) * batch_size]
                batch_y_train = epoch_y_train[step * batch_size: (step + 1) * batch_size]

                loss, accuracy = gradient_step(model,
                                               optimizer,
                                               batch_x_train,
                                               batch_y_train,
                                               loss_function,
                                               learning_rate,
                                               V,
                                               S,
                                               gamma_V,
                                               gamma_S,
                                               gradient_steps_count
                                               )
                gradient_steps_count += 1
                epoch_loss += loss
                epoch_accuracy += keras.ops.mean(accuracy)

            epoch_loss = epoch_loss / steps_per_epoch
            epoch_accuracy = epoch_accuracy / steps_per_epoch
            tb_log_writer_train.set_as_default()
            tf.summary.scalar('loss', epoch_loss, epoch)
            tf.summary.scalar('accuracy', epoch_accuracy, epoch)
            tb_log_writer_train.flush()

            y_test_pred = model.predict(x_test, batch_size, verbose=None)
            test_loss = loss_function(y_test, y_test_pred)
            test_accuracy = keras.ops.mean(keras.metrics.categorical_accuracy(y_test, y_test_pred))

            tb_log_writer_test.set_as_default()
            tf.summary.scalar('loss', test_loss, epoch)
            tf.summary.scalar('accuracy', test_accuracy, epoch)
            tb_log_writer_test.flush()

            print(f'epoch : {epoch}, loss : {epoch_loss}, acc: {epoch_accuracy}, val_loss : {test_loss}, val_acc: {test_accuracy}')


def run_experiment_on_fashion_mnist_with_small_mlp():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_test = (x_test - np.mean(x_train)) / np.std(x_train)
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    batch_size = 2048
    seeds = [8, 42, 12, 3, 369]

    learning_rate = 0.32
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        create_small_mlp,
        my_categorical_cross_entropy,
        "SGD",
        f"09_fashion_mnist_small_mlp_SGD_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        batch_size=batch_size
    )
    learning_rate = 0.08
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        create_small_mlp,
        my_categorical_cross_entropy,
        "SGD_with_momentum",
        f"10_fashion_mnist_small_mlp_SGD_with_momentum_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        batch_size=batch_size,
        gamma_V=0.95
    )
    learning_rate = 0.64 * 2.0
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        create_small_mlp,
        my_categorical_cross_entropy,
        "SGD_with_EMA_momentum",
        f"11_fashion_mnist_small_mlp_SGD_with_EMA_momentum_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        batch_size=batch_size
    )
    learning_rate = 3e-3
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        create_small_mlp,
        my_categorical_cross_entropy,
        "RMS_Prop",
        f"13_fashion_mnist_small_mlp_RMS_Prop_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        batch_size=batch_size
    )
    learning_rate = 3e-3
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        create_small_mlp,
        my_categorical_cross_entropy,
        "Adam",
        f"14_fashion_mnist_small_mlp_Adam_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        batch_size=batch_size,
        gamma_S=0.999
    )



def run_experiment_on_fashion_mnist_with_deep_mlp():

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_test = (x_test - np.mean(x_train)) / np.std(x_train)
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    batch_size = 2048

    learning_rate = 0.32
    epochs = 15
    seeds = [8, 42]
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SGD",
    #     f"16_fashion_mnist_deep_mlp_SGD_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )
    # learning_rate = 0.08
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SGD_with_momentum",
    #     f"17_fashion_mnist_deep_mlp_SGD_with_momentum_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     gamma_V=0.95
    # )
    # learning_rate = 0.64 * 2.0
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SGD_with_EMA_momentum",
    #     f"18_fashion_mnist_deep_mlp_SGD_with_EMA_momentum_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )
    # learning_rate = 3e-3
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "RMS_Prop",
    #     f"19_fashion_mnist_deep_mlp_RMS_Prop_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )
    # learning_rate = 3e-3
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "Adam",
    #     f"20_fashion_mnist_deep_mlp_Adam_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     gamma_S=0.999
    # )
    learning_rate = 3e-3
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        create_deep_mlp,
        my_categorical_cross_entropy,
        "SignSGD",
        f"21_fashion_mnist_deep_mlp_SignSGD_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        gamma_S=0.999
    )



def run_experiment_on_fashion_mnist_with_small_conv_net():

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_test = (x_test - np.mean(x_train)) / np.std(x_train)
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    batch_size = 2048

    learning_rate = 0.32
    epochs = 100
    seeds = [8, 42, 12, 3, 369]
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SGD",
    #     f"16_fashion_mnist_deep_mlp_SGD_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )
    # learning_rate = 0.08
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SGD_with_momentum",
    #     f"17_fashion_mnist_deep_mlp_SGD_with_momentum_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     gamma_V=0.95
    # )
    learning_rate = 0.64 * 2.0
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        create_small_conv_net,
        my_categorical_cross_entropy,
        "SGD_with_EMA_momentum",
        f"24_fashion_mnist_small_conv_net_SGD_with_EMA_momentum_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    # learning_rate = 3e-3
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "RMS_Prop",
    #     f"19_fashion_mnist_deep_mlp_RMS_Prop_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )
    # learning_rate = 3e-3
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "Adam",
    #     f"20_fashion_mnist_deep_mlp_Adam_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     gamma_S=0.999
    # )
    # learning_rate = 3e-3
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SignSGD",
    #     f"21_fashion_mnist_deep_mlp_SignSGD_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     gamma_S=0.999
    # )



def run_experiment_on_fashion_mnist_with_really_small_mlp():

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_test = (x_test - np.mean(x_train)) / np.std(x_train)
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    batch_size = 2048

    learning_rate = 0.32
    epochs = 100
    seeds = [8, 42, 12, 3, 369]
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SGD",
    #     f"16_fashion_mnist_deep_mlp_SGD_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )
    # learning_rate = 0.08
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SGD_with_momentum",
    #     f"17_fashion_mnist_deep_mlp_SGD_with_momentum_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     gamma_V=0.95
    # )
    learning_rate = 0.64 * 2.0
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        create_really_small_mlp,
        my_categorical_cross_entropy,
        "SGD_with_EMA_momentum",
        f"28_fashion_mnist_really_small_mlp_SGD_with_EMA_momentum_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    # learning_rate = 3e-3
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "RMS_Prop",
    #     f"19_fashion_mnist_deep_mlp_RMS_Prop_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )
    # learning_rate = 3e-3
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "Adam",
    #     f"20_fashion_mnist_deep_mlp_Adam_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     gamma_S=0.999
    # )
    # learning_rate = 3e-3
    # run_experiment(
    #     ((x_train, y_train), (x_test, y_test)),
    #     seeds,
    #     create_deep_mlp,
    #     my_categorical_cross_entropy,
    #     "SignSGD",
    #     f"21_fashion_mnist_deep_mlp_SignSGD_bs_{batch_size}_lr_{learning_rate}",
    #     learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     gamma_S=0.999
    # )

def run_experiment_on_fashion_mnist_with_small_residual_dense_net():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_test = (x_test - np.mean(x_train)) / np.std(x_train)
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model_factory = lambda: create_small_residual_dense_net((28, 28, 1))

    batch_size = 2048

    epochs = 100
    seeds = [8, 42, 12, 3, 369]
    learning_rate = 0.32
    run_experiment(
        ((x_train, y_train), (x_test, y_test)),
        seeds,
        model_factory,
        my_categorical_cross_entropy,
        "SGD_with_EMA_momentum",
        f"30_fashion_mnist_small_residual_dense_net_SGD_with_EMA_momentum_bs_{batch_size}_lr_{learning_rate}",
        learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

def exp_tanh():
    for seed in [8, 42, 12, 3, 369]:
        keras.utils.set_random_seed(seed)

        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=keras.activations.tanh),
            keras.layers.Dense(16, activation=keras.activations.tanh),
            keras.layers.Dense(10, activation=keras.activations.tanh)
        ])

        x_test = (x_test - np.mean(x_train)) / np.std(x_train)
        x_train = (x_train - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10) * 2.0 - 1.0
        y_test = keras.utils.to_categorical(y_test, 10) * 2.0 - 1.0

        exp_name = f"01_tanh_and_mse_seed_{seed}"

        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.SGD(),
            metrics=[keras.metrics.categorical_accuracy]
        )

        model.fit(
            x_train, y_train, 128, 100, callbacks=[keras.callbacks.TensorBoard(log_dir=f"./logs/{exp_name}")]
        )

def exp_softmax_and_cat_cross_entropy():
    for seed in [8, 42, 12, 3, 369]:
        keras.utils.set_random_seed(seed)

        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=keras.activations.tanh),
            keras.layers.Dense(16, activation=keras.activations.tanh),
            keras.layers.Dense(10, activation=my_softmax)
        ])

        x_test = (x_test - np.mean(x_train)) / np.std(x_train)
        x_train = (x_train - np.mean(x_train)) / np.std(x_train)

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        exp_name = f"02_softmax_and_cat_crossentropy_{seed}"

        model.compile(
            loss=my_categorical_cross_entropy,
            optimizer=keras.optimizers.SGD(),
            metrics=[keras.metrics.categorical_accuracy]
        )

        model.fit(
            x_train, y_train, 128, 100, callbacks=[keras.callbacks.TensorBoard(log_dir=f"./logs/{exp_name}")]
        )

def main():
    # exp_tanh()
    # exp_softmax_and_cat_cross_entropy()
    # run_experiment_on_fashion_mnist_with_small_mlp()
    # run_experiment_on_fashion_mnist_with_deep_mlp()
    # run_experiment_on_fashion_mnist_with_small_conv_net()
    # run_experiment_on_fashion_mnist_with_really_small_mlp()
    run_experiment_on_fashion_mnist_with_small_residual_dense_net()


if __name__ == "__main__":
    main()