"""
architecture: n layers of relu, output layer of 1 relu neuron
to do: create functions:
model(hyerparams)
forward
backward
"""
import numpy as np


def sigmoid(Z):
    """ returns activation and cache containing Z"""
    return 1 / (1 + np.exp(-Z)), Z


def relu(Z):
    """ returns activation and cache containing Z"""
    cache = Z
    Z[Z < 0] = 0
    return Z, cache


def relu_backward(dA, Z):
    """ used as dZ = relu_backward(dA, activation_cache)"""
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return dA * Z


def sigmoid_backward(dA, Z):
    """ used as dZ = sigmoid_backward(dA, activation_cache)"""
    return np.exp(-Z) / (1 + np.exp(-Z)) ** 2 * dA


def initialize_parameters_deep(layer_dims, w_init):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * w_init
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
             cache consists of (A_prev, W, b) and Z
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = np.dot(W, A_prev) + b, (A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = np.dot(W, A_prev) + b, (A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, act, last_act):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        W, b = parameters['W' + str(l)], parameters['b' + str(l)]
        if act == 'sigmoid':
            A, cache = linear_activation_forward(A_prev, W, b, "sigmoid")
        elif act == 'relu':
            A, cache = linear_activation_forward(A_prev, W, b, "relu")

        caches.append(cache)

    W, b = parameters['W' + str(L)], parameters['b' + str(L)]
    if last_act == 'sigmoid':
        AL, cache = linear_activation_forward(A, W, b, "sigmoid")
    elif last_act == 'relu':
        AL, cache = linear_activation_forward(A, W, b, "relu")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y, cost_func, lambd, parameters):
    """
    Arguments:
    AL -- grade predictions, shape (1, number of examples)
    Y -- true grades, shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]

    cost = 0
    for i in range(m):
        y = Y[:,i]
        a = AL[:,i]
        if cost_func == 'cross_entropy':
            cost += - 1 / m * (np.dot(y.T, np.log(a)) + np.dot((1 - y).T, np.log(1 - a)))  # can I vectorize acrooss samples?
        elif cost_func == 'quadratic':
            cost += 1 / 2 / m * np.dot((y - a).T, (y - a))

    # add weight decay term
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(L):
        w = parameters["W" + str(l + 1)]
        cost += lambd / 2 / m * np.sum(w ** 2)

    return cost


def compute_n_correct(AL, Y):
    """
    Arguments:
    AL -- grade predictions, shape (1, number of examples)
    Y -- true grades, shape (1, number of examples)

    Returns:
    n_correct - number of correctly predicted grades
    """

    m = Y.shape[1]
    n_correct = 0
    error_sum = 0
    for i in range(m):
        y = Y[0,i]
        a = AL[0,i]
        predicted_grade = (a - 0.25) * 16 / 0.5
        correct_grade = int((y - 0.25) * 16 / 0.5)
        n_correct += int(round(predicted_grade) == correct_grade)
        error_sum += np.abs(predicted_grade - correct_grade)

    mae = error_sum / m
    return n_correct, mae


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, cost_func, act, last_act):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- grade prediction, output of the forward propagation (L_model_forward())
    Y -- true grade
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    if cost_func == 'cross_entropy':
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # for cross-entropy cost function
    elif cost_func == 'quadratic':
        dAL = AL - Y  # for quadratic cost function

    # Lth layer. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    if last_act == 'sigmoid':
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      "sigmoid")
    elif last_act == 'relu':
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                          current_cache,
                                                                                                          "relu")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        if act == 'sigmoid':
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                        "sigmoid")
        elif act == 'relu':
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, eta, lambd, m):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] * (1 - eta * lambd / m) - eta * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - eta * grads["db" + str(l + 1)]
    return parameters


def train_model(X, Y, X_test, Y_test, layers_dims, mini_batch_size, eta=0.1, n_epochs=30, cost_func = 'cross_entropy', act='sigmoid', last_act='sigmoid', w_init=1.0, lambd=0):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (n_data, number of examples)
    Y -- true grades, of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    eta -- learning rate of the gradient descent update rule
    n_epochs -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    parameters = initialize_parameters_deep(layers_dims, w_init)

    m = X.shape[1]

    # grad check
    X_temp = np.random.rand(X.shape[0], 1)
    Y_temp = np.random.rand(Y.shape[0], 1)
    AL, caches = L_model_forward(X_temp, parameters, act, last_act)
    grads = L_model_backward(AL, Y_temp, caches, cost_func, act, last_act)
    approx_grads = find_approx_gradients(parameters, X_temp, Y_temp, act, last_act, cost_func, lambd)

    flat_grads = np.array([])
    flat_grads_approx = np.array([])
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(L):
        dW = grads["dW" + str(l + 1)]
        db = grads["db" + str(l + 1)]
        flat_grads = np.append(flat_grads, dW.flatten())
        flat_grads = np.append(flat_grads, db.flatten())

        dW_appr = approx_grads["dW" + str(l + 1)]
        db_appr = approx_grads["db" + str(l + 1)]
        flat_grads_approx = np.append(flat_grads_approx, dW_appr.flatten())
        flat_grads_approx = np.append(flat_grads_approx, db_appr.flatten())

    err = flat_grads - flat_grads_approx
    result = np.linalg.norm(err) / (np.linalg.norm(flat_grads) + np.linalg.norm(flat_grads_approx))
    print('grad check result:', result)
    print(flat_grads[0:5], flat_grads_approx[0:5])


    # Loop (gradient descent)
    for i in range(0, n_epochs):

        permutation = np.random.permutation(m)
        X = X[:, permutation]
        Y = Y[:, permutation]
        n_minibatches = int(m / mini_batch_size)

        for i_mini in range(n_minibatches):
            X_mini = X[:, i_mini*mini_batch_size : (i_mini+1)*mini_batch_size]
            Y_mini = Y[:, i_mini * mini_batch_size: (i_mini + 1) * mini_batch_size]

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(X_mini, parameters, act, last_act)

            # Backward propagation.
            grads = L_model_backward(AL, Y_mini, caches, cost_func, act, last_act)

            # Update parameters.
            parameters = update_parameters(parameters, grads, eta, lambd, mini_batch_size)

        # Compute cost and estamate performance
        if i % 10 == 0:
            AL_full, caches_full = L_model_forward(X, parameters, act, last_act)  # compute for all samples (not mini-batch)
            cost = compute_cost(AL_full, Y, cost_func, lambd, parameters)
            n_correct_training, mae_training = compute_n_correct(AL_full, Y)
            training_result = str(n_correct_training) + ' / ' + str(m) + ' correct, mae = ' + "{:.2f}".format(mae_training)

            m_test = X_test.shape[1]
            AL_test, caches_test = L_model_forward(X_test, parameters, act, last_act)
            n_correct_test, mae_test = compute_n_correct(AL_test, Y_test)
            test_result = str(n_correct_test) + ' / ' + str(m_test) + ' correct, mae = ' + "{:.2f}".format(mae_test)

            print('epoch:', i, ', cost:', "{:.3f}".format(1000 * cost), 'training preformance:', training_result, 'test preformance:', test_result)

    return parameters



def find_approx_gradients(parameters, X, Y, act, last_act, cost_func, lambd):
    eps = 1e-7


    grads_approx = {}
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(L):
        W = parameters["W" + str(l + 1)]
        grads_approx["dW" + str(l + 1)] = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_new = np.copy(W)

                W_new[i,j] -= eps
                params_new = parameters.copy()
                params_new["W" + str(l + 1)] = W_new
                AL_full, caches_full = L_model_forward(X, params_new, act, last_act)
                cost_minus = compute_cost(AL_full, Y, cost_func, lambd, parameters)

                W_new[i, j] += 2 * eps
                params_new = parameters.copy()
                params_new["W" + str(l + 1)] = W_new
                AL_full, caches_full = L_model_forward(X, params_new, act, last_act)
                cost_plus = compute_cost(AL_full, Y, cost_func, lambd, parameters)

                grads_approx["dW" + str(l + 1)][i,j] = (cost_plus - cost_minus) / 2 / eps

        B = parameters["b" + str(l + 1)]
        grads_approx["db" + str(l + 1)] = np.zeros_like(B)
        for i in range(B.shape[0]):
            B_new = np.copy(B)

            B_new[i] -= eps
            params_new = parameters.copy()
            params_new["b" + str(l + 1)] = B_new
            AL_full, caches_full = L_model_forward(X, params_new, act, last_act)
            cost_minus = compute_cost(AL_full, Y, cost_func, lambd, parameters)

            B_new[i] += 2 * eps
            params_new = parameters.copy()
            params_new["b" + str(l + 1)] = B_new
            AL_full, caches_full = L_model_forward(X, params_new, act, last_act)
            cost_plus = compute_cost(AL_full, Y, cost_func, lambd, parameters)

            grads_approx["db" + str(l + 1)][i,0] = (cost_plus - cost_minus) / 2 / eps

    return grads_approx


