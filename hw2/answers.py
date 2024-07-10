r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1.

A. The tensor shape is in correlation with the input, output and batch size. We want to calculate for each output, the amount of input derivetives, and of course, take the batch size into account. In total we will get [batch_size, out_features, in_features] =  [64, 512 , 1024]

B. No, the Jacobian matrix is not necessarily sparse as this is a fully connected layer meaning that every input is not trivialy 0. Each ouput is a function of every input and for it to be sparse most input element need to be zero and this is not the case for every input.

C. In order to calculate the downstream gradient w.r.t $\delta\mat{X}$ we do not need to explicitly matrialize the Jacobian matrix. Instead, we can use the following calculation: $\delta\mat{X} =\pderiv{L}{\mat{Y}}W^T$ which is much easier to calculate. We can argue that $W^T$ is the Jacobian matrix but we don't actually need to derive it, as it is given to us explicitly by the model paramters.

2.

A. The tensor shape is in correlation with the weight matrix shape and the output size. Using the same logic above, apart from the batch size which is irrelavent here because W shape is not dependant on the batch size. We get $[512, 512\cdot1024]$.

B. Yes, the Jacobian matrix is sparse. Each Y element is depandent on the corresponding column in W matrix. For example, the first Y element is depandent on the first row because when we do the matrix multiplication $\mat{W}\cdot \mat{X},$ $Y_1$ is multiplied only by $W_{11}$ in the first row, for the second row $Y_1$ is multiplied by $W_{21}$ only etc. Therfore, the rest of the elements will be zero ($W_{ij}, j\ne1$) for the first Y element.

C. As mentioned in question 1.c , we do not need to calculate materialize the Jacobian. The same reasons are apllied in this section, as we only need to calculate $\delta\mat{X} =\pderiv{L}{\mat{Y}}W^T$.

"""

part1_q2 = r"""
**Your answer:**

Yes.
While using back-propagation is todays standard in training neural networks, it is not required in order to just train the network. However, if it is our goal to train the network with decent-based optimization, it is **required**.
If we weren't to use back-propagation, we would have to do a forward pass in order to calculate every partial derivative, which is highly inefficient.


"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 1.5, 0.07, 0.02
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======

    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr_vanilla = 0.045
    lr_momentum = 0.003
    lr_rmsprop = 0.0002
    reg = 0.005
    #reg = 0.006 this is a test
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Yes they do match our expectations. We can see that when we use 0.4 dropout rate, we "force" the model to use all it's weights to make a prediction. When we don't use dropout, the model can rely heavily on certain weights that can lead to biases in prediction. In some sense, dropout acts as a regulrization mechanism to help the model generalize, this is why accuracy is better when using 0.4 dropour rate.

2. We notice that when the dropout rate is too big (like the case of 0.8) the model practicaly does not learn as we drop most weights. It can be seen in the graph that accuracy is very low accordingly.

"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible. This can happen because accuracy measures only the correctness of predictions. Cross-entropy loss is a meausre of uncertainty, this is not necessarily with correlation to the correctness of predictions. If the model becomes less confident in its correct predictions (meaning that uncertainty is higher), the loss can increase even if accuracy improves.

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.02
    weight_decay = 0.005
    momentum = 0.6
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""