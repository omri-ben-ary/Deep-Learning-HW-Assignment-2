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
1. These are two different algorithms that performs different tasks to achieve a common goal. Gradient descent is an iterative optimization algorithm. The algorithm method is to calculate to gradient of the function we want to minimize and to take a small step in the opposite direction as the gradient points to the local steepest ascent. Back propogation is an algorithm that helps us implement gradient descent efficiently. By applying the chain rule on the calculations and storing relavent results from forward pass we can efficiently calculate all the desired partial derivatives efficiently (partial derivatives compose the gradient).

2. Gradient descent takes into account all samples in dataset to calculate the accurate gradient while stochastic gradient descent takes into account only a fixed amount of samples. In SGD, we can take one sample at a time which will be highly inaccurate and very inconsistent. Alternatively, we can use a mini-batch of samples, this way we get a better proxy of the gradient and thus better results.

3. Firstly, it is highly inefficient to calculate the gradient w.r.t the entire dataset, as GD does, SGD provides a very good tradeoff between compute runtime and accuracy. Secondly, SGD adds noise to the iterative method and so it helps us "escape" local minimas and give us a better opportunity to reach global minimas. Finaly, while SGD convergence may be empiricaly noisier it can achieve satisfactory results quicker than GD (quicker == less epochs).

4. A. Yes, it is equivelent. If we denote the loss as L, in GD we compute $\nabla_{\theta} L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} L_i(\theta)$. Using the method above we simply split the sum to a few disjoint sums so when calculating the derivatives we get the same results due to linearity, therfore: $\sum_{m=1}^{M} \sum_{j=1}^{b} \nabla_{\theta} L_{m,j}(\theta) = \sum_{i=1}^{N} \nabla_{\theta} L_{i}(\theta)$

B. Recall that when we implemented the layers we stored the relavent information in the forward pass to use for the backward pass. This means that we will have to do the same for the entire dataset even if it is split into batches, we will have to store information about the entire dataset which will eventually lead to an out of memory error.

"""

part2_q4 = r"""
1.

A.
In forward mode AD, we compute the derivative of each function $f_i $ with respect to its input while evaluating the function itself. If we were to store only the necessary derivative values at each step, we avoid the need to store all intermediate values. This results in a memory complexity of ${O}(n)$.

B.
In backward mode AD, the computation proceeds by first evaluating the function from input to output, storing the necessary intermediate values, and then computing the derivatives from output to input. To reduce memory complexity, we can use a technique called checkpointing. Instead of storing all intermediate values, we store only a subset of these values at strategic points (checkpoints) during the forward pass. During the backward pass, when an intermediate value is needed but not stored, we recompute it from the nearest checkpoint. This reduces the memory complexity to ${O}(n)$, as we only store a limited number of intermediate values and recompute others as necessary, balancing memory use and computation cost.

2.

Yes, these techniques can be generalized for arbitrary computational graphs. In both forward and backward mode AD, we can store only the necessary derivative values at each step. Checkpointing can be used in backward mode AD to further reduce memory complexity. These methods ensure efficient memory usage for gradient computation in arbitrary computational graphs.

3.

When applied to deep architectures like VGGs or ResNets, these memory reduction techniques significantly benefit the backpropagation algorithm. By reducing memory complexity through forward or backward mode AD and employing checkpointing strategies, we can manage the large number of layers in deep networks more efficiently. This enables the training and inference processes to handle deep architectures with limited memory resources, thus preventing excessive memory consumption.

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

**High optimization error** occurs when the training algorithm can't find the best parameters to minimize the training loss. This might happen due to a poor choice of optimization algorithm, insufficient training time, or poorly chosen hyperparameters like learning rate and batch size. To reduce optimization error, you can use better optimization algorithms such as Adam or RMSprop, try different hyperparameters, increase training epochs, and use regularization techniques like dropout or batch normalization. In terms of population loss,
the population loss is closely related to generalization as well as training loss, when generalization is good we can say that optimization error is a good proxy to the population loss. In addition, when the model is underfitted we will experience high optimization error and high population loss due to high bias. If the model is overfitted we will see low optimization error but high population loss due to high variance. In terms of receptive field, if the receptive grows slowly it will be hard for the model to learn broad contextual features and thus optimization error will increase. Optimally we want receptive field to grow just at the right rate capturing local features as well as broad contextual features that will lead to lower optimization error.

**High generalization error** is the difference between training error and test error, showing how well the model performs on new, unseen data. It can be caused by overfitting (model is too complex) or underfitting (model is too simple). To reduce generalization error, use regularization methods, data augmentation, cross-validation, increase training data, apply early stopping, and use ensemble methods to combine multiple models and improve performance. In terms of population loss, the generalization loss is one component of the population loss. Optimally we want low generalization loss and therefore a good generalization, the model's ability to achieve high accuracy on unseen data, is necessary to improve population loss. In terms of receptive field, we want the receptive field to grow along the layers of the network gradually so it will be able to capture local features and global features. It is important for the model to learn both as they both give relavent information for the model to  learn and preform well on it's task. A model with a receptive field growing to raipidly will miss important local features, overfit and increase the importance of global features to an undesired degree. A model with a receptive field growing vey slowly might underfit as it will not learn any global features relavent to the task. A model with the exact right receptive field design will be able to balance between global and local features in way that will generalize optimally.

**High approximation error** happens when the model can't accurately represent the target function because it's too simple or lacks capacity. To reduce approximation error, use more complex models or deeper neural networks, increase the receptive field in convolutional neural networks, improve input features through feature engineering, design advanced architectures suited for the problem, and use cross-validation to ensure the model fits the data's complexity. By addressing these errors, you can improve your model's performance and reliability. In terms of receptive field we would like to increase it so the model is able to learn more global features that are usally harder to find and require high receptive fields, thus increasing model class capacity. In terms of population loss, high approximation error tell us that population loss will also be high as the model class capicity is not rich enough to learn from the data.
"""

part3_q2 = r"""

In a binary classifier, the false positive rate (FPR) is higher when the cost of missing a positive instance is very high, such as in fraud detection systems for financial transactions. Here, the classifier may flag many legitimate transactions as suspicious to avoid missing actual fraudulent activities, leading to more false positives. Conversely, the false negative rate (FNR) is higher when the cost of a false positive is significant, like in software deployment for cybersecurity. In this case, the classifier might avoid flagging legitimate software updates as malicious to ensure critical updates are not blocked, resulting in a higher FNR as some threats may go undetected.
"""

part3_q3 = r"""
1.

You might still opt for a higher threshold on the ROC curve to balance the trade-off between false positives and false negatives. Since the symptoms will eventually confirm the diagnosis, it’s acceptable to have some false positives leading to unnecessary tests, as patients will ultimately receive treatment once symptoms arise. This allows for broader screening without significant harm.

2.
Here the approach changes significantly. You would likely choose a lower threshold on the ROC curve to minimize false negatives, even at the expense of increasing false positives. The priority is to ensure that as many individuals as possible with the disease are identified early, despite the higher costs and risks associated with further testing. Early detection is crucial to prevent life-threatening outcomes, making it essential to screen aggressively, even if it results in more follow-up tests for healthy individuals.
"""


part3_q4 = r"""
MLP may not be the best choice for training on sequential data, such as classifying the sentiment of a sentence where each data point is a word, As MLPs treat inputs as independent features, lacking the ability to capture temporal dependencies or sequential context. In sentiment analysis, the meaning of a word often depends on its surrounding words, making it essential to understand the order and relationships between words in a sentence. MLPs do not posses the ability to model these dependencies. If the task we want to achieve is training on sequential data and classifying the sentiment of a sentence, we are better off using architectures like RNN or Transformers.
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

1.

Bottleneck block reduces computation by projecting to a lower dimension of feature representation, convolves, and projects output back to original dimension. Therefore we get:

The convolution **without bottleneck** has $64 \cdot (256 \cdot 3 \cdot 3 + 1) + 256 \cdot (64 \cdot 3 \cdot 3 + 1) = 295,232$ parameters.

The convolution **with bottleneck** has $(256+1)\cdot64 + 64\cdot(3\cdot3\cdot64 + 1) + (64 + 1)\cdot256 = 70,016$ parameters.

In general for each layer we have filter size (e.g 3 by 3) times depth of tensor (e.g 256) so we get $256 \cdot 3 \cdot 3$ add one parameter for bias and multiply by number of filters (e.g 64) so in total: $64 \cdot (256 \cdot 3 \cdot 3 + 1)$ or $filterNumber \cdot (height \cdot width \cdot depth + 1)$. 

We can see that using bottleneck is **reducing** a lot of parameters.

2.

In general, each dot product in the convolution consists of $filterHeight \cdot filterWidth \cdot \cdot depth \cdot filterNumber $ floating points operations. We want to this for every pixel so we multuply this by Image width times Image height totaling to: $(filterHeight \cdot filterWidth \cdot \cdot depth \cdot filterNumber) \cdot (imageHeight \cdot imageWidth) $ floating poing operations.

For the bottleneck block, the computation is $((1 \cdot 1 \cdot 256 \cdot 64) + (3 \cdot 3 \cdot 64 \cdot 64) + (1 \cdot 1 \cdot 256 \cdot 64)) \cdot (H \cdot W) = (16,384 + 36,864 + 16,384) \cdot (H \cdot W) = 69,632 \cdot (H \cdot W)$ floating poing operations.

For the regular block, the computation is $((3 \cdot 3 \cdot 256 \cdot 64) + (3 \cdot 3 \cdot 64 \cdot 256)) \cdot(W \cdot H) = (147,456 + 147,456) \cdot (W \cdot H) = 294,912 \cdot (W \cdot H)$ floating poing operations.

Again, it can be seen that bottleneck block **reduces** number of floating point opertions.

3. 

Spatial

   In the regular block we have two convolution layers of 3x3 therefore the receptive field is 5x5 (assuming stride=1).
   In the bottleneck block we have two convolution layers of 1x1 and one convolution layer of 3x3 therefore the
   receptive field is 3x3. The spatial input combination using the bottleneck block is worse because it's receptive is smaller
   while for the regular block the receptive big is bigger. Bigger receptive field means that each pixel in the output has information about more
   pixels in the input image, by this sense regular block is better.
   
   
Across feature map

   In the bottleneck block we firstly do a dimension reduction, convolving with a 1x1 filter, which allows us to combine diffrent feature maps in
   various ways using weighted sum. The second convolution, 3x3, is more for spatial combination and does not combine across the feature map. Lastly,
   the 1x1 convolution at the end alows to combine again between the different features. In total, bottleneck blocks allow for a rich combintaion across feature map.

   As for the regular block, in this case we only do 3x3 convolutions with no dimension reduction so this meaans that the block does not allow combination of
   different features as the filter computes each channel separately and does not combine between different channels. In total, regular blocks don't allow
   for combination across the feature map.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

1.

Firstly we notice that for both K=32 and K=64 we got similar results, so our discussion is relavent to both. Secondly, it is very clear that L=16 did not learn at all, and since the next quesion discusses exactly this issue, for this discussion we will disregard the L=64 graphs. When we analyze the train accuracy it is clear that the deeper the network the better the accuracy, but not too deep as the L=16 is too deep. This makes senese because deeper networks increase model class capcity allowing for more complex data patterns to be learned by the model and more abstract high level features that improve the model's performence. As for the test accuracy results, we see that all models did quite the same with no major diffrences. This tells us, mainly for the deeper networks, that the models did not generalize well as we should expect to see better results on the deeper networks like we saw in train accuracy. To improve generalization we can use regularization methods more effectivly like dropout or weight decay. Putting all of these together, The model with L=8 produces the best results.

2.

In both cases (K=32 and K=64) we observe that the L=16 did not learn at all and was not trainable. This happens for multiple reasons. Firstly, in very deep networks we can come across vanishing/exploding gradients during training which effectively stops the learning. When gradients are too small, we stay in the same place in the hyperparameter space because the step size is almost zero due to gradient size. When gradients are too big, we take too big of a step which could cause us to move up along the lost function instead of downwards. Another reason that could explain why we were not able to train the network is the fact that the number of hyperparameters are much bigger in deeper networks which means that optimization process on a much bigger hyperparameter space (dimension wisw) is much harder and extremly slower. To solve these problems we can take a few measures. As for exploding gradients we can use gradient clipping which limits the gradients size. For vanishing gradients we can use different activation functions like leaky ReLU that has non zero gradients. Also, we can use residual blocks that act as shortcuts in the network that allow us to skip layers and decrease this phenomenon. As for the harder optimzation problem we can use more sophisticated optimization algorithms like Adam or RMSprop which may do better. Also allowing for a longer optimzation time might help by tuning the early stopping parameter/
"""

part5_q2 = r"""
**Your answer:**
????????????????????????
We notice that K has no large effects on the results. We notice 
"""

part5_q3 = r"""
**Your answer:**

All models got high accuracy both in test and train. Having said that, it is apparent that all models overfitted a bit as the train loss monotonously decreases while the test loss starts to increase at a certain point. This makes sense because these networks are not so deep so it is easy to overfit. In addition, the model class capicity is much smaller and so approximation error could be high making generalization worse, this explains the diffrent trends between train loss and test loss towards the end.
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
1.
The model detection rate was not high in those particular images. Although there were some detections, they were mostly of the wrong label. This is probrably due to overlapping objects that alter the spatiality of the image, making the model mistakenly predict object. We can also notice that even when the object were detects, which is not always the case as we can see in the second image (the cat in the middle is not detected as an object at all), the detection is for the wrong label. Another factor to show us that the model did not succeed so well is the confidence rate, aside from one with a confidence of 90% (on a mislabled dolphin), all other were between 35% and 65% which is considerably low.

2.
A main reason for the model to fail is as we said due to the overlapping in the images. It alters the spatialty of the images and causes them to differ greatly from images the model has seen and has been trained on before. 
Lets take a better look at the first image detection and infer reasons of failure from it. Firstly, dolphins are rarely out of the water, this angle of image with the sun in the back may be similar to a large amount of images it has been trained on that aren't dolphins. We can assume that the model would have been able to detect the dolphins if they were in the water, as this is a more common image to train on. Secondly, we see that the detections were of a human and a surfboard, which are much more common with the sky and waves and this is probrably the reason it detected the dolphins as persons and a surfboard.
We can suggest a few things in order to better the detection rate in our opinion. We can train the model on more of these images, the more we add similar images the dataset the more it is likely to be detected correctly. Another option is to manipulate the image and "cut" it to fit less of the environment and more of just the object, making it focus more on the object itself.

3.
To attack a YOLO object detection model using Projected Gradient Descent (PGD), you start with the original image and iteratively adjust it to increase the model's loss, causing it to make incorrect predictions. This involves computing the loss gradient and updating the image within a given limit. After several iterations, you evaluate the adversarial image to check if it causes the model to misdetect or misclassify, revealing the model's vulnerabilities and helping to develop robust defenses.

"""


part6_q2 = r"""
*Your answer:*


Write your answer using *markdown* and $\LaTeX$:
python
# A code block
a = 2

An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""

We chose to demostrate Occlusion, Model Bias and Textured Background and Blurring pit falls.

*Occlusion:*

The picture we chose is a picture of a pack of dogs sitting next to and behind each other. We can clearly see the effects of occlusion here, the dogs in the front are detected with a good enough confidence, whilst the dogs in the back are either wrongfully detected or not detected at all. We can also see that the bounding box of those in the front is much more precise meaning it was able to detect those dogs that weren't occluded better.

*Model bias:*

The picture we chose is a picture of a humanoid robot called sophia.
The robot clearly has some non human aspects: the bald head with metal at its top and the robotic arms and chest.
We can guss that the model detects the robot as a person as this is a very humanoid robot so there are a lot of features that are similar to human.Additionaly, we assume that another reason is that there aren't many images of robots in the dataset, at least not as much as persons that causes the bias.

*Textured Background and Blurring:*

The picture we chose shows the effect of blurring very well, object the were static during the capture of the image, like the traffic light, are not blurry and are very well detected. Whereas object that were moving are detected poorly or not detected at all. We can assume that the persons that were moving slightly during the capture are the ones that eventually got detected and the ones that moved a lot are those who weren't detected at all.

"""

part6_bonus = r"""
The image we chose to perform the enhancment on is a picture with an extermly low contrast of a person. 
Firstly, we displayed the original image, it it very dificult even for a human eye to detect that this is a person, and the model stuglled to detect an object at all, let alone a correct detection of a human.
Secondly, We tried to enhance the contrast of the image using tools learned in EE's course - Images Proccesing and Analysis (046200) and used an image sharpening method using histogram equalization, adding gaussian blur and finally using a sharpening kernal. We can see that this already increased the contrast massively, yet the model still hasn't detected any objects in the image.
Finally, we used a method called Contrast Limited Adaptive Histogram Equalization (CLAHE), as well as a more aggressive sharpenning kernel in order to achieve better overall contrast.
We can see that after performing this method we indeed got a detection, and a detection of a person as it should be.

"""