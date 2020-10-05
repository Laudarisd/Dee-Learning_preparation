Some comon problems and answers
=================================

1. What is Deep Learning?
2. What is Neural Network?
3. What is a Multi-layer Perception(MLP)?
4. What Is Data Normalization, and Why Do We Need It
5. What is the Boltzmann Machine
6. What Is the Role of Activation Functions in a Neural Network?
7. What Is the Cost Function?
8. What Is Gradient Descent?
9. What Do You Understand by Backpropagation?
10. What Is the Difference Between a Feedforward Neural Network and Recurrent Neural Network?
11. What Are the Applications of a Recurrent Neural Network (RNN)?
12. What Are the Softmax and ReLU Functions?
13. What Are Hyperparameters?
14. What Will Happen If the Learning Rate Is Set Too Low or Too High?
15. What Is Dropout and Batch Normalization?
16. What Is the Difference Between Batch Gradient Descent and Stochastic Gradient Descent?
17. What is Overfitting and Underfitting, and How to Combat Them?
18. How Are Weights Initialized in a Network?
19. What Are the Different Layers on CNN?
20. What is Pooling on CNN, and How Does It Work?
21. How Does an LSTM Network Work?
22. What Are Vanishing and Exploding Gradients?
23. What Is the Difference Between Epoch, Batch, and Iteration in Deep Learning?
24. Why is Tensorflow the Most Preferred Library in Deep Learning?
25. What Do You Mean by Tensor in Tensorflow?
26. What Are the Programming Elements in Tensorflow?
27. Explain a Computational Graph.
28. Explain Generative Adversarial Network.
29. What Is an Auto-encoder?
30. What Is Bagging and Boosting?
31. What is batch normalization?
32. How does YOLO work?
33. What is the difference between R-CNN and CNN?
34. Why do we need pretrain model?
35. Explain the architecture of pretrain model(Resnet, mobilenet, etc).
36. What are supervised and unsupervised learning algorithms in deep learning?
37. What’s the trade-off between bias and variance?
38. How is KNN different from k-means clustering?
39. Explain how a ROC curve works.
40. Define precision and recall.
41. Explain stride.
42. What is Bayes’ Theorem? How is it useful in a machine learning context?
43. Why is “Naive” Bayes naive?
44. Explain the difference between L1 and L2 regularization.
45. What’s your favorite algorithm, and can you explain it to me in less than a minute?
46. What’s the difference between Type I and Type II error?
47. What’s a Fourier transform?
48. What’s the difference between probability and likelihood?
49. What’s the difference between a generative and discriminative model?
50. What is cross-validation? What cross-validation technique would you use on a time series dataset?
51. How is a decision tree pruned?
52. Which is more important to you: model accuracy or model performance?
53. What’s the F1 score? How would you use it?
54. How would you handle an imbalanced dataset?
55. When should you use classification over regression?
56. Name an example where ensemble techniques might be useful.
57. How do you ensure you’re not overfitting with a model?
58. How would you evaluate a logistic regression model?
59. What’s the “kernel trick” and how is it useful?
60. Pick an algorithm. Write the pseudo-code for a parallel implementation.
61. What are some differences between a linked list and an array?
62. Which data visualization libraries do you use? What are your thoughts on the best data visualization tools?
63. Given two strings, A and B, of the same length n, find whether it is possible to cut both strings at a common point such that the first part of A and the second part  of B form a palindrome.
64. How are primary and foreign keys related in SQL?
65. How does XML and CSVs compare in terms of size?
66. What are the data types supported by JSON? 
67. How would you build a data pipeline?
68. How do you think Google is training data for self-driving cars?
69. How would you simulate the approach AlphaGo took to beat Lee Sedol at Go?
70. What are your thoughts on GPT-3 and OpenAI’s model?
71. What models do you train for fun, and what GPU/hardware do you use?
72. What are some of your favorite APIs to explore? 
73. Why is it necessary to introduce non-linearities in a neural network?

Solution: otherwise, we would have a composition of linear functions, which is also a linear function, giving a linear model. A linear model has a much smaller number of parameters, and is therefore limited in the complexity it can model.

74. Describe two ways of dealing with the vanishing gradient problem in a neural network.

*Solution:*

* Using ReLU activation instead of sigmoid.
* Using Xavier initialization.

75. What are some advantages in using a CNN (convolutional neural network) rather than a DNN (dense neural network) in an image classification task?

*Solution:* while both models can capture the relationship between close pixels, CNNs have the following properties:

* It is translation invariant — the exact location of the pixel is irrelevant for the filter.
* It is less likely to overfit — the typical number of parameters in a CNN is much smaller than that of a DNN.
* Gives us a better understanding of the model — we can look at the filters’ weights and visualize what the network “learned”.
* Hierarchical nature — learns patterns in by describing complex patterns using simpler ones.

76. Describe two ways to visualize features of a CNN in an image classification task.

*Solution:*

* Input occlusion — cover a part of the input image and see which part affect the classification the most. For instance, given a trained image classification model, give the images below as input. If, for instance, we see that the 3rd image is classified with 98% probability as a dog, while the 2nd image only with 65% accuracy, it means that

<table border="0">
   <tr>
      <td>
      <img src="./src/img/ob1.jpg" width="100%" />
      </td>
   </tr>
   </table>


* Activation Maximization — the idea is to create an artificial input image that maximize the target response (gradient ascent).

77. Is trying the following learning rates: 0.1,0.2,…,0.5 a good strategy to optimize the learning rate?

*Solution:* No, it is recommended to try a logarithmic scale to optimize the learning rate.

78. Suppose you have a NN with 3 layers and ReLU activations. What will happen if we initialize all the weights with the same value? what if we only had 1 layer (i.e linear/logistic regression?)

*Solution:* If we initialize all the weights to be the same we would not be able to break the symmetry; i.e, all gradients will be updated the same and the network will not be able to learn. In the 1-layers scenario, however, the cost function is convex (linear/sigmoid) and thus the weights will always converge to the optimal point, regardless of the initial value (convergence may be slower).

79. Explain the idea behind the Adam optimizer.

*Solution:* Adam, or adaptive momentum, combines two ideas to improve convergence: per-parameter updates which give faster convergence, and momentum which helps to avoid getting stuck in saddle point.

80. What is saddle point?


81. Compare batch, mini-batch and stochastic gradient descent.

*Solution:* batch refers to estimating the data by taking the entire data, mini-batch by sampling a few datapoints, and SGD refers to update the gradient one datapoint at each epoch. The tradeoff here is between how precise the calculation of the gradient is versus what size of batch we can keep in memory. Moreover, taking mini-batch rather than the entire batch has a regularizing effect by adding random noise at each epoch.

82. What is data augmentation? Give examples.

*Solution:* Data augmentation is a technique to increase the input data by performing manipulations on the original data. For instance in images, one can: rotate the image, reflect (flip) the image, add Gaussian blur






[click](https://www.simplilearn.com/tutorials/deep-learning-tutorial/deep-learning-interview-questions)

[click](https://www.springboard.com/blog/machine-learning-interview-questions/)

[click](https://towardsdatascience.com/50-deep-learning-interview-questions-part-1-2-8bbc8a00ec61)
