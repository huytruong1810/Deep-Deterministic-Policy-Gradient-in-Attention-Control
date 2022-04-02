# Deep Deterministic Policy Gradient in Attention Control
## Author: Truong Nguyen Huy

## Background:
The rising of geometric deep learning has brought about a structured framework for developing neural network architectures dedicated to contextualizing geometric invariance and equivariance in many domains, such as graphs or string representations of data. By feature-engineering data under a symmetry group of interest, we allow neural models to pay special attention to the extraordinary properties of these landscapes. I want to further elaborate on the idea of attention in Euclidian spaces by designing a convolution-class network that pays attention to critical regions in the input space.

## Project Description:
The Recurrent Attention Convolutional Network (RACN) is to be trained using the Deep Deterministic Policy Gradient method (DDPG) to complete this task. The task of the RACN agent can be described as follows:

• The agent will be operating in a partially observable environment that is the MNIST handwritten-digit dataset. At the beginning of each episode, a single image is randomly sampled from the dataset and is used to train/test the agent's capability of adjusting attention and digit recognition. This methodology follows the Monte-Carlo sampling scheme. That is, over a large number of episodes, the agent is able to experience the whole environment in a sample-efficient manner.

• At any given time step, the agent positions an identity filter, by sampling continuous values of angle and distance, over an image patch and computes the dot product to build a feature element (FE) and add it to a regressive feature map. Over the duration of the episode, this regressive feature map is used by the agent's digit classification head to predict the digit's label. In detail, the continuous action head outputs a Radians angle value and a Euclidian distance value that can be used to compute the 2D Euclidian location to move the filter to. This head is trained using the DDPG Actor-Critic method. The digit classification head outputs the probability distribution spanned over different digit possibilities. This head is optimized to minimize the Cross-Entropy Loss (CEL) between prediction and the true label of the episode's image. The whole architecture uses the regressive feature map and the immediate previous actions as inputs at every time step. Suppose we have the feature map at the final time step, then the equivalent prediction is the conclusive label prediction of the episode's image. That is, predictions at preceding time steps are only useful for the agent's behavioral analysis.

We define a description for this Partially Observable Markov Decision Process (POMDP) as follows:

• The state space is the space of 2D MNIST grayscaled images with their ground-true labels. The observation space is an image, its label, the constructed feature map, and the filter's Euclidian position. Initially, the feature map is a zero tensor and the filter's position is at the center of the image.

• The action space consists of the continuous actions of selecting a Euclidian distance and a Randians angle. Therefore, we bound the distance to be between 0 and a finite upper bound; and the angle to be between negative pi and positive pi. We define the random process used in exploration as the scaling of random non-negative values and Radians values using a decaying epsilon rate (which is a value between 0 and 1) and applying them to the action outputs.

• The reward function considers the valid choice of filter's positioning, computed FE's valuableness, and the CEL of the agent's conclusive digit prediction. FE's valuableness is defined as the qualities of being different from previously computed FEs and being non-zero. The first quality demonstrates the diversity within the feature map and the second demonstrates its informativeness. This is, of course, used to guide the positioning of the filter to achieve the greatest MNIST digit classification accuracy and is specific to this task alone. This reward function need not generalize to the general object recognition in images.

• The transition dynamic is deterministic. That is, the values of distance and angle are used to compute the location of the filter and move it accordingly. If the position is invalid, the filter stays still. An FE and a prediction are outputted by the agent at every filter's location to add to the regressive feature map. The time horizon of the episode is the feature map's length, which is a hyperparameter.

## Preferences:
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention - https://arxiv.org/abs/1502.03044

- Reinforcement Learning with Attention that Works: A Self-Supervised Approach - https://arxiv.org/abs/1904.03367

- Where to Look: A Unified Attention Model for Visual Recognition with Reinforcement Learning - https://arxiv.org/abs/2111.07169


