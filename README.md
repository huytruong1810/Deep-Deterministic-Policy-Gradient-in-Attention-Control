# Deep Deterministic Policy Gradient in Attention Control
## Author: Truong Nguyen Huy

The rising of geometric deep learning has brought about a structured framework for developing neural network architectures dedicated to contextualizing geometric invariance and equivariance in many domains, such as graphs or string representations of data. By feature-engineering data under a symmetry group of interest, we allow neural models to pay special attention to the extraordinary properties of these landscapes. I want to further elaborate on the idea of attention in Euclidian spaces by designing a convolution-class network that can recurrently pay attention to critical regions in the input space. The Recurrent Attention Convolutional Network (RACN) is to be trained using Deterministic Policy Gradient to complete this task. As an overview, at any given time step t, RACN positions a filter F_t used to build feature maps μ_t≔{μ_(t-1),x_t} with x_t  being the inner product between F_t and a patch on 2D images, a recurrent hidden state h_t=f(μ_t,h_(t-1) ), and a multi-head output that verifies F_(t+1), the prediction P_t, termination action T_t, the distance D_(t+1) and the angle θ_(t+1) that F is to be pushed to in the next time step. The whole architecture is parametrized by ω and is designed to build feature maps that regressively get better by traversing F across 2D images until self-termination determined by the output T. Suppose μ_(T-1) is the feature map at the final time step, then P_(T-1) is the conclusive label prediction, that is, predictions at preceding time steps are only useful for behavior analysis. To contextualize the task that RACN must complete, we define a high-level description for an MDP as follows:

1. The state space is a space of 2D images, their ground true labels Y, the label predictions P, and the positions of F. Initially, F_0 is at the center of any given image I. In this project, the MNIST dataset for digit classification is the subject of experimentation.
2. The action space consists of the binary termination action T_t, the continuous actions of selecting distance D_(t+1) and the angle θ_(t+1) that F is pushed to.
3. The reward signal is computed at the end of each I classification session. It considers the number of time steps until self-termination and the matching of the label prediction P_(T-1)^ω and ground truth Y.
4. The transition dynamics is deterministic. F_t is at the location that is determined by the distance D_(t-1) and the angle θ_(t-1).

## References:
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention - https://arxiv.org/abs/1502.03044
- Reinforcement Learning with Attention that Works: A Self-Supervised Approach - https://arxiv.org/abs/1904.03367
- Where to Look: A Unified Attention Model for Visual Recognition with Reinforcement Learning - https://arxiv.org/abs/2111.07169


