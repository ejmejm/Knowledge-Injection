## Weight importances

Weight importances indicate how much a change in a given weight will affect the model's output for a given sample or set of samples.
They are calculated as the absolute value of the gradient of the l2-norm of the model's output with respect to the weight.
If multiple samples are provided, the weight importances are averaged over the samples.

## List of experiments

1. Capability & Knowledge Localization

#### Motivation

One of the core goals of this project is to be able to imbue a language model (or any other model) with new knowledge even after the model has been instruction fine-tuned or trained with RLHF.
This is a challenge because most of the knowledge in LLMs comes from the pre-training phase. After the model has been fine-tuned, however, returning to the pre-training phase will regress the model's performance.
Before we attempt to construct a solution, we first need to better understand how weights for different capabilities are distributed throughout the network.
If, for example, weights for storing facts and weights for more abstract capabilities like question answering were separately clustered in different parts of the network, that could make the injection of new knowledge easier.

#### Experiment

The experiment requires preparing 2 sets of prompts for a pre-trained language model.
In each set there is a one to one mapping between the prompts, but the prompts in the first set are questions, and the prompts in the second set are statements that *indirectly* include the information to answer the questions.
When we calculate the weight importances averaged over both sets, the differences should indicate which weights are important for which type of information/capabilities.
Once we have both sets of importances and their differences, we need to visualize them in a way that is easy to compare.