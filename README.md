# CLARE


The language we present in this paper has to be thought of as a uniform framework to face both learning and inference tasks by requiring the satisfaction of a set of constraints on the domain of discourse. 

In practice, CLARE is a Tensor- Flow environment where you can define any many–sorted logical theory, namely you can declare some domains of different sort, with constants, functions and relations on them. For each of such functions and relations, you can attach to it an opportune purpose-built function if it is already known, or you can learn it by an optimization program. In this case, you have to specify the general architecture of the function you are going to learn, e.g. an MLP, CNN, RNN and so on. Once you defined the objects in the problem, you can write down a set of logical constraints expressing the knowledge you have about the task. As a special case, any supervision has to be thought of as an atomic constraint or its negation. However, in this setting you are able to manage both partially labeled data (semi-supervised learning) and totally unsupervised data (pure inference) by means of learning from logical constraints.
