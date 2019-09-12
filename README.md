# LYRICS


(**L**earning **Y**ourself) (**R**easoning and **I**nference) with **C**onstraint**S**

*DISCLAIMER: A new version of the theory and the correspondent framework can be found at: https://github.com/GiuseppeMarra/tf-logic*

*DISCLAIMER2: This is a work-in-progress repository. Code needs to be commented and we aim at providing you with a Jupiter notebook for each of them. Moreover, some advanced features of this framework are not used in any example and some examples are in order for them.*

*DISCLAIMER3: This code has been tested with Python 2.7 and Tensorflow 1.4.*

## Introduction
This framework is aimed at facing both learning and inference tasks by requiring the satisfaction of a set of constraints on the domain of discourse. 

In practice, LYRICS is a TensorFlow environment where you can define any many–sorted logical theory, namely you can declare some domains of different sort, with constants, functions and relations on them.
For example:

```python
    lyr.Domain(label="Points", data=some_tensor)
    lyr.Relation(label="A", domains=("Points"), function=isA())
```

For each of such functions and relations, you can attach to it an opportune purpose-built function if it is already known, or you can learn it by an optimization program. In this case, you have to specify the general architecture of the function you are going to learn, e.g. an MLP, CNN, RNN and so on.
For example:

```python
    isA = lyr.functions.FFNClassifier(name="isA", input_size = 2, n_classes = 1, hidden_sizes = [10,5])
```

Once you defined the objects in the problem, you can write down a set of logical constraints expressing the knowledge you have about the task. Supervisions has to be thought of as a special case of constraints. In this setting you are able to manage both partially labeled data (semi-supervised learning) and totally unsupervised data by means of learning from logical constraints.

```python
    lyr.Constraint("forall p: forall q: areClose(p,q) -> (A(p)<->A(q))")
```


## Content of the repository

We provided a set of fully runnable simple examples; in particular:
1. `manifold.py` is an example on how to implement manifold regularization by means of logical constraints;
2. `missing.py` is an example on how to find missing features of individuals by a logic description of them;
3. `reasoning.py` is an example on how to implement classical logical reasoning;
4. `deduction.py` is a naive example on how infer logical rules from data by model checking;
5. `collective.py` is an example on how to implement collective classification using our logical framework.
