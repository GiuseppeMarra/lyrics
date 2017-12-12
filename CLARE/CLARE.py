import tensorflow as tf
import utils
import functions
from functions import *
from fuzzy import Lukasiewicz
class World(object):

    def __init__(self):

        #Map from domain name to domain object
        self.domains = {}

        #Map from function name to function object
        self.functions = {}

        #Map from relation name to relation object
        self.relations = {}

        #Map from individual id to individual object
        self.individuals = {}

        self.logic = Lukasiewicz()

        self.loss = tf.constant(0.)


current_world = World()
from parser import Variable, Constant, FOLParser



class Domain(object):
    def __init__(self, label, data=None, dom_type="float"):
        if label in current_world.domains:
            raise Exception("Domain %s already exists" % label)
        self.dom_type = dom_type
        self.label = label
        self.tensor = tf.convert_to_tensor(data, dtype=tf.float32)

        current_world.domains[self.label] = self

        #Map from individual id to the corresponding row inside the domain
        self.individuals = {} #TODO

    def add_individual(self, individual):
        self.tensor = tf.concat((self.tensor, individual.tensor), axis=0)
        self.individuals[individual.label]=self


class Individual(object):
    def __init__(self, label, domain, value=None):
        if label in current_world.individuals:
            raise Exception("Duplicate individual id %s " % label)
        self.label = label
        if isinstance(domain, str):
            self.domain = current_world.domains[domain]
        if value is not None:
            self.tensor = tf.constant([value], dtype=tf.float32, name=label)
            assert(self.tensor.get_shape()[1]==self.domain.tensor.get_shape()[1],"The individual shape '%s' does not match with the '%s' domain shape" % (self.label, self.domain.label))
            self.tensor = tf.constant([value], dtype=tf.float32, name=label)
            self.columns = len(value)
        else:
            self.columns = domain.columns
            self.tensor = tf.Variable(tf.random_normal([1, self.columns], mean=1), name=label)
        current_world.individuals[self.label] = self

class Relation(object):
    def __init__(self, label, domains, function=None):
        if label in current_world.relations:
            raise Exception("Relation %s already exists" % label)
        self.label = label

        if utils.isIterableNotString(domains):
            domains = list(domains)
            for i in range(len(domains)):
                if isinstance(domains[i], str):
                    domains[i] = current_world.domains[domains[i]]
            self.domains = tuple(domains)  # tuple of domains
        else:
            if isinstance(domains, str):
                domains = current_world.domains[domains]
            self.domains = (domains,)
        current_world.relations[label] = self
        self.arity = len(self.domains)
        if function is None:
            raise NotImplementedError("Default Functions in Relation not yet implemented")
        else:
            self.function = function

    def evaluate(self, tensors):

        return self.function.call(*tensors)

class Function(object):
    def __init__(self, label, domains, function):
        if label in current_world.functions:
            raise Exception("Function %s already exists" % label)
        self.label = label

        if utils.isIterableNotString(domains):
            domains = list(domains)
            for i in range(len(domains)):
                if isinstance(domains[i], str):
                    domains[i] = current_world.domains[domains[i]]
            self.domains = tuple(domains)  # tuple of domains
        else:
            if isinstance(domains, str):
                domains = current_world.domains[domains]
            self.domains = (domains,)
        current_world.functions[label] = self

        self.function = function


    def evaluate(self, tensors):

        return self.function.call(*tensors)

class PointwiseConstraint(object):

    def __init__(self, output, labels):
        if isinstance(output, functions.Learner):
            current_world.loss += output.cost(labels)
        # TODO other cases of output (i.e. CLARE.Relation or CLARE.Function)

class Constraint(object):


    def __init__(self, definition, weight=1.0):

        # String of the formula
        self.definition = definition

        # String of variable
        self.ambiguos_variables = {}

        # Each constraint will have a set of variables
        self.variables_dict = {}
        self.variables_list = []
        self.variable_indices = {}

        # Each constraint will have a set of variables
        self.constant_dict = {}

        # We keep track of the columns range associated to each variable
        self.last_column = 0

        # Atomics formulas will be added during the next step (i.e. formula's parsing)
        self.atomics = {}

        # Parsing the FOL formula
        parser = FOLParser()
        self.root = parser.parse(self.definition, constraint=self)

        # This is the shape of the multi-dimensional tensor, where each dimension corresponds to a variable
        if len(self.variables_list)>0: #not gounded formula
            self.cartesian_shape = [tf.shape(a.domain.tensor)[0] for a in self.variables_list]
        else:
            self.cartesian_shape = [1]


        # Compiling the expression tree
        self.root.compile()
        self.tensor = self.root.tensor

        # Adding a loss term to
        # tf.summary.scalar("Contraints/" + "_".join(definition.split()), 1 - self.tensor)
        # constraints_l += (1 - self.tensor)
        current_world.loss += weight * (1 - self.tensor)

    def create_or_get_variable(self, var_name):

        if var_name in self.variables_dict:
            return self.variables_dict[var_name]

        else:
            new_var = Variable(var_name, self)
            self.variables_dict[var_name] = new_var
            self.variable_indices[var_name] = len(self.variables_list)
            self.variables_list.append(new_var)
            return new_var


    def create_or_get_constant(self, id):
        if id in self.constant_dict:
            return self.constant_dict[id]
        else:
            new_const = Constant(id, self)
            self.constant_dict[id] = new_const
            return new_const

def learn(learning_rate=0.001, num_epochs=1000, print_iters=1, sess=None):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(current_world.loss)
    if sess is None:
        sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        if i%print_iters==0:
            _, cost = sess.run((train_op, current_world.loss))
            print(cost)
        else:
            _ = sess.run(train_op)



# def learn(learning_rate=0.001, num_epochs=1000, sess=None):
#     train_op = tf.train.AdamOptimizer(learning_rate).minimize(current_world.loss)
#     if sess is None:
#         sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     min = 10000
#     last = 0
#     count = 0
#     for i in range(10000000):
#         _, last = sess.run((train_op, current_world.loss))
#         if i%100==0: print(last)
#         if last<min:
#             if min-last < 0.001:
#                 count+=1
#             else:
#                 count=0
#                 min=last
#         elif last>min:
#             count+=1
#         if count==300:
#             count=0
#             min = 10000
#             sess.run(tf.global_variables_initializer())
#         if last<0.3:
#             break





