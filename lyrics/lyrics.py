import tensorflow as tf
import numpy as np
import utils
import functions
from fuzzy import LogicFactory
import re
import copy


class World(object):

    def __init__(self, logic="lukasiewicz"):

        #Map from domain name to domain object
        self.domains = {}

        #Map from function name to function object
        self.functions = {}

        #Map from relation name to relation object
        self.relations = {}

        #Map from individual id to individual object
        self.individuals = {}

        #This is a storage for already computed function
        self._precomputed = {}

        self.logic = LogicFactory.create(logic)


        self.lambda_p = tf.constant(1.)
        self.pointwise_loss = tf.constant(0.)

        self.lambda_r = tf.constant(1.)
        self.regularization_loss = tf.constant(0.)

        self.lambda_c = tf.constant(1.)
        self.constraint_loss = tf.constant(0.)


        self.lagrangian_loss = tf.constant(0.)

    def reset(self):
        tf.reset_default_graph()


        #Map from domain name to domain object
        self.domains = {}

        #Map from function name to function object
        self.functions = {}

        #Map from relation name to relation object
        self.relations = {}

        #Map from individual id to individual object
        self.individuals = {}

        #This is a storage for already computed function
        self._precomputed = {}


        self.lambda_p = tf.constant(1.)
        self.pointwise_loss = tf.constant(0.)

        self.lambda_r = tf.constant(1.)
        self.regularization_loss = tf.constant(0.)

        self.lambda_c = tf.constant(1.)
        self.constraint_loss = tf.constant(0.)


        self.lagrangian_loss = tf.constant(0.)



    def loss(self):


        tf.summary.scalar("Losses/PointwiseLoss",self.pointwise_loss)
        tf.summary.scalar("Losses/RegularizationLoss",self.regularization_loss)
        tf.summary.scalar("Losses/ConstraintLoss",self.constraint_loss)

        return self.lambda_p*self.pointwise_loss + \
               self.lambda_r*self.regularization_loss + \
               self.lambda_c*self.constraint_loss

    def lagrangian_train_op(self, lr=0.001):
        lagrangian_multipliers = [var for var in tf.trainable_variables() if "lagrangian_" in var.name]
        return tf.train.GradientDescentOptimizer(lr).minimize(-self.lagrangian_loss,
                                                              var_list=lagrangian_multipliers), (tf.reduce_mean([tf.reduce_mean(m) for m in lagrangian_multipliers]),
                                                                                                 tf.reduce_max([tf.reduce_max(m)for m in lagrangian_multipliers]))

    def train_op(self, loss=None, lr=0.001):
        if loss is None:
            loss = self.constraint_loss
        vars = [var for var in tf.trainable_variables() if "lagrangian_" not in var.name]
        return tf.train.AdamOptimizer(lr).minimize(loss, var_list=vars)




current_world = World()
actOpt = False
from parser import FOLParser



class Domain(object):

    @staticmethod
    def get_domain(domain):
        if isinstance(domain, str):
            return current_world.domains[domain]
        elif isinstance(domain, Domain):
            return domain
        else:
            raise Exception("Domain %s not recognised" % str(domain))


    def __init__(self, label, data, dom_type=tf.float32):
        if label in current_world.domains:
            raise Exception("Domain %s already exists" % label)
        self.dom_type = dom_type
        self.label = label
        assert isinstance(data, tf.Tensor) or len(data)>0, "You need to provide at least one element of the domain %s" % (label)
        self.tensor = tf.convert_to_tensor(data)
        assert len(self.tensor.get_shape())==2, "Data for domain %s must be a two-dimensional tensor(i.e. matrix)" % self.label

        self.columns = self.tensor.get_shape()[1].value
        current_world.domains[self.label] = self

        #Map from individual id to the corresponding row inside the domain
        self.individuals = {} #TODO

    def add_individual(self, individual):
        self.tensor = tf.concat((self.tensor, individual.tensor), axis=0)
        self.individuals[individual.label]=self

    def __str__(self):
        return str(self.tensor.numpy())

class SubDomain(Domain):

    def __init__(self, label, data, father):
        super(SubDomain,self).__init__(label=label,data=data)

        self.father = Domain.get_domain(father)
        assert isinstance(self.father, Domain) or isinstance(self.father, SubDomain), "super attribute must be a Domain or a SubDomain"
        self.ancestors = set(self._get_ancestors())

    def _get_ancestors(self):

        if isinstance(self.father,Domain):
            return [self, self.father]
        return [self] + self.father._get_ancestors()





class RangeDomain(Domain):

    def __init__(self, label, max_range):
        super(RangeDomain,self).__init__(label=label,
                                         data=np.reshape(np.arange(max_range), [-1,1]))


class Individual(object):
    def __init__(self, label, domain, value=None):
        if label in current_world.individuals:
            raise Exception("Duplicate individual id %s " % label)
        self.label = label
        self.domain = Domain.get_domain(domain)
        if value is not None:
            if isinstance(value, tf.Variable):
                self.tensor = value
            else:
                self.tensor = tf.constant([value], dtype=tf.float32, name=label)
            assert self.tensor.get_shape()[1]==self.domain.tensor.get_shape()[1],"The individual shape '%s' does not match with the '%s' domain shape" % (self.label, self.domain.label)
            self.columns = self.domain.columns
        else:
            self.columns = self.domain.columns
            self.tensor = tf.get_variable(name=self.label, initializer=tf.random_normal(shape=[1, self.columns]))
        current_world.individuals[self.label] = self
        self.domain.individuals[self.label] = self.domain.tensor.get_shape()[0]
        self.domain.tensor = tf.concat((self.domain.tensor,self.tensor), axis=0)


def compute_domains_tensor(domains):

    num_domains = len(domains)
    domains_rows= [domain.tensor.get_shape()[0] for domain in domains]
    domains_columns = [domain.tensor.get_shape()[1] for domain in domains]
    tensors = []
    for i,domain in enumerate(domains):
        tensor = domain.tensor


        to_reshape = [1 for _ in range(num_domains)]
        to_reshape[i] = domains_rows[i]
        to_reshape = to_reshape+ [domains_columns[i]]
        tensor = tf.reshape(tensor, to_reshape)


        to_repeat = copy.copy(domains_rows)
        to_repeat[i] = 1
        to_repeat = to_repeat+[1]
        tensor = tf.tile(tensor, to_repeat)


        tensor = tf.reshape(tensor, [-1, domains_columns[i]])
        tensors.append(tensor)

    return tensors

class Function(object):
    def __init__(self, label, domains, function, features=True):
        if label in current_world.functions:
            raise Exception("Function %s already exists" % label)
        self.label = label
        self.features = features

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
        self.arity = len(self.domains)


        if function is None:
            raise NotImplementedError("Default functions implementation in Function not yet implemented")
        else:
            self.function = function

        # if isinstance(self.function, functions.Slice):
        #     father_function = self.function.function
        #     if constants.DOMAINS_VALUE + str(father_function) not in current_world._precomputed:
        #         domains_cartesian_tensors = compute_domains_tensor(self.domains)
        #         current_world._precomputed[constants.DOMAINS_VALUE + str(father_function)] = father_function(
        #             *domains_cartesian_tensors)
        # with tf.variable_scope(constants.DOMAINS_VALUE + label):
        #     if isinstance(function, functions.Slice):
        #         current_world._precomputed[constants.DOMAINS_VALUE + label] = \
        #             current_world._precomputed[constants.DOMAINS_VALUE + str(self.function.function)][:,
        #             self.function.axis]
        #     else:
        #         if self.arity == 1:
        #             domains_cartesian_tensors = [self.domains[0].tensor]
        #         else:
        #             domains_cartesian_tensors = compute_domains_tensor(self.domains)
        #         current_world._precomputed[constants.DOMAINS_VALUE + label] = self.function(
        #             *domains_cartesian_tensors)


class Relation(object):
    def __init__(self, label, domains, function, features=True, lambda_block = tf.constant(False)):
        if label in current_world.relations:
            raise Exception("Relation %s already exists" % label)
        self.label = label
        self.features = features
        self.lambda_block = lambda_block


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
            raise NotImplementedError("Default function implementation in Relation not yet implemented")
        else:
            self.function = function

        # if isinstance(self.function, functions.Slice):
        #     father_function = self.function.function
        #     if constants.DOMAINS_VALUE+str(father_function) not in current_world._precomputed:
        #         domains_cartesian_tensors = compute_domains_tensor(self.domains)
        #         current_world._precomputed[constants.DOMAINS_VALUE + str(father_function)] = father_function(
        #             *domains_cartesian_tensors)
        # with tf.variable_scope(constants.DOMAINS_VALUE+label):
        #     if isinstance(function, functions.Slice):
        #         current_world._precomputed[constants.DOMAINS_VALUE+label] = \
        #             current_world._precomputed[constants.DOMAINS_VALUE + str(self.function.function)][:, self.function.axis]
        #     else:
        #         if self.arity == 1:
        #             domains_cartesian_tensors = [self.domains[0].tensor]
        #         else:
        #             domains_cartesian_tensors = compute_domains_tensor(self.domains)
        #         current_world._precomputed[constants.DOMAINS_VALUE+label] = self.function(*domains_cartesian_tensors)




class PointwiseConstraint(object):


    def __init__(self, function, labels ,*inputs):
        # if isinstance(function, functions.Learner):
            current_world.pointwise_loss += function.cost(labels, *inputs)
        # TODO other cases of output (i.e. CLARE.Relation or CLARE.Function)


class RegularizationConstraint(object):

    def __init__(self, function, weight):
        if isinstance(function, functions.RegularizedLearner):
            current_world.regularization_loss += weight * function.regularization_cost()
        # TODO other cases of output (i.e. CLARE.Relation or CLARE.Function)

class Constraint(object):

    count = 0


    def __init__(self, definition, weight=None, filter= None, hard=False, validation_block=False):

        self.id = Constraint.count+ 1
        Constraint.count += 1

        self.validation_block = validation_block

        # String of the formula
        self.definition = definition

        # String of variable
        self.ambiguos_variables = {}

        # Each constraint will have a set of variables
        self.variables_dict = {}
        self.variables_list = []
        self.variable_indices = {}

        # Each constraint will have a set of constants
        self.constant_dict = {}

        # We keep track of the columns range associated to each variable
        self.last_column = 0

        # Parsing the FOL formula
        parser = FOLParser()
        self.root = parser.parse(self.definition, constraint=self)


        #External filtering handling
        if filter is not None:
            self.filter(filter)


        with tf.name_scope(re.sub('[^0-9a-zA-Z_]', '', self.definition)):

            with tf.name_scope("CartesianShape"):
                # This is the shape of the multi-dimensional tensor, where each dimension corresponds to a variable
                if len(self.variables_list)>0: #not gounded formula
                    try:
                        self.cartesian_shape = [a.domain.tensor.shape.as_list()[0] for a in self.variables_list]
                    except AttributeError: #None domain for a variable
                        raise Exception("In constraint [%s], a variable has not been used in any predicate or function" % self.definition)
                else:
                    self.cartesian_shape = [1]



        # Compiling the expression tree
        self.root.compile()
        self.tensor = self.root.tensor



        c_weight = 1.
        if weight is not None:
            c_weight = weight
        if hard:
            if len(self.tensor.get_shape()) == 0:
                c_weight = c_weight*tf.get_variable("lagrangian_" + str(self.id), initializer=0.)
            else:
                c_weight = c_weight*tf.get_variable("lagrangian_" + str(self.id), initializer=tf.zeros_like(self.tensor))


        def hinge(x):
            return tf.where(x > 0.90, tf.ones_like(x), x)

        current_world.constraint_loss += tf.reduce_sum(c_weight * current_world.logic.loss(self.tensor))
        current_world.lagrangian_loss += tf.reduce_sum(c_weight * current_world.logic.loss(hinge(self.tensor)))


    def filter(self, filter):
        # Filtering
        for k, v in filter.items():
            v = Domain.get_domain(v)
            assert isinstance(v, SubDomain)
            self.variables_dict[k].check_or_assign_domain(v, force=True)




class Query(object):

    count = 0

    def __init__(self, definition, filter= None, hard = False):


        self.id = Query.count
        Query.count+=1

        self.validation_block = False
        # String of the formula
        self.definition = definition

        # String of variable
        self.ambiguos_variables = {}

        # Each constraint will have a set of variables
        self.variables_dict = {}
        self.variables_list = []
        self.variable_indices = {}

        # Each constraint will have a set of constants
        self.constant_dict = {}

        # We keep track of the columns range associated to each variable
        self.last_column = 0

        # Parsing the FOL formula
        parser = FOLParser()
        self.root = parser.parse(self.definition, constraint=self)


        #External filtering handling
        if filter is not None:
            self.filter(filter)


        with tf.name_scope(re.sub('[^0-9a-zA-Z_]', '', self.definition)):

            with tf.name_scope("CartesianShape"):
                # This is the shape of the multi-dimensional tensor, where each dimension corresponds to a variable
                if len(self.variables_list)>0: #not gounded formula
                    try:
                        self.cartesian_shape = [a.domain.tensor.shape.as_list()[0] for a in self.variables_list]
                    except AttributeError: #None domain for a variable
                        raise Exception("In constraint [%s], a variable has not been used in any predicate or function" % self.definition)
                else:
                    self.cartesian_shape = [1]





        # Compiling the expression tree
        self.root.compile()
        self.tensor = self.root.tensor


    def filter(self, filter):
        # Filtering
        for k, v in filter.items():
            v = Domain.get_domain(v)
            assert isinstance(v, SubDomain)
            self.variables_dict[k].check_or_assign_domain(v, force=True)








