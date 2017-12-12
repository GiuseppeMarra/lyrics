from pyparsing import *
from CLARE import current_world as world
import tensorflow as tf
import operator
import copy
ParserElement.enablePackrat()


class Variable(object):

        def __init__(self, name, constraint):
            self.name = name
            self.constraint = constraint
            if self.name in world.individuals:
                raise ParseFatalException(
                    "Ambiguity for the variable '%s' in constraint '%s': in the world, an individual with the same name exists." % (
                    self.name, self.constraint.definition))
            self.domain = None
            self.tensor = None
            self.vars = [self] #it is a generic property of Terms; needed to keep a standard way to process

        def compile(self):
            if self.tensor== None:
                assert(self.domain!=None, "Trying to compile variable %s before assigning a domain" % self.name)
                self.tensor = self.domain.tensor
                for i in range(len(self.constraint.variables_list)):
                    if self.constraint.variables_list[i]!=self:
                        self.tensor = tf.expand_dims(self.tensor, i)



        def check_or_assign_domain(self, domain):
            if self.domain is None:
                self.domain = domain
            else:
                assert (self.domain == domain,
                        "Inconsistency between the domains in which variable %s has been used. Previous: %s, New: %s" %
                        (self.name, self.domain.label, domain.label))


class Constant(object):
    def __init__(self, name, constraint):
        self.name = name
        self.constraint = constraint
        self.individual = world.individuals[self.name]
        self.domain = self.individual.domain
        self.tensor = self.individual.tensor

    def compile(self):
        pass

def create_or_get_variable(tokens, constraint):
    var_name = tokens[0]
    return constraint.create_or_get_variable(var_name)

def create_or_get_constant(tokens, constraint):
    const_id = tokens[0]
    return constraint.create_or_get_constant(const_id)


class Function(object):
    def __init__(self, t, constraint):
        self.constraint = constraint
        self.name = t[0]
        if self.name not in world.functions:
            raise Exception("There is no function " + self.name)
        self.args = t[1:]

        self.function = world.functions[self.name]
        self.tensor = None

        if len(self.args) != len(self.function.domains):
            raise Exception("Wrong number of variables for function " + self.name)

        self.all_vars = True # flag for functions defined only on variables
        self.argument_to_variable = {}
        for i, v in enumerate(self.args):
            assert (isinstance(v, Variable) or isinstance(v, Constant) or isinstance(v, Function),
                    "Function object %s has an argument that is not Variable nor a Constant nor a Function: %s " % (
                    self.name, str(v)))
            if isinstance(v, Variable):
                v.check_or_assign_domain(self.function.domains[i])
            else:
                self.all_vars = False

        self.vars = []
        for arg in self.args:
            if isinstance(arg, Constant): continue
            for var in arg.vars:
                if var not in self.vars:
                    self.vars.append(var)



    def compile(self):

        for i in self.args:
            i.compile()


        self.function_shape = copy.copy(self.constraint.cartesian_shape)
        set_ = set(self.vars)
        for i, var_constr in enumerate(self.constraint.variables_list):
            if var_constr not in set_:
                    self.function_shape[i] = 1

        fun_shape = tf.stack(self.function_shape, axis=0)

        tensors = []
        for arg in self.args:

            shape = tf.shape(arg.tensor)
            shape_arg = shape[:-1] # not considering domain columns
            size = shape[-1]
            to_repeat = tf.reshape(fun_shape - shape_arg + 1, [-1]) # putting to 1 all existent axis

            to_repeat = tf.concat((to_repeat, [1]), axis = 0)  # adding no repetition for domain columns
            tensor = tf.tile(arg.tensor, to_repeat)
            tensor = tf.reshape(tensor, [-1, size]) # flattening the tensor into its cartesian product projection
            tensors.append(tensor)

        self.tensor = self.function.evaluate(tensors)

        # bringing back the tensor to its function shape and puttind the new dimension as last dimension
        last_dim = tf.shape(self.tensor)[-1]
        self.tensor = tf.reshape(self.tensor, tf.concat((fun_shape, [last_dim]), 0))


class Atomic(object):
    def __init__(self, t, constraint):
        self.constraint = constraint
        self.name = t[0]
        if self.name not in world.relations:
            raise Exception("There is no predicate " + self.name)
        self.args = t[1:]

        self.predicate = world.relations[self.name]
        self.tensor = None

        if len(self.args) != len(self.predicate.domains):
            raise Exception("Wrong number of variables for predicate " + self.name)

        for i, v in enumerate(self.args):
            assert (isinstance(v, Variable) or isinstance(v, Constant) or isinstance(v, Function),
                    "Atomic object %s has an argument that is not Variable nor a Constant nor a Function: %s " % (
                    self.name, str(v)))
            if isinstance(v, Variable):
                v.check_or_assign_domain(self.predicate.domains[i])


        if self.name in self.constraint.atomics:
            self.constraint.atomics[self.name].append(self)
        else:
            self.constraint.atomics[self.name] = [self]


        self.vars = []
        for arg in self.args:
            if isinstance(arg, Constant): continue
            for var in arg.vars:
                if var not in self.vars:
                    self.vars.append(var)


    def compile(self):
        for i in self.args:
            i.compile()

        self.function_shape = copy.copy(self.constraint.cartesian_shape)
        set_ = set(self.vars)
        for i, var_constr in enumerate(self.constraint.variables_list):
            if var_constr not in set_:
                    self.function_shape[i] = 1

        fun_shape = tf.stack(self.function_shape, axis=0)

        tensors = []
        for arg in self.args:

            shape = tf.shape(arg.tensor)
            shape_arg = shape[:-1] # not considering domain columns
            size = shape[-1]
            to_repeat = tf.reshape(fun_shape - shape_arg + 1, [-1])

            to_repeat = tf.concat((to_repeat, [1]), axis = 0)  # adding no repetition for domain columns
            tensor = tf.tile(arg.tensor, to_repeat)
            tensor = tf.reshape(tensor, [-1, size])
            tensors.append(tensor)

        self.tensor = self.predicate.evaluate(tensors)
        self.tensor = tf.reshape(self.tensor, fun_shape)


        #Atomics tile immediately to cartesian shape (they are monodimensional)
        #TODO think if it is necessary to make the connectives expand dimensions
        shape_arg = tf.shape(self.tensor)
        to_repeat = tf.stack(self.constraint.cartesian_shape, axis=0) - shape_arg + 1
        self.tensor = tf.tile(self.tensor, to_repeat)



class PNode(object):
    """The PNode class represents a generic node of a parsing tree.
            Attributes
            ----------
            args : array of PNode s
                Arguments of the node
            label : str
                The node operation label.
            tensor: tf.Tensor
                The tensor object containing the truth value of the operation

            """

    def __init__(self, t):
        self.args = []
        self.label = None
        self.tensor = None

    def compile(self):
        for i in self.args:
            i.compile()

    def __str__(self):
        return self.label

    def __iter__(self):
        for i in self.args:
            yield i


class Op(PNode):
    """The Op class represents a generic n-ary operation of a parsing tree.
            Attributes
            ----------
            args : array of PNode s
                The arguments of the node
            label : str
                The node operation label.
            tensor: tf.Tensor
                The tensor object containing the truth value of the operation

            """

    def __init__(self, t):
        super(Op, self).__init__(t)
        self.args = t[0][0::2]
        self.label = t[0][1]


class And(Op):
    """The And class represents the logical AND operation of a parsing tree.
            Attributes
            ----------
            args : array of PNode s
                The two arguments of the node
            label : str
                The node operation label.
            tensor: tf.Tensor
                The tensor object containing the truth value of the operation

            """

    def __init__(self, t):
        super(And, self).__init__(t)

    def compile(self):
        super(And, self).compile()
        sess = tf.Session()
        self.tensor = world.logic.weak_conj([a.tensor for a in self.args])


class Or(Op):
    """The Or class represents the logical OR operation of a parsing tree.
            Attributes
            ----------
            args : array of PNode s
                The two arguments of the node
            label : str
                The node operation label.
            tensor: tf.Tensor
                The tensor object containing the truth value of the operation

            """

    def __init__(self, t):
        super(Or, self).__init__(t)

    def compile(self):
        super(Or, self).compile()
        self.tensor = world.logic.strong_disj([a.tensor for a in self.args])


class Implies(Op):
    """The Implies class represents the logical IMPLICATION operation of a parsing tree.
            Attributes
            ----------
            args : array of PNode s
                The two arguments of the node
            label : str
                The node operation label.
            tensor: tf.Tensor
                The tensor object containing the truth value of the operation

            """

    def __init__(self, t):
        super(Implies, self).__init__(t)

    def compile(self):
        super(Implies, self).compile()
        assert len(self.args) == 2, "n-ary implication not allowed. Use parentheses to group chains of implications"
        self.tensor = world.logic.implication(self.args[0].tensor, self.args[1].tensor)


class Not(PNode):
    """The Not class represents the logical NOT operation of a parsing tree.
               Attributes
               ----------
               args : array of PNode s
                   The single argument of the node
               label : str
                   The node operation label.
               tensor: tf.Tensor
                   The tensor object containing the truth value of the operation

               """

    def __init__(self, t):
        super(Not, self).__init__(t)
        self.args = [t[0][1]]
        self.label = 'not'

    def compile(self):
        super(Not, self).compile()
        assert len(self.args) == 1
        self.tensor = world.logic.negation(self.args[0].tensor)


class Quantifier(PNode):
    """The Quatifier class represents a logical quantifier of a parsing tree.
                   Attributes
                   ----------
                   args : array of PNode s
                       The single argument of the node
                   label : str
                       The node operation label.
                   tensor: tf.Tensor
                       The tensor object containing the truth value of the operation
                   var: str
                        The variable of the quantifier
                   """

    def __init__(self, t):
        super(Quantifier, self).__init__(t)
        self.args = [t[2][0]]
        self.label = t[0]
        self.var = t[1]

    def __str__(self):
        return self.label + " " + self.var


class ForAll(Quantifier):
    def __init__(self, constraint, t):
        super(ForAll, self).__init__(t)
        self.constraint = constraint

    def compile(self):
        super(ForAll, self).compile()
        var_axis = self.constraint.variable_indices[self.var.name]
        self.tensor = world.logic.forall(self.args[0].tensor, var_axis)
        self.constraint.cartesian_shape[var_axis] = 1


class Exists(Quantifier):
    def __init__(self, constraint, t):
        super(Exists, self).__init__(t)
        self.constraint = constraint

    def compile(self):
        super(Exists, self).compile()
        var_axis = self.constraint.variable_indices[self.var.name]
        self.tensor = world.logic.exists(self.args[0].tensor, var_axis)
        self.constraint.cartesian_shape[var_axis] = 1


class Exists_n(Quantifier):
    def __init__(self, constraint, t):
        super(Exists_n, self).__init__(t)
        self.constraint = constraint
        self.args = [t[0][3]]
        self.label = t[0][0]
        self.n = int(t[0][1])
        self.var = t[0][2]

    def compile(self):
        super(Exists_n, self).compile()
        var_axis = self.constraint.variables_indices[self.var.name]
        self.tensor = world.logic.exists_n(self.args[0].tensor, var_axis, self.n)
        self.constraint.cartesian_shape[var_axis] = 1


class FOLParser(object):
    def _createParseAction(self, class_name, constraint):
        def _create(tokens):
            if class_name == "Variable":
                return create_or_get_variable(tokens, constraint)
            elif class_name == "Constant":
                return create_or_get_constant(tokens, constraint)
            elif class_name == "Function":
                return Function(tokens, constraint)
            elif class_name == "Atomic":
                return Atomic(tokens, constraint)
            elif class_name == "NOT":
                return Not(tokens)
            elif class_name == "AND":
                return And(tokens)
            elif class_name == "OR":
                return Or(tokens)
            elif class_name == "IMPLIES":
                return Implies(tokens)
            elif class_name == "IFF":
                return (tokens, "IFF")
            elif class_name == "FORALL":
                return ForAll(constraint, tokens)
            elif class_name == "EXISTS":
                return Exists(constraint, tokens)
            elif class_name == "EXISTN":
                return Exists_n(constraint, tokens)

        return _create

    # def parse(self, definition, constraint):
    #
    #     left_parenthesis, right_parenthesis, colon = map(Suppress, "():")
    #     symbol = Word(alphas)
    #     number = Word(nums)
    #
    #     ''' TERMS '''
    #     term = Forward()
    #     var = symbol
    #     var.setParseAction(self._createParseAction("Variable", constraint))
    #     const = oneOf(list(world.individuals.keys()))
    #     const.setParseAction(self._createParseAction("Constant", constraint))
    #     func = oneOf(list(world.functions.keys()))
    #     func_term = func + left_parenthesis + delimitedList(term) + right_parenthesis
    #     func_term.setParseAction(self._createParseAction("Function", constraint))
    #     term << (func_term | const | var )
    #
    #     ''' FORMULAS '''
    #     formula = Forward()
    #     not_ = Keyword("not")
    #     and_ = Keyword("and")
    #     or_ = Keyword("or")
    #     implies = Keyword("->")
    #     iff = Keyword("<->")
    #     forall = Keyword("forall") + var + colon
    #     # forall.setParseAction(self._createParseAction("FORALL", constraint))
    #     exists = Keyword("exists") + var + colon
    #     # exists.setParseAction(self._createParseAction("EXISTS", constraint))
    #     exists_n = Keyword("existn ") + number + var + colon
    #     # exists_n.setParseAction(self._createParseAction("EXISTN", constraint))
    #     # quantifier = forall | exists | exists_n
    #
    #     relation = oneOf(list(world.relations))
    #     atomic_formula = relation + left_parenthesis + delimitedList(term) + right_parenthesis
    #     atomic_formula.setParseAction(self._createParseAction("Atomic", constraint))
    #     formula << infixNotation(atomic_formula, [
    #         (not_, 1, opAssoc.RIGHT, self._createParseAction("NOT", constraint)),
    #         (and_, 2, opAssoc.LEFT, self._createParseAction("AND", constraint)),
    #         (or_, 2, opAssoc.LEFT, self._createParseAction("OR", constraint)),
    #         (implies, 2, opAssoc.RIGHT, self._createParseAction("IMPLIES", constraint)),
    #         (iff, 2, opAssoc.RIGHT, self._createParseAction("IFF", constraint)),
    #         (exists, 1, opAssoc.LEFT, self._createParseAction("EXISTS", constraint)),
    #         # (exists_n, 1, opAssoc.RIGHT, self._createParseAction("EXISTN", constraint)),
    #         (forall, 1, opAssoc.RIGHT, self._createParseAction("FORALL", constraint)),
    #     ])
    #
    #     tree = formula.parseString(definition, parseAll=True)
    #     return tree[0]

    def parse(self, definition, constraint):

        left_parenthesis, right_parenthesis, colon = map(Suppress, "():")
        symbol = Word(alphas)
        number = Word(nums)

        ''' TERMS '''
        term = Forward()
        var = symbol
        var.setParseAction(self._createParseAction("Variable", constraint))
        const = oneOf(list(world.individuals.keys()))
        const.setParseAction(self._createParseAction("Constant", constraint))
        func = oneOf(list(world.functions.keys()))
        func_term = func + left_parenthesis + delimitedList(term) + right_parenthesis
        func_term.setParseAction(self._createParseAction("Function", constraint))
        term << (func_term | const | var )

        ''' FORMULAS '''
        formula = Forward()
        not_ = Keyword("not")
        and_ = Keyword("and")
        or_ = Keyword("or")
        implies = Keyword("->")
        iff = Keyword("<->")

        forall = Keyword("forall")
        exists = Keyword("exists")
        forall_expression = forall + symbol + colon + Group(formula)
        forall_expression.setParseAction(self._createParseAction("FORALL", constraint))
        exists_expression = exists + symbol + colon + Group(formula)
        exists_expression.setParseAction(self._createParseAction("EXISTS", constraint))


        relation = oneOf(list(world.relations))
        atomic_formula = relation + left_parenthesis + delimitedList(term) + right_parenthesis
        atomic_formula.setParseAction(self._createParseAction("Atomic", constraint))
        espression = forall_expression | exists_expression | atomic_formula
        formula << infixNotation(espression, [
            (not_, 1, opAssoc.RIGHT, self._createParseAction("NOT", constraint)),
            (and_, 2, opAssoc.LEFT, self._createParseAction("AND", constraint)),
            (or_, 2, opAssoc.LEFT, self._createParseAction("OR", constraint)),
            (implies, 2, opAssoc.RIGHT, self._createParseAction("IMPLIES", constraint)),
            (iff, 2, opAssoc.RIGHT, self._createParseAction("IFF", constraint))
        ])

        tree = formula.parseString(definition, parseAll=True)
        return tree[0]
