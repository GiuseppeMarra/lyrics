from pyparsing import *
from CLARE_future import current_world as world
import tensorflow as tf
import operator
import copy
ParserElement.enablePackrat()

class PNode(object):
    """The PNode class represents a generic node of a parsing tree.
            Attributes
            ----------
            args : array of PNode s
                Arguments of the node
            label : str
                The node label.
            tensor: tf.Tensor
                The tensor object containing the value of the node

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


class Term(PNode):
    def __init__(self):
        super(Term, self).__init__(None)

class Number(Term):
    """This class represent numbers used in constraint. They are a fast way of using numbers
    without the explicit definition of a numeric domain and of a constant in that domain.

    Attributes
    ----------
    tensor: tf.Tensor
       The tensor object containing the number
    """

    def __init__(self, tokens, constraint):
        super(Number, self).__init__()
        self.number = float(tokens[0])
        self.constraint = constraint

    def compile(self):
        super(Number,self).compile()
        self.tensor = tf.reshape(tf.constant(self.number),
                                 tf.concat((tf.ones_like(self.constraint.cartesian_shape), [1]), axis=0))



class Constant(Term):
    """This class represent a constant belonging to a domain. It is linked one2one with a previously defined
    individual for this domain.

    Attributes
    ----------
    name: str
        The name of this constant in the constraint
    individual: Individual
        The individual object linked to this constant
    domain: Domain
        The domain object of this constant
    tensor: tf.Tensor
       The tensor object containing the constant values
    """
    def __init__(self, name, constraint, domain=None, index=None):
        super(Constant, self).__init__()
        self.name = name
        self.constraint = constraint
        if domain is None:
            self.individual = world.individuals[self.name]
            self.domain = self.individual.domain
            self.tensor = self.individual.tensor
        else:
            self.domain = world.domains[domain]
            self.tensor = self.domain.tensor[index,:]

    def compile(self):
        pass

class Variable(Term):
    """This class represent a variable in a constraint.

       Attributes
       ----------
       name: str
           The name of this variable in the constraint
       domain: Domain
           The domain object of this constant. This is not defined from the beginning but will
           be assigned by functions and predicates on the basis of their domains
       vars: list of :type:Variable
           This is a common property for variable-dependent terms used for correct reshaping of
           tensor based on the variables.
       tensor: tf.Tensor
          The tensor object containing the constant values


       """

    def __init__(self, name, constraint):
        super(Variable, self).__init__()
        self.name = name
        self.constraint = constraint
        if self.name in world.individuals:
            raise ParseFatalException(
                "Ambiguity for the variable '%s' in constraint '%s': in the world, an individual with the same name exists." % (
                    self.name, self.constraint.definition))
        self.domain = None
        self.tensor = None
        self.vars = [
            self]  # it is a generic property of Terms that are variable dependent; needed to keep a standard way to process

    def compile(self):
        if self.tensor == None:
            assert self.domain != None, "Trying to compile variable %s before assigning a domain" % self.name
            self.tensor = self.domain.tensor

            # Expanding the variable to the constraint shape (i.e. the one depending on the variables defined)
            for i in range(len(self.constraint.variables_list)):
                if self.constraint.variables_list[i] != self:
                    self.tensor = tf.expand_dims(self.tensor, i)

    def check_or_assign_domain(self, domain):
        """Variables do not know their domain until a Function or a Relation assign one to them on the basis
        of their domain."""

        if self.domain is None:
            self.domain = domain
        else:
            assert self.domain == domain,\
                   "Inconsistency between the domains in which variable %s has been used. Previous: %s, New: %s" %\
                    (self.name, self.domain.label, domain.label)

def create_or_get_variable(tokens, constraint):
    var_name = tokens[0]
    if var_name in constraint.variables_dict:
        return constraint.variables_dict[var_name]
    else:
        new_var = Variable(var_name, constraint)
        constraint.variables_dict[var_name] = new_var
        constraint.variable_indices[var_name] = len(constraint.variables_list)
        constraint.variables_list.append(new_var)
        return new_var

def create_or_get_constant(tokens, constraint):
    if len(tokens)==1: #constant defined by label
        id = tokens[0]
        if id in constraint.constant_dict:
            return constraint.constant_dict[id]
        else:
            new_const = Constant(id, constraint)
            constraint.constant_dict[id] = new_const
            return new_const
    else: #constant defined by domain indexing (e.g. D[32])
        domain = tokens[0]
        index = int(tokens[1])
        return Constant(name=domain+str(index),
                        constraint=constraint,
                        domain=domain,
                        index=index)




class Function(Term):
    def __init__(self, t, constraint):
        super(Function, self).__init__()
        self.constraint = constraint
        self.name = t[0]
        assert self.name in world.functions, "There is no function " + self.name
        self.args = t[1:]

        self.function = world.functions[self.name]
        self.tensor = None

        assert len(self.args) == len(self.function.domains), "Wrong number of arguments for function " + self.name

        #Checking arguments consistency
        self.all_vars= True
        self.vars = []
        for i, arg in enumerate(self.args):
            assert isinstance(arg, Term),"Function object %s has an argument that is not a Term: %s " % (self.name, str(arg))
            if isinstance(arg, Constant):
                self.all_vars = False
                continue
            elif isinstance(arg, Number):
                self.all_vars = False
                assert self.function.domains[i].tensor.get_shape()[1] == 1,"Function %s does not accept Numbers as %d argument" % (self.name, i)
                continue
            elif isinstance(arg, Variable):
                arg.check_or_assign_domain(self.function.domains[i])
            else: #isinstance(arg, Function):
                self.all_vars = False
            for var in arg.vars:
                if var not in self.vars:
                    self.vars.append(var)

    def compile(self):
        for i in self.args:
            i.compile()

        if len(self.vars)==1 and self.all_vars:
            self.tensor =  self.function.domain_value

        '''We want to work always with tensors of the same number of dimensions (i.e. rank), that is
        the cartesian shape of the constraint. In this code, we create a `function_shape`,
        that is equal to the cartesian shape apart from those variables not present in this
        function, whose dimensions will be set to 1'''
        self.function_shape = copy.copy(self.constraint.cartesian_shape)
        set_ = set(self.vars)
        for i, var_constr in enumerate(self.constraint.variables_list):
            if var_constr not in set_:
                    self.function_shape[i] = 1

        fun_shape = tf.stack(self.function_shape, axis=0)

        tensors = [] #tensor arguments for the function implementation
        for arg in self.args:

            shape = tf.shape(arg.tensor)
            shape_arg = shape[:-1] # shape_arg is same rank of cartesian shape but with 1 in non-dependent variable; we do not consider domain dimension (-1)
            size = shape[-1] #domain dimension
            to_repeat = tf.reshape(fun_shape - shape_arg + 1, [-1]) # putting to 1 all existent axis

            to_repeat = tf.concat((to_repeat, [1]), axis = 0)  # adding no repetition for domain columns
            tensor = tf.tile(arg.tensor, to_repeat)
            tensor = tf.reshape(tensor, [-1, size]) # flattening the tensor into its cartesian product projection
            tensors.append(tensor)

        self.tensor = self.function.evaluate(tensors)

        # bringing back the tensor to its function shape and putting the new dimension as last dimension
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

        assert len(self.args) == len(self.predicate.domains),"Wrong number of arguments for predicate " + self.name


        self._all_vars = True
        self.vars = []
        for i, arg in enumerate(self.args):
            assert isinstance(arg, Term),"Atomic object %s has an argument that is not a Term" % self.name
            if isinstance(arg, Constant):
                self._all_vars = False
                continue
            elif isinstance(arg, Number):
                self._all_vars = False
                assert self.predicate.domains[i].tensor.get_shape()[1] == 1,"Relation %s does not accept Numbers as %d argument" %(self.name, i)
                continue
            elif isinstance(arg, Variable):
                arg.check_or_assign_domain(self.predicate.domains[i])
            else: # isinstance(arg, Function)
                self._all_vars = False
            for var in arg.vars:
                if var not in self.vars:
                    self.vars.append(var)


    def compile(self):
        for i in self.args:
            i.compile()

        if len(self.vars)==1 and self._all_vars:
            self.tensor =  self.predicate.domain_value

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
                The arguments of the node
            label : str
                The node operation label.
            tensor: tf.Tensor
                The tensor object containing the truth value of the operation

            """

    def __init__(self, t):
        super(And, self).__init__(t)

    def compile(self):
        super(And, self).compile()
        self.tensor = world.logic.weak_conj([a.tensor for a in self.args])


class Or(Op):
    """The Or class represents the logical OR operation of a parsing tree.
            Attributes
            ----------
            args : array of PNode s
                The  arguments of the node
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


class Iff(Op):
    def __init__(self, t):
        super(Iff, self).__init__(t)

    def compile(self):
        super(Iff, self).compile()
        assert len(self.args) == 2, "n-ary double implication not allowed. Use parentheses to group chains of implications"
        self.tensor = world.logic.iff(self.args[0].tensor, self.args[1].tensor)


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
            if class_name == "Number":
                return Number(tokens, constraint)
            elif class_name == "Variable":
                return create_or_get_variable(tokens, constraint)
            elif class_name == "Constant":
                return create_or_get_constant(tokens, constraint)
            elif class_name == "Function":
                return Function(tokens, constraint)
            # elif class_name == "PLUS":
            #     return Plus(tokens, constraint)
            # elif class_name == "MINUS":
            #     return Minus(tokens, constraint)
            # elif class_name == "TIMES":
            #     return Multipy(tokens, constraint)
            # elif class_name == "DIVIDE":
            #     return Divide(tokens, constraint)
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
                return Iff(tokens)
            elif class_name == "FORALL":
                return ForAll(constraint, tokens)
            elif class_name == "EXISTS":
                return Exists(constraint, tokens)
            elif class_name == "EXISTN":
                return Exists_n(constraint, tokens)
            elif class_name == "ARITHM_REL":
                #TODO
                raise NotImplementedError("Arithmetic Relations not already implemented")


        return _create

    def parse(self, definition, constraint):

        left_parenthesis, right_parenthesis, colon, left_square, right_square = map(Suppress, "():[]")
        symbol = Word(alphas)


        #Numbers
        point = Literal('.')
        e = CaselessLiteral('E')
        plusorminus = Literal('+') | Literal('-')
        number = Word(nums)
        integer = Combine(Optional(plusorminus) + number)
        floatnumber = Combine(integer +
                              Optional(point + Optional(number)) +
                              Optional(e + integer)
                              )
        floatnumber.setParseAction(self._createParseAction("Number", constraint))

        ''' TERMS '''
        term = Forward()
        var = symbol
        var.setParseAction(self._createParseAction("Variable", constraint))
        const = oneOf(list(world.individuals.keys())) | oneOf(list(world.domains.keys()))+ left_square +Word(nums) + right_square
        const.setParseAction(self._createParseAction("Constant", constraint))

        # #Arithmetic Operators
        # arithm_term = Forward()
        # plus = Keyword("+")
        # minus = Keyword("-")
        # times = Keyword("*")
        # divide = Keyword("/")
        # arithm_term << infixNotation(term, [
        #     (plus, 1, opAssoc.LEFT, self._createParseAction("PLUS", constraint)),
        #     (minus, 2, opAssoc.LEFT, self._createParseAction("MINUS", constraint)),
        #     (times, 2, opAssoc.LEFT, self._createParseAction("TIMES", constraint)),
        #     (divide, 2, opAssoc.LEFT, self._createParseAction("DIVIDE", constraint)),
        # ])
        func = oneOf(list(world.functions.keys()))
        func_term = func + left_parenthesis + delimitedList(term) + right_parenthesis
        func_term.setParseAction(self._createParseAction("Function", constraint))
        # term << (func_term | arithm_term| const | number | var )
        term << (func_term ^ const ^ floatnumber ^ var )

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

        # arithm_relation = term + oneOf([">","<",">=","<="]) + term
        # arithm_relation.setParseAction(self._createParseAction("ARITHM_REL", constraint))

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

        constraint = term ^ formula
        tree = constraint.parseString(definition, parseAll=True)
        # tree = formula.parseString(definition)
        return tree[0]
