import geonerative.Geometric_Category as GC
import string
from functools import partial
global_objects = {}
# TOKENS

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_GEO = 'GEO'
TT_STR = 'STR'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD = 'KEYWORD'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_EQUALS = 'EQUALS'
TT_DEQUALS = 'DEQUALS'
TT_NEQUALS = 'NEQUALS'
TT_LTE = 'LTE'
TT_GTE = 'GTE'
TT_LT = 'LT'
TT_GT = 'GT'
TT_ANDSYMBOL = 'ANDSYMBOL'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EOF'

KEYWORDS = [
    'VAR',
    'AND',
    'OR',
    'NOT',
    'IF',
    'THEN',
    'ELIF',
    'ELSE',
    'SHOW',
    'HIDE',
    'create',
    'circle',
    'with',
    'radius',
    'x_loc',
    'y_loc',
    'color'
]


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start
            self.pos_end = self.pos_start.copy()
            self.pos_end.advance()
        if pos_end:
            self.pos_end = pos_end

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


# CONSTANTS

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS


# ERRORS

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.error_name = error_name
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end

    def as_string(self):
        result = f'{self.error_name} : {self.details}'
        result += f'\nFile {self.pos_start.file_name}, line {self.pos_start.line + 1}'
        return result


class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)


class RunTimeError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'RunTime Error', details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name} : {self.details}'
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context
        while ctx:
            result = f'  File {pos.file_name}, line {str(pos.line + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        return 'Traceback (most recent call last):\n' + result


# POSITION

class Position:
    def __init__(self, index, line, column, file_name, file_text):
        self.index = index
        self.line = line
        self.column = column
        self.file_name = file_name
        self.file_text = file_text

    def advance(self, current_char=None):
        self.index += 1
        self.column += 1
        if current_char == '\n':
            self.line += 1
            self.column = 0

        return self

    def copy(self):
        return Position(self.index, self.line, self.column, self.file_name, self.file_text)

# LEXER

class Lexer:
    def __init__(self, file_name, text):
        self.text = text
        self.file_name = file_name
        self.pos = Position(-1, 0, -1, file_name, text)
        self.current_char = None

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.index] if self.pos.index < len(self.text) else None

    def make_tokens(self):
        tokens = []
        single_symbol_tokens = {'+': TT_PLUS, '-': TT_MINUS, '*': TT_MUL, '/': TT_DIV, '(': TT_LPAREN, ')': TT_RPAREN,
                                '^': TT_POW, ':': TT_ANDSYMBOL}
        self.advance()
        while self.current_char:
            if self.current_char in r' \t\n':
                self.advance()
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif single_symbol_tokens.get(self.current_char):
                tokens.append(Token(single_symbol_tokens[self.current_char], pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                token, error = self.make_not_equals()
                if error: return [], error
                tokens.append(token)
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            elif self.current_char in DIGITS + '.':
                tokens.append(self.make_number())
            elif self.current_char == '#':
                tokens.append(self.make_geo())
            elif self.current_char == '"':
                tokens.append(self.make_str())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_geo(self):
        id_ = ''
        pos_start = self.pos.copy()
        self.advance()
        while self.current_char and self.current_char in LETTERS_DIGITS + '_':
            id_ += self.current_char
            self.advance()
        return Token(TT_GEO, id_, pos_start, self.pos)

    def make_str(self):
        str_ = ''
        pos_start = self.pos.copy()
        self.advance()
        while self.current_char and self.current_char != '"':
            str_ += self.current_char
            self.advance()
        if self.current_char == '"':
            self.advance()
            return Token(TT_STR, str_, pos_start, self.pos)
        return None, ExpectedCharError(pos_start, self.pos, "'" + '"' + "'")

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()
        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(TT_NEQUALS, pos_start=pos_start, pos_end=self.pos), None
        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' after '!'")

    def make_equals(self):
        token_type = TT_EQUALS
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            token_type = TT_DEQUALS
        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        token_type = TT_LT
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            token_type = TT_LTE
        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        token_type = TT_GT
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            token_type = TT_GTE
        return Token(token_type, pos_start=pos_start, pos_end=self.pos)


# NODES

class NumberNode:
    def __init__(self, token):
        self.token = token
        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end

    def __repr__(self):
        return f'{self.token}'


class StringNode:
    def __init__(self, token):
        self.token = token
        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end

    def __repr__(self):
        return f'{self.token}'


class GeoNode:
    def __init__(self, token):
        self.token = token
        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end

    def __repr__(self):
        return f'{self.token}'


class VarAccessNode:
    def __init__(self, var_name_token):
        self.var_name_token = var_name_token
        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.var_name_token.pos_end


class VarAssignNode:
    def __init__(self, var_name_token, value_node):
        self.var_name_token = var_name_token
        self.value_node = value_node
        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.value_node.pos_end


class BinaryNode:
    def __init__(self, left_node, token, right_node):
        self.left_node = left_node
        self.token = token
        self.right_node = right_node
        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.token}, {self.right_node})'


class UnaryNode:
    def __init__(self, op_token, node):
        self.op_token = op_token
        self.node = node
        self.pos_start = self.op_token.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self):
        return f'({self.op_token}, {self.node})'


class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case
        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases) - 1][1]).pos_end


class CreateNode:
    def __init__(self, properties, pos_end):
        self.properties = properties
        self.pos_start = self.properties['id'].pos_start
        self.pos_end = pos_end


# PARSER

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0

    def register_advancement(self):
        self.advance_count += 1

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error:
            self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_index = -1
        self.advance()

    def advance(self):
        self.token_index += 1
        if self.token_index < len(self.tokens):
            self.current_token = self.tokens[self.token_index]
        return self.current_token

    #####################################

    def parse(self):
        res = self.expr()
        if not res.error and self.current_token.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected '+', '-', '*' or '/'"
            ))
        return res

    #####################################

    def atom(self):
        res = ParseResult()
        token = self.current_token
        if token.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(token))
        elif token.type == TT_GEO:
            res.register_advancement()
            self.advance()
            return res.success(GeoNode(token))
        elif token.type == TT_STR:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(token))
        elif token.type == TT_IDENTIFIER:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(token))

        elif token.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_token.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end, "Expected ')'"
                ))

        elif token.matches(TT_KEYWORD, 'IF'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif token.matches(TT_KEYWORD, 'create'):
            create_expr = res.register(self.create_expr())
            if res.error:
                return res
            return res.success(create_expr)

        return res.failure(InvalidSyntaxError(
            token.pos_start, token.pos_end,
            "Expected int, float, identifier, '+', '-' or '('"
        ))

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None
        res.register_advancement()
        self.advance()
        condition = res.register(self.expr())
        if res.error:
            return res
        if not self.current_token.matches(TT_KEYWORD, 'THEN'):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected 'THEN'"
            ))
        res.register_advancement()
        self.advance()
        expr = res.register(self.expr())
        if res.error:
            return res
        cases.append((condition, expr))
        while self.current_token.matches(TT_KEYWORD, 'ELIF'):
            res.register_advancement()
            self.advance()
            condition = res.register(self.expr())
            if res.error:
                return res
            if not self.current_token.matches(TT_KEYWORD, 'THEN'):
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    f"Expected 'THEN'"
                ))
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            cases.append((condition, expr))
        if self.current_token.matches(TT_KEYWORD, 'ELSE'):
            res.register_advancement()
            self.advance()
            else_case = res.register(self.expr())
            if res.error:
                return res
        return res.success(IfNode(cases, else_case))

    def create_expr(self):
        res = ParseResult()
        properties = {}
        res.register_advancement()
        self.advance()
        properties['type'] = self.current_token.value
        if self.current_token.matches(TT_KEYWORD, 'circle'):
            res.register_advancement()
            self.advance()
            id_expr = res.register(self.expr())
            if res.error:
                return res
            properties['id'] = id_expr
            properties_counter = 0
            if self.current_token.matches(TT_KEYWORD, 'with'):
                while properties_counter == 0 or self.current_token.type == TT_ANDSYMBOL:
                    properties_counter += 1
                    res.register_advancement()
                    self.advance()
                    if self.current_token.type == TT_KEYWORD:
                        property_ = self.current_token.value
                        res.register_advancement()
                        self.advance()
                        if not self.current_token.type == TT_EQUALS:
                            return res.failure(InvalidSyntaxError(
                                self.current_token.pos_start, self.current_token.pos_end,
                                f"Expected '='"
                            ))
                        res.register_advancement()
                        self.advance()
                        value = res.register(self.expr())
                        properties[property_] = value
                    else:
                        return res.failure(InvalidSyntaxError(
                            self.current_token.pos_start, self.current_token.pos_end,
                            f"Expected property"
                        ))
                return res.success(CreateNode(properties, value.pos_end))
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected keyword 'with'"
            ))
        return res.failure(InvalidSyntaxError(
            self.current_token.pos_start, self.current_token.pos_end,
            f"Expected geometric object"
        ))

    def power(self):
        return self.binary_op(self.atom, (TT_POW,), self.factor)

    def factor(self):
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()

            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryNode(token, factor))

        return self.power()

    def term(self):
        return self.binary_op(self.factor, (TT_MUL, TT_DIV))

    def comp_expr(self):
        res = ParseResult()
        if self.current_token.matches(TT_KEYWORD, 'NOT') or self.current_token.matches(TT_KEYWORD, 'SHOW') \
                or self.current_token.matches(TT_KEYWORD, 'HIDE'):
            op_token = self.current_token
            res.register_advancement()
            self.advance()
            node = res.register(self.comp_expr())
            if res.error:
                return res
            return res.success(UnaryNode(op_token, node))
        node = res.register(self.binary_op(self.arith_expr, (TT_DEQUALS, TT_NEQUALS, TT_LT, TT_GT, TT_LTE, TT_GTE)))
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected int, float, identifier, '+', '-', '(' or 'NOT'"
            ))
        return res.success(node)

    def arith_expr(self):
        return self.binary_op(self.term, (TT_PLUS, TT_MINUS))

    def expr(self):
        res = ParseResult()
        if self.current_token.matches(TT_KEYWORD, 'VAR'):
            res.register_advancement()
            self.advance()
            if self.current_token.type != TT_IDENTIFIER:
                return res.failure(
                    InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, "Expected identifier"))
            var_name = self.current_token
            res.register_advancement()
            self.advance()
            if self.current_token.type != TT_EQUALS:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end, "Expected '='"
                ))
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.binary_op(self.comp_expr, ((TT_KEYWORD, 'AND'), (TT_KEYWORD, 'OR'))))
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected 'VAR', int, float, identifier, '+', '-', or '('"
            ))
        return res.success(node)

    def binary_op(self, func_a, ops, func_b=None):
        if func_b is None:
            func_b = func_a
        res = ParseResult()
        left = res.register(func_a())
        if res.error:
            return res
        while self.current_token.type in ops or (self.current_token.type, self.current_token.value) in ops:
            op_tok = self.current_token
            res.register_advancement()
            self.advance()

            right = res.register(func_b())
            if res.error:
                return res
            left = BinaryNode(left, op_tok, right)

        return res.success(left)


# RUNTIME RESULT

class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error:
            self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


# VALUES

class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    # properties
    def set_context(self, context=None):
        self.context = context
        return self

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subtracted_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multiplied_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def divided_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RunTimeError(other.pos_start, other.pos_end, 'Division by zero', self.context)
            return Number(self.value / other.value).set_context(self.context), None

    def powered_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)


class Geo:
    def __init__(self, id):
        self.id = id
        self.obj = GC.global_objects.get(id)
        self.set_pos()
        self.set_context()

    def set_context(self, context=None):
        self.context = context
        return self

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def show(self):
        self.obj.show()

    def hide(self):
        self.obj.hide()


class String:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    # properties
    def set_context(self, context=None):
        self.context = context
        return self

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def added_to(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, String):
            return Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_ne(self, other):
        if isinstance(other, String):
            return Number(int(self.value != other.value)).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)


# CONTEXT

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


# SYMBOLTABLE

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        value = self.symbols.get(name, None)
        if value is None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]


global_symbol_table = SymbolTable()
global_symbol_table.set('NULL', Number(0))
global_symbol_table.set('TRUE', Number(1))
global_symbol_table.set('FALSE', Number(0))


# INTERPRETER

class Interpreter:
    def __init__(self):
        self.a = 0

    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_GeoNode(self, node, context):
        return RTResult().success(
            Geo(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_StringNode(self, node, context):
        return RTResult().success(
            String(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_token.value
        value = context.symbol_table.get(var_name)
        if not value:
            return res.failure(
                RunTimeError(node.pos_start, node.pos_end,
                             f"'{var_name}' is not defined", context))
        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_token.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res

        context.symbol_table.set(var_name, value)
        return res.success(value)

    def visit_BinaryNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        tok_type = node.token.type
        tokens_funcs = {
            TT_PLUS: left.added_to, TT_MINUS: left.subtracted_by, TT_MUL: left.multiplied_by, TT_DIV: left.divided_by,
            TT_POW: left.powered_by,
            TT_DEQUALS: left.get_comparison_eq, TT_NEQUALS: left.get_comparison_ne, TT_LT: left.get_comparison_lt,
            TT_LTE: left.get_comparison_lte, TT_GT: left.get_comparison_gt, TT_GTE: left.get_comparison_gte
        }
        if tokens_funcs.get(tok_type):
            result, error = partial(tokens_funcs[tok_type])(right)
        elif node.token.matches(TT_KEYWORD, 'AND'):
            result, error = left.anded_by(right)
        elif node.token.matches(TT_KEYWORD, 'OR'):
            result, error = left.ored_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryNode(self, node, context):
        res = RTResult()
        value = res.register(self.visit(node.node, context))
        if res.error:
            return res
        error = None
        if node.op_token.type == TT_MINUS:
            value, error = value.multiplied_by(Number(-1))
        elif node.op_token.matches(TT_KEYWORD, 'NOT'):
            value, error = value.notted()
        elif node.op_token.matches(TT_KEYWORD, 'SHOW'):
            value, error = value.show()
        elif node.op_token.matches(TT_KEYWORD, 'HIDE'):
            value, error = value.hide()

        if error:
            return res.failure(error)
        else:
            return res.success(value.set_pos(node.pos_start, node.pos_end))

    def visit_IfNode(self, node, context):
        res = RTResult()
        case_index = 0
        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error:
                return res
            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error:
                    return res
                return res.success(expr_value)
        if node.else_case:
            else_value = res.register(self.visit(node.else_case, context))
            if res.error:
                return res
            return res.success(else_value)
        return res.success(None)


    def visit_CreateNode(self, node, context):
        res = RTResult()
        properties = node.properties
        temp = {}
        if properties.pop('type', None) == 'circle':
            id = properties.pop('id', 'RANDOM_ID').token.value
            temp['id_']=id
            for property_, value in properties.items():
                valued_node = res.register(self.visit(value, context))
                if res.error:
                    return res
                temp[property_]=valued_node.value
            properties = temp
            global_objects["obj_"+properties['id_']] = GC.Geo('circle', properties['id_'], temp)
        return res.success(Geo(temp['id_']).set_context(context).set_pos(node.pos_start, node.pos_end))




# RUN

def run(file_name, text):
    # Genereate tokens
    lexer = Lexer(file_name, text)
    tokens, error = lexer.make_tokens()
    if error:
        return None, error

    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error:
        return None, ast.error
    print('ok')
    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
