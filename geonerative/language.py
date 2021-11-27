import geonerative.Geometric_Category as GC
import string
from functools import partial
from geonerative.token_names import *

global_objects = {}


# TOKENS


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
                                '^': TT_POW, ',': TT_ANDSYMBOL, '#': TT_GEO}
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
            elif self.current_char == '"':
                tokens.append(self.make_str())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

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
        tok_type = TT_IDENTIFIER
        if id_str in KEYWORDS:
            tok_type = TT_KEYWORD
            for tt, kws in KW_LISTS.items():
                if id_str in kws:
                    tok_type = tt
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
    def __init__(self, id_node):
        self.id_node = id_node
        self.pos_start = self.id_node.pos_start
        self.pos_end = self.id_node.pos_end

    def __repr__(self):
        return f'(GEO,{self.token})'


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


class GeoActionNode:
    def __init__(self, action, properties, pos_end):
        self.action = action
        self.properties = properties
        self.pos_start = self.action.pos_start
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
            geo_expr = res.register(self.expr())
            if res.error:
                return res
            return res.success(GeoNode(geo_expr))

        # elif token.type == TT_GEO:
        #     res.register_advancement()
        #     self.advance()
        #     geo_expr = res.register(self.expr())
        #     if res.error:
        #         return res
        #     return res.success(GeoNode(geo_expr))
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

        elif token.type == TT_KWS_LOGICAL:
            logical_expr = res.register(self.logical_expr())
            if res.error:
                return res
            return res.success(logical_expr)

        elif token.type == TT_KWS_GEO_ACTIONS:
            action_expr = res.register(self.geo_action_expr())
            if res.error:
                return res
            return res.success(action_expr)

        return res.failure(InvalidSyntaxError(
            token.pos_start, token.pos_end,
            "Expected value, identifier, action or '('"
        ))

    def logical_expr(self):
        res = ParseResult()
        action = self.current_token
        res.register_advancement()
        self.advance()
        if action.matches(TT_KWS_LOGICAL, KW_IF):
            cases = []
            else_case = None
            condition = res.register(self.expr())
            if res.error:
                return res
            if not self.current_token.matches(TT_KWS_LOGICAL, KW_THEN):
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    f"Expected '{KW_THEN}'"
                ))
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            cases.append((condition, expr))
            while self.current_token.matches(TT_KWS_LOGICAL, KW_ELIF):
                res.register_advancement()
                self.advance()
                condition = res.register(self.expr())
                if res.error:
                    return res
                if not self.current_token.matches(TT_KWS_LOGICAL, KW_THEN):
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        f"Expected '{KW_THEN}'"
                    ))
                res.register_advancement()
                self.advance()
                expr = res.register(self.expr())
                cases.append((condition, expr))
            if self.current_token.matches(TT_KWS_LOGICAL, KW_ELSE):
                res.register_advancement()
                self.advance()
                else_case = res.register(self.expr())
                if res.error:
                    return res
            return res.success(IfNode(cases, else_case))
        elif action.matches(TT_KWS_LOGICAL, KW_FOR):
            pass

        elif action.matches(TT_KWS_LOGICAL, KW_WHILE):
            pass
        else:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected '{KW_IF}'"
            ))

    def geo_action_expr(self):
        res = ParseResult()
        properties = {}
        action = self.current_token
        res.register_advancement()
        self.advance()
        if action.value == KW_CREATE:
            if self.current_token.type == TT_KWS_GEO_OBJECTS:
                properties['type'] = self.current_token.value
                res.register_advancement()
                self.advance()
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    f"Expected geometric type"
                ))
        geo_expr = res.register(self.expr())
        if res.error:
            return res
        properties['geo'] = geo_expr

        value = geo_expr
        if self.current_token.matches(TT_KWS_GEOMETRIC, KW_WITH):
            properties_counter = 0
            while properties_counter == 0 or self.current_token.type == TT_ANDSYMBOL:
                properties_counter += 1
                res.register_advancement()
                self.advance()
                if self.current_token.type == TT_KWS_GEO_PROPERTIES:
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
        return res.success(GeoActionNode(action, properties, value.pos_end))

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
        if self.current_token.matches(TT_KWS_LOGICAL, 'NOT'):
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

        node = res.register(self.binary_op(self.comp_expr, ((TT_KWS_LOGICAL, KW_AND), (TT_KWS_LOGICAL, KW_OR))))
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
        self.type = TT_INT if int(value) == value else TT_FLOAT
        self.set_pos()
        self.set_context()
        self.tokens_funcs = {
            TT_PLUS: self.added_to, TT_MINUS: self.subtracted_by, TT_MUL: self.multiplied_by,
            TT_DIV: self.divided_by,
            TT_POW: self.powered_by,
            TT_DEQUALS: self.get_comparison_eq, TT_NEQUALS: self.get_comparison_ne,
            TT_LT: self.get_comparison_lt,
            TT_LTE: self.get_comparison_lte, TT_GT: self.get_comparison_gt, TT_GTE: self.get_comparison_gte
        }

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
        return None, RunTimeError(self.pos_start, other.pos_end, f"cannot add Number to {type(other)}", self.context)

    def subtracted_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        return None, RunTimeError(self.pos_start, other.pos_end, f"cannot subtract Number by {type(other)}",
                                  self.context)

    def multiplied_by(self, other):
        if isinstance(other, Number):
                return Number(self.value * other.value).set_context(self.context), None
        elif isinstance(other, String):
                return String(self.value * other.value).set_context(self.context), None
        elif isinstance(other, List):
                return List(self.value * other.value).set_context(self.context), None
        return None, RunTimeError(self.pos_start, other.pos_end, f"cannot multiply Number to {type(other)}",
                                  self.context)

    def divided_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RunTimeError(other.pos_start, other.pos_end, 'Division by zero', self.context)
            return Number(self.value / other.value).set_context(self.context), None
        return None, RunTimeError(self.pos_start, other.pos_end, f"cannot divide Number by {type(other)}", self.context)

    def powered_by(self, other):
        if isinstance(other, Number):
            if self.value >= 0 or other.type == TT_INT:
                return Number(self.value ** other.value).set_context(self.context), None
            else:
                return None, RunTimeError(self.pos_start, other.pos_end,
                                          f"{other.value} cannot be an exponent for {self.value}", self.context)

        return None, RunTimeError(self.pos_start, other.pos_end, f"{type(other)} cannot be an exponent for Number",
                                  self.context)

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        return None, self.logical_rterror(other)

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        return None, self.logical_rterror(other)

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        return None, self.logical_rterror(other)

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        return None, self.logical_rterror(other)

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        return None, self.logical_rterror(other)

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        return None, self.logical_rterror(other)

    def logical_rterror(self, other):
        return RunTimeError(self.pos_start, other.pos_end, f"cannot compare Number and {type(other)}", self.context)

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
        return None, RunTimeError(self.pos_start, other.pos_end, f"cannot apply logical gates to {type(other)}",
                                  self.context)

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
        return None, RunTimeError(self.pos_start, other.pos_end, f"cannot apply logical gates to {type(other)}",
                              self.context)

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
    def __init__(self, id_):
        self.id = id_
        self.obj = global_objects.get(id_)
        self.set_pos()
        self.set_context()
        self.tokens_funcs = {
            TT_DEQUALS: self.get_comparison_eq, TT_NEQUALS: self.get_comparison_ne
        }

    def __repr__(self):
        return self.obj.__repr__() if self.obj.__repr__() == "" else f"#{str(self.id)}"

    def set_context(self, context=None):
        self.context = context
        return self

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def get_comparison_eq(self, other):
        if isinstance(other, Geo):
            return Number(int(self.id == other.id)).set_context(self.context), None
        return Number(0), None

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.id != other.id)).set_context(self.context), None
        return Number(1), None

    def copy(self):
        copy = Geo(self.id)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def show(self, properties):
        self.obj.show()

    def hide(self, properties):
        self.obj.hide()

    def create(self, properties):
        global_objects[properties['id_']] = GC.Geo(properties['type'], properties['id_'], properties)
        self.obj = global_objects[properties['id_']]
        return self

    def modify(self, properties):
        if self.obj:
            self.obj.modify(properties)
            return self
        else:
            return self.create(properties)

    def delete(self, properties):
        global_objects.pop(self.id, None)
        return self

    def action(self, action, properties):
        func = {
            KW_HIDE: self.hide,
            KW_SHOW: self.show,
            KW_CREATE: self.create,
            KW_MODIFY: self.modify,
            KW_DELETE: self.delete,
        }
        return func[action](properties)


class String:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()
        self.tokens_funcs = {
            TT_PLUS: self.added_to, TT_MUL: self.multiplied_by,
            TT_DEQUALS: self.get_comparison_eq, TT_NEQUALS: self.get_comparison_ne,
        }

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
        return None, RunTimeError(self.pos_start, other.pos_end, f"cannot add {type(other)} to String",
                                  self.context)

    def multiplied_by(self, other):
        if isinstance(other, Number) and other.value >= 0 and other.type == TT_INT:
            return String(other.value * self.value).set_context(self.context), None
        return None, RunTimeError(self.pos_start, other.pos_end, f"cannot multiply String by {type(other)}",
                                  self.context)

    def get_comparison_eq(self, other):
        if isinstance(other, String):
            return Number(int(self.value == other.value)).set_context(self.context), None
        return Number(0), None

    def get_comparison_ne(self, other):
        if isinstance(other, String):
            return Number(int(self.value != other.value)).set_context(self.context), None
        return Number(1), None

    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)


class Tuple: pass


class List: pass


class Dictionary: pass


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
        res = RTResult()
        id_ = res.register(self.visit(node.id_node, context))
        if res.error:
            return res
        elif type(id_) != String:
            return res.failure(RunTimeError(node.pos_start, node.pos_end, f"Geo id is not a string", context))
        return res.success(
            Geo(id_.value).set_context(context).set_pos(node.pos_start, node.pos_end)
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
        tokens_funcs = left.tokens_funcs
        if tokens_funcs.get(tok_type):
            result, error = tokens_funcs[tok_type](right)
        elif node.token.matches(TT_KWS_LOGICAL, KW_AND):
            result, error = left.anded_by(right)
        elif node.token.matches(TT_KWS_LOGICAL, KW_OR):
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
        elif node.op_token.matches(TT_KWS_LOGICAL, KW_NOT):
            value, error = value.notted()
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

    def visit_GeoActionNode(self, node, context):
        res = RTResult()
        properties = node.properties
        temp = {}
        geo = res.register(self.visit(properties.pop('geo', None), context))
        if res.error:
            return res
        if not isinstance(geo, Geo):
            return res.failure(RunTimeError(geo.pos_start, geo.pos_end, "Geo id should start with '#'", context))
        temp['type'] = properties.pop('type', None)
        temp['id_'] = geo.id
        for property_, value in properties.items():
            valued_node = res.register(self.visit(value, context))
            if res.error:
                return res
            temp[property_] = valued_node.value
        properties = temp
        geo.action(node.action.value, properties)
        return res.success(Geo(temp['id_']).set_context(context).set_pos(node.pos_start, node.pos_end))


# RUN

def run(file_name, text):
    # Initiate KEYWORDS
    for key in KW_LISTS:
        KEYWORDS.extend(KW_LISTS[key])

    # Present global_objects better

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
    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
