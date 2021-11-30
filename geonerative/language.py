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
        single_symbol_tokens = {
            '+': TT_PLUS, '*': TT_MUL, '/': TT_DIV, '(': TT_LPAREN, ')': TT_RPAREN,
            '^': TT_POW, ',': TT_COMMA, '#': TT_GEO, '$': TT_LISTENER,
            '[': TT_LBRACK, ']': TT_RBRACK, '{': TT_LCPAREN, '}': TT_RCPAREN
        }
        multi_symbol_tokens = {
            '=': self.make_equals, '<': self.make_less_than, '>': self.make_greater_than, '-': self.make_minus,
        }
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
            elif self.current_char in DIGITS + '.':
                tokens.append(self.make_number())
            elif multi_symbol_tokens.get(self.current_char):
                tokens.append(multi_symbol_tokens[self.current_char]())

            elif self.current_char in ('"', "'"):
                new_tokens, error = self.make_str()
                if error:
                    return [], error
                tokens.extend(new_tokens)
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_str(self):
        str_ = ''
        bracket = self.current_char
        pos_start = self.pos.copy()
        self.advance()
        while self.current_char and self.current_char != bracket:
            str_ += self.current_char
            self.advance()
        if self.current_char == '"':
            self.advance()
            return [Token(TT_STR, str_, pos_start, self.pos)], None
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

    def make_minus(self):
        token_type = TT_MINUS
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '>':
            self.advance()
            token_type = TT_ARROW
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
    def __init__(self, var_name_token, listening_to={}):
        self.var_name_token = var_name_token
        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.var_name_token.pos_end
        self.listening_to = listening_to


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


class ListNode:
    def __init__(self, elements):
        pass



class FuncDefNode:
    def __init__(self, var_name_token, arg_name_tokens, body_node):
        self.var_name_token = var_name_token
        self.arg_name_tokens = arg_name_tokens
        self.body_node = body_node
        if self.var_name_token:
            self.pos_start = self.var_name_token.pos_start
        elif len(self.arg_name_tokens) > 0:
            self.pos_start = self.arg_name_tokens[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start
        self.pos_end = self.body_node.pos_end


class CallFuncNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes
        self.pos_start = self.node_to_call.pos_start
        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end


class MutedNode:
    def __init__(self, value):
        self.value = value
        self.pos_start = self.value.pos_start
        self.pos_end = self.value.pos_end


class ListenerNode:
    def __init__(self, var_name_token):
        self.var_name_token = var_name_token
        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.var_name_token.pos_end


# PARSER

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0

    def register_advancement(self):
        self.advance_count += 1

    def register(self, res):
        if not res:
            return "a"
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

    def call_func(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error:
            return res
        if self.current_token.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            arg_nodes = []
            if self.current_token == TT_RPAREN:
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        f"Expected ')', 'VAR', '{KW_IF}' int, float, identifier, '+', '-', or '('"
                    ))
                while self.current_token.type == TT_COMMA:
                    res.register_advancement()
                    self.advance()
                    arg_nodes.append(res.register(self.expr()))
                    if res.error:
                        return res
                if self.current_token.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        f"Expected ',' or ')'"
                    ))
                res.register_advancement()
                self.advance()
            return res.success(CallFuncNode(atom, arg_nodes))
        return res.success(atom)

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
        elif token.type == TT_LISTENER:
            res.register_advancement()
            self.advance()
            token = self.current_token
            if token.type == TT_IDENTIFIER:
                res.register_advancement()
                self.advance()
                return res.success(ListenerNode(token))
            return res.failure(InvalidSyntaxError(
                token.pos_start, self.current_token.pos_end, "Can only listen to identifiers"
            ))
        elif token.type == TT_STR:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(token))
        elif token.type == TT_IDENTIFIER:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(token))
        elif token.matches(TT_KWS_LOGICAL, KW_FUNCTION):
            func_def = res.register(self.func_def())
            if res.error:
                return res
            return res.success(func_def)

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

    def func_def(self):
        res = ParseResult()
        if not self.current_token.matches(TT_KWS_LOGICAL, KW_FUNCTION):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected '{KW_FUNCTION}'"
            ))
        res.register_advancement()
        self.advance()
        if self.current_token.type == TT_IDENTIFIER:
            var_name_token = self.current_token
            res.register_advancement()
            self.advance()
            if self.current_token.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    f"Expected '('"
                ))
        else:
            var_name_token = None
            if self.current_token.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    f"Expected '('"
                ))
        res.register_advancement()
        self.advance()
        arg_name_tokens = []
        if self.current_token.type == TT_IDENTIFIER:
            arg_name_tokens.append(self.current_token)
            res.register_advancement()
            self.advance()
            while self.current_token.type == TT_COMMA:
                res.register_advancement()
                self.advance()
                if self.current_token.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        f"Expected identifier"
                    ))
                arg_name_tokens.append(self.current_token)
                res.register_advancement()
                self.advance()
            if self.current_token.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    f"Expected ',' or ')'"
                ))
        else:
            if self.current_token.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    f"Expected identifier or ')'"
                ))
        res.register_advancement()
        self.advance()
        if self.current_token.type != TT_ARROW:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected '->'"
            ))
        res.register_advancement()
        self.advance()
        node_to_return = res.register(self.expr())
        if res.error:
            return res
        return res.success(FuncDefNode(
            var_name_token, arg_name_tokens, node_to_return
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
            while properties_counter == 0 or self.current_token.type == TT_COMMA:
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
        return self.binary_op(self.call_func, (TT_POW,), self.factor)

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
                f"Expected 'VAR', '{KW_IF}' int, float, identifier, '+', '-', or '('"
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
        self.node = None
        self.listening_to = set()

    def register(self, res):
        if res.error:
            self.error = res.error
        self.listening_to.update(res.listening_to)
        return res.value, res.node

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self

    def set_node(self, node):
        self.node = node
        return self

    def listen(self, listening_to):
        self.listening_to.update(listening_to)
        return self


# VALUES

class Value:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()
        self.tokens_funcs = {
            TT_DEQUALS: self.get_comparison_eq, TT_NEQUALS: self.get_comparison_ne
        }

    def set_context(self, context=None):
        self.context = context
        return self

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def get_comparison_eq(self, other):
        if isinstance(other, type(self)):
            return Number(int(self.value == other.value)).set_context(self.context), None
        return Number(0), None

    def get_comparison_ne(self, other):
        if isinstance(other, type(self)):
            return Number(int(self.value != other.value)).set_context(self.context), None
        return Number(1), None


class Number(Value):
    def __init__(self, value):
        super().__init__(value)
        self.type = TT_INT if int(value) == value else TT_FLOAT
        self.tokens_funcs.update({
            TT_PLUS: self.added_to, TT_MINUS: self.subtracted_by, TT_MUL: self.multiplied_by,
            TT_DIV: self.divided_by,
            TT_POW: self.powered_by,
            TT_DEQUALS: self.get_comparison_eq, TT_NEQUALS: self.get_comparison_ne,
            TT_LT: self.get_comparison_lt,
            TT_LTE: self.get_comparison_lte, TT_GT: self.get_comparison_gt, TT_GTE: self.get_comparison_gte
        })

    # properties

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


class Geo(Value):
    def __init__(self, id_):
        super().__init__(id_)
        self.obj = global_objects.get(id_)

    def __repr__(self):
        return self.obj.__repr__() if self.obj.__repr__() == "" else f"#{str(self.value)}"

    def copy(self):
        copy = Geo(self.value)
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
        global_objects.pop(self.value, None)
        del new_listeners[self.value]
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


class String(Value):
    def __init__(self, value):
        if not isinstance(value, str):
            print(type(value), value)
        super().__init__(value)
        self.tokens_funcs.update({
            TT_PLUS: self.added_to, TT_MUL: self.multiplied_by,
            TT_DEQUALS: self.get_comparison_eq, TT_NEQUALS: self.get_comparison_ne,
        })

    # properties

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


class Function(Value):
    def __init__(self, name, body_node, arg_names):
        super().__init__(name)
        self.name = name or "<anonymous>"
        self.body_node = body_node
        self.arg_names = arg_names

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)

        if len(args) > len(self.arg_names):
            return res.failure(RunTimeError(
                self.pos_start, self.pos_end,
                f"{len(args) - len(self.arg_names)} too many args passed into '{self.name}",
                self.context
            ))
        if len(args) < len(self.arg_names):
            return res.failure(RunTimeError(
                self.pos_start, self.pos_end,
                f"{len(self.arg_names) - len(args)} too few args passed into '{self.name}",
                self.context
            ))
        for i in range(len(args)):
            arg_name = self.arg_names[i]
            arg_value = args[i]
            arg_value.set_context(new_context)
            new_context.symbol_table.set(arg_name, arg_value)
        value, node = res.register(interpreter.visit(self.body_node, new_context))
        if res.error:
            return res
        return res.success(value).set_node(node)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"


class BuiltInFunction(Value):
    def __init__(self, name, action, arg_names):
        super().__init__(name)
        self.name = name
        self.action = action
        self.arg_names = arg_names

    def execute(self, args):
        res = RTResult()
        tuple = ()
        if len(args) > len(self.arg_names):
            return res.failure(RunTimeError(
                self.pos_start, self.pos_end,
                f"{len(args) - len(self.arg_names)} too many args passed into '{self.name}",
                self.context
            ))
        if len(args) < len(self.arg_names):
            return res.failure(RunTimeError(
                self.pos_start, self.pos_end,
                f"{len(self.arg_names) - len(args)} too few args passed into '{self.name}",
                self.context
            ))
        if res.error:
            return res
        result, res_node = res.register(self.action(args))
        if res.error:
            return res
        else:
            return res.success(result).set_node(res_node)

    def copy(self):
        copy = BuiltInFunction(self.name, self.action, self.arg_names)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"


class Tuple(Value): pass


class List(Value): pass


class Dictionary(Value): pass


# LISTENERS

new_listeners = {}
listeners = {}
current_listeners = {}


class Listener(Value):
    def __init__(self, value):
        super().__init__(value)


# CONTEXT

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


# SYMBOLTABLE

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

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


# BUILTINFUNCTIONS

def ACTION_TO_STR(args):
    arg = args[0]
    res = RTResult()
    if isinstance(arg, String) or isinstance(arg, Number):
        return res.success(String(str(arg.value)))
    return res.failure(RunTimeError(arg.pos_start, arg.pos_end, f"Cannot turn {type(arg)} to String", arg.context))


def ACTION_TO_INT(args):
    arg = args[0]
    res = RTResult()
    if isinstance(arg, String) or isinstance(arg, Number):
        return res.success(String(int(float(arg.value))))
    return res.failure(RunTimeError(arg.pos_start, arg.pos_end, f"Cannot turn {type(arg)} to Int", arg.context))


def ACTION_TO_FLOAT(args):
    arg = args[0]
    res = RTResult()
    if isinstance(arg, String) or isinstance(arg, Number):
        return res.success(String(float(arg.value)))
    return res.failure(RunTimeError(arg.pos_start, arg.pos_end, f"Cannot turn {type(arg)} to Float", arg.context))


global_symbol_table.set(FUNC_TO_FLOAT, BuiltInFunction(FUNC_TO_FLOAT, ACTION_TO_FLOAT, ['s']))
global_symbol_table.set(FUNC_TO_STR, BuiltInFunction(FUNC_TO_STR, ACTION_TO_STR, ['s']))
global_symbol_table.set(FUNC_TO_INT, BuiltInFunction(FUNC_TO_INT, ACTION_TO_INT, ['s']))


# INTERPRETER

class Interpreter:
    def __init__(self):
        pass

    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        value = method(node, context)
        return value

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        ).set_node(MutedNode(Number(node.token.value)))

    def visit_GeoNode(self, node, context):
        res = RTResult()
        id_, id_node = res.register(self.visit(node.id_node, context))
        if res.error:
            return res
        if res.listening_to:
            res.failure(RunTimeError(node.pos_start, node.pos_end, f"Geo id cannot listen", context))
        elif type(id_) != String:
            return res.failure(RunTimeError(node.pos_start, node.pos_end, f"Geo id is not a string", context))
        result = Geo(id_.value)
        return res.success(
            result.set_context(context).set_pos(node.pos_start, node.pos_end)
        ).set_node(MutedNode(result))

    def visit_StringNode(self, node, context):
        return RTResult().success(
            String(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        ).set_node(MutedNode(String(node.token.value)))

    def visit_MutedNode(self, node, context):
        return RTResult().success(
            node.value.copy().set_context(context).set_pos(node.pos_start, node.pos_end)
        ).set_node(node)

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_token.value
        value = context.symbol_table.get(var_name)
        if not value:
            return res.failure(
                RunTimeError(node.pos_start, node.pos_end,
                             f"'{var_name}' is not defined", context))
        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value).set_node(MutedNode(value))

    def visit_ListenerNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_token.value
        value = context.symbol_table.get(var_name)
        if not value:
            return res.failure(
                RunTimeError(node.pos_start, node.pos_end,
                             f"'{var_name}' is not defined", context))
        value = value.copy().set_pos(node.pos_start, node.pos_end)
        listening_to = {var_name}
        if listeners.get(var_name, None):
            listening_to = listening_to | listeners[var_name].listening_to
        return res.success(value).set_node(node).listen(listening_to)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_token.value
        value, value_node = res.register(self.visit(node.value_node, context))
        if res.error:
            return res
        res_node = MutedNode(value)
        if res.listening_to:
            if var_name in res.listening_to:
                return res.failure(
                    RunTimeError(node.pos_start, node.pos_end, f"{var_name} cannot listen to itself", context))
            new_listeners[var_name] = RTResult().set_node(VarAssignNode(node.var_name_token, value_node)).listen(
                res.listening_to)
            res_node = VarAccessNode(node.var_name_token)
        listeners.pop(var_name, None)
        context.symbol_table.set(var_name, value)
        return res.success(value).set_node(res_node)

    def visit_BinaryNode(self, node, context):
        res = RTResult()
        left, left_node = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right, right_node = res.register(self.visit(node.right_node, context))
        if res.error:
            return res
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
        if res.listening_to:
            res_node = BinaryNode(left_node, node.token, right_node)
        else:
            res_node = MutedNode(result)
        return res.success(result.set_pos(node.pos_start, node.pos_end)).set_node(res_node)

    def visit_UnaryNode(self, node, context):
        res = RTResult()

        value, value_node = res.register(self.visit(node.node, context))
        if res.error:
            return res
        error = None
        if node.op_token.type == TT_MINUS:
            value, error = value.multiplied_by(Number(-1))
        elif node.op_token.matches(TT_KWS_LOGICAL, KW_NOT):
            value, error = value.notted()
        if error:
            return res.failure(error)
        res_node = node if res.listening_to else MutedNode(value)
        return res.success(value.set_pos(node.pos_start, node.pos_end)).set_node(res_node)

    def visit_IfNode(self, node, context):
        res = RTResult()
        case_index = 0
        cases = []
        else_case = None
        final_expr = None
        listener = False
        for condition, expr in node.cases:
            condition_value, condition_node = res.register(self.visit(condition, context))
            if res.error:
                return res
            expr_value, expr_node = res.register(self.visit(expr, context))
            if res.error:
                return res
            if not final_expr and expr_value:
                final_expr = expr_value

            cases.append((condition_node, expr_node))
        if node.else_case:
            else_value, else_node = res.register(self.visit(node.else_case, context))
            if res.error:
                return res
            if not final_expr:
                final_expr = else_value
        else:
            else_node = None
        if res.listening_to:
            res_node = IfNode(cases, else_node)
        else:
            res_node = MutedNode(final_expr)
        return res.success(final_expr).set_node(res_node)

    def visit_GeoActionNode(self, node, context):
        res = RTResult()
        properties = node.properties.copy()
        temp = {}
        node_properties = {}
        geo_node = properties.pop('geo', None)
        geo, geo_node = res.register(self.visit(geo_node, context))
        if res.error:
            return res
        if not isinstance(geo, Geo):
            return res.failure(RunTimeError(geo.pos_start, geo.pos_end, "Geo id should start with '#'", context))
        if (not geo.obj) and not node.action.matches(TT_KWS_GEO_ACTIONS, KW_CREATE):
            return res.failure(RunTimeError(node.pos_start, geo.pos_end,
                                            f"Cannot {node.action.value} nonexistent Geo object #{geo.value}", context))

        node_properties['geo'] = geo_node
        temp['type'] = properties.pop('type', None)
        node_properties['type'] = temp['type']
        temp['id_'] = geo.value
        for property_, value in properties.items():
            valued, node_value = res.register(self.visit(value, context))
            if res.error:
                return res
            temp[property_] = valued.value
            node_properties[property_] = node_value
        properties = temp
        geo.action(node.action.value, properties)
        if f"#{geo.value}" in res.listening_to:
            return res.failure(RunTimeError(node.pos_start, geo.pos_end,
                                            f"#{geo.value} cannot listen to itself", context))
        if res.listening_to:
            res_node = GeoActionNode(node.action, node_properties, value.pos_end)
            new_listeners[f"#{properties['id_']}"] = RTResult().set_node(res_node).listen(res.listening_to)
        listeners.pop(f"#{properties['id_']}", None)
        return res.success(Geo(properties['id_']).set_context(context).set_pos(node.pos_start, node.pos_end)).set_node(
            res_node)

    def visit_FuncDefNode(self, node, context):
        res = RTResult()
        func_name = node.var_name_token.value if node.var_name_token else None
        if func_name in BUILT_IN_FUNCS:
            return res.failure(RunTimeError(node.pos_start, node.pos_end,
                                            f"cannot overwrite a built-in function", context))
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_tokens]
        func_value = Function(func_name, body_node, arg_names).set_context(context).set_pos(node.pos_start,
                                                                                            node.pos_end)
        if node.var_name_token:
            context.symbol_table.set(func_name, func_value)

        return res.success(func_value)

    def visit_CallFuncNode(self, node, context):
        res = RTResult()
        args = []
        listener = False
        value_to_call, value_node = res.register(self.visit(node.node_to_call, context))
        if res.error:
            return res
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)
        arg_nodes = []
        for arg in node.arg_nodes:
            arg_value, arg_node = res.register(self.visit(arg, context))
            args.append(arg_value)
            if res.error:
                return res
            arg_nodes.append(arg_node)
        if isinstance(value_to_call, Function) or isinstance(value_to_call, BuiltInFunction):
            return_value, return_node = res.register(value_to_call.execute(args))
            if res.error:
                return res
            if res.listening_to:
                return_node = CallFuncNode(value_node, arg_nodes)
            return res.success(return_value).set_node(MutedNode(return_value))
        return res.failure(RunTimeError(node.pos_start, node.pos_end,
                                        f"{value_to_call} is not a function", context))


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
    current_listeners = listeners.copy()
    result = interpreter.visit(ast.node, context)
    for id_, res in current_listeners.items():
        interpreter.visit(res.node, context)
    listeners.update(new_listeners)
    print("C = ", global_symbol_table.get('C'))
    return result.value, result.error
