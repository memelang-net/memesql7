# Memelang v7.07 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
# This script is optimized for teaching LLMs

import random, re, json
from typing import List

CMA, SPC, END, WILD, SIGIL_L2R = ',', ' ', ';', '*', '#'

TOKEN_KIND_PATTERNS = (
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'),		# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS a="John \"Jack\" Kennedy"
	('COMMENT',		r'//[^\n]*'),
	('STATE_SEP',	r';'),
	('PAIR_SEP',	r'\s+'), 					# COLLAPSES TO ONE WHITESPACE
	('DAT_SEP',		r','),						# OR LIST
	('OPR',			r'!=|>=|<=|[=><!]'),
	('WILD',		re.escape(WILD)),
	('IDENT',		r'[A-Za-z][A-Za-z0-9_]*'),	# ALPHANUMERIC IDENTIFIERS MAY BE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('VAR',			rf'{SIGIL_L2R}[1-9]\d*'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

DAT_KINDS = {'VAR', 'WILD', 'IDENT', 'INT', 'FLOAT', 'QUOTE'}


class Token:
	def __init__(self, beg: int, kind: str, lexeme: str):
		self.beg = beg
		self.kind = kind
		self.lexeme = lexeme

		if kind == 'QUOTE':		self.datum = json.loads(lexeme)
		elif kind == 'FLOAT':	self.datum = float(lexeme)
		elif kind == 'INT':		self.datum = int(lexeme)
		else:					self.datum = str(lexeme)

	def __iter__(self): yield from (self.beg, self.kind, self.lexeme)
	def __str__(self)->str: return self.lexeme
	__repr__ = __str__


# OPR_TOKEN DAT_TOKEN {, DAT_TOKEN}
class Comp:
	def __init__(self, opr: Token, dat: List[Token], beg: int, end: int):
		self.opr = opr
		self.dat = dat
		self.beg = beg
		self.end = end

	@classmethod
	def from_tokens(cls, tokens: List[Token], i: int, implicit_opr: bool = False) -> 'Comp':
		# NEVER SPACES INSIDE
		beg = i
		n = len(tokens)
		if i >= n: raise SyntaxError("E_EOF")
		
		# 1. OPR
		if tokens[i].kind == 'OPR': opr = tokens[i]; i+=1
		elif implicit_opr: opr = Token(i, 'OPR', '')
		else: raise SyntaxError(f'E_OPR')

		# 2. SET = TOKEN {, TOKEN}
		# NEVER QUOTE WRAP WHOLE LIST
		dat: List[Token] = []
		while i < n:
			# 2b. TOKEN
			if i >= n or tokens[i].kind not in DAT_KINDS: raise SyntaxError(f'E_LIST')
			dat.append(tokens[i])
			i += 1

			# 2c. COMMA BEFORE ANOTHER OPTIONAL TOKEN
			if i < n and tokens[i].kind == 'DAT_SEP':
				# NEVER SPACES INSIDE
				# NEVER WILDCARD IN LIST
				if dat[-1].kind=='WILD': raise SyntaxError(f'E_WILD_LIST')
				i += 1
			else: break

		return cls(opr, dat, beg, i)

	# =DAT1,DAT2
	def __str__(self) -> str: return str(self.opr) + CMA.join(map(str, self.dat))
	__repr__ = __str__


# KEY_COMP VAL_COMP
class Pair:
	def __init__(self, keycomp: Comp, valcomp: Comp, beg:int, end:int):
		self.key = keycomp
		self.val = valcomp
		self.beg = beg
		self.end = end

	@classmethod
	def from_tokens(cls, tokens: List[Token], i: int) -> 'Pair':
		# NEVER SPACES INSIDE
		beg = i
		n = len(tokens)
		pairs = {}

		# KEY OPERATOR IS: ALMOST ALWAYS '' (EMPTY), SOMETIMES '!' (NOT)
		# VAL STRING OPERATOR IS: OFTEN '=', SOMETIMES '!='
		# VAL NUMERIC OPERATORS ARE: '=', '!=', '>', '<', '>=', '<='
		for slot in ('KEY', 'VAL'):
			# NEVER SPACES INSIDE
			pairs[slot]=Comp.from_tokens(tokens, i, (slot == 'KEY'))
			i = pairs[slot].end

		return cls(pairs['KEY'], pairs['VAL'], beg, i)

	# !KEY1,KEY2=VAL1,VAL2
	def __iter__(self): yield from (self.key, self.val, self.beg, self.end)
	def __str__(self) -> str: return str(self.key) + str(self.val)
	__repr__ = __str__


# Pair {SPC Pair}
class Clause:
	def __init__(self):
		self.pairs: List[Pair] = []
		self.beg = 0
		self.end = 0

	def add(self, pairs: Pair):
		self.pairs.append(pairs)

	@classmethod
	def from_tokens(cls, tokens: List[Token], i: int) -> "Clause":
		clause = cls()
		clause.beg = i
		clause.end = i
		n = len(tokens)
		pair: Pair

		while i < n and tokens[i].kind not in {'STATE_SEP'}:
			
			# SKIP SEP BETWEEN PAIRS
			if tokens[i].kind == 'PAIR_SEP': i += 1

			# PROCESS PAIR
			else:
				pair = Pair.from_tokens(tokens, i)

				i = pair.end
				clause.end = i
				clause.add(pair)

				# ACCUMULATE ONE-CHAR LITERAL KEYSETS
				# BREAK ON THE FIRST MULTI-CHAR OR NON-LITERAL KEYSET
				if pair.key.opr.lexeme!='' or len(pair.key.dat)>1 or len(pair.key.dat[0].lexeme)>1: break

		return clause

	def __str__(self) -> str: return SPC.join(map(str,self.pairs))
	__repr__ = __str__


# Clause {SPC Clause} ;
class Statement:
	def __init__(self, clauses: List[Clause], beg:int, end: int):
		self.clauses: List[Clause] = clauses
		self.beg = beg
		self.end = end

	@classmethod
	def from_tokens(cls, tokens: List[Token], i: int) -> "Statement":
		n = len(tokens)
		beg = i
		clauses: List[Clause] = []

		while i < n and tokens[i].kind != 'STATE_SEP':
			clause = Clause.from_tokens(tokens, i)
			i = clause.end
			if clause.pairs: clauses.append(clause)
	
		return cls(clauses, beg, min(i + 1, n))

	def __str__(self) -> str: return SPC.join(map(str, self.clauses))
	__repr__ = __str__


class Parser:
	def __init__(self, source: str):
		self.source = source
		self.statements: List[Statement] = []

		tokens = self.parse_token(source)
		n = len(tokens)		
		i = 0

		while i < n:
			statement = Statement.from_tokens(tokens, i)
			i = statement.end
			if statement.clauses: self.statements.append(statement)

	@staticmethod
	def parse_token(src: str) -> List[Token]:
		tokens: List[Token] = []
		for m in MASTER_PATTERN.finditer(src):
			kind = m.lastgroup
			text = m.group()
			if kind == 'COMMENT': continue
			if kind == 'MISMATCH': raise SyntaxError(f"Unexpected char {text!r} at {m.start()}")
			tokens.append(Token(m.start(), kind, text))
		return tokens

	def __str__(self) -> str: return END.join(map(str, self.statements))
	__repr__ = __str__
