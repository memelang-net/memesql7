# Memelang v7.10 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
# This script is optimized for teaching LLMs

import random, re, json, copy
from typing import List, Dict, Any

Vector = Dict[int, Any]

CMA, SPC, END, WILD, SIGIL_L2R = ',', ' ', ';', '*', '#'

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('STAR',		r'"\*"'), # LITERAL ASTERISK, NOT WILDCARD
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'), # ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS a="John \"Jack\" Kennedy"
	('STATE_SEP',	r';'),
	('PAIR_SEP',	r'\s+'), # COLLAPSES TO ONE WHITESPACE
	('DAT_SEP',		r','), # OR LIST
	('OPR',			r'!=|>=|<=|[=><!]'),
	('WILD',		r'\*'), # WILDCARD
	('IDENT',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERIC IDENTIFIERS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('VAR',			rf'{SIGIL_L2R}[1-9]\d*'),
	('SAME',		r'_'), # VARIABLE: "SAME VALUE FROM PRIOR KEY"
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

LIT_STR_KINDS = {'IDENT', 'QUOTE', 'STAR'}
LIT_NUM_KINDS = {'INT', 'FLOAT'}
VAR_KINDS = {'VAR', 'WILD', 'SAME'}


# pre-orthotope
class Token:
	def __init__(self, kind: str, lexeme: str, source_position: int = -1):
		self.kind: str = kind
		self.lexeme: str = lexeme
		self.source_position: int = source_position
		self.datum: Any

		if kind == 'QUOTE':		self.datum = json.loads(lexeme)
		elif kind == 'FLOAT':	self.datum = float(lexeme)
		elif kind == 'INT':		self.datum = int(lexeme)
		else:					self.datum = str(lexeme)

	def __iter__(self): yield from (self.kind, self.lexeme, self.source_position, self.datum)
	def __str__(self)->str: return self.lexeme
	__repr__ = __str__

KEY_EQUALS = Token('OPR', '') # INTENTIONALLY EMPTY
VAL_EQUALS = Token('OPR', '=')


class TokenStream:
	def __init__(self, src: str):
		self.tokens: List[Token] = []
		self.length = 0
		self.i = 0
		for m in MASTER_PATTERN.finditer(src):
			kind = m.lastgroup
			text = m.group()
			if kind == 'COMMENT': continue
			if kind == 'MISMATCH': raise SyntaxError(f"Unexpected char {text!r} at {m.start()}")
			self.append(Token(kind, text, m.start()))

	def append(self, token: Token):
		self.tokens.append(token)
		self.length+=1

	def peek(self) -> Token | None:
		return self.tokens[self.i] if self.i < self.length else None

	def peek_kind(self, kinds: str|set) -> bool:
		token = self.peek()
		if isinstance(kinds, str): kinds = {kinds}
		return token and token.kind in kinds

	def next(self) -> Token:
		tok = self.peek()
		if tok is None: raise SyntaxError("E_EOF")
		self.i += 1
		return tok

	def continue_until(self, continue_until_kind:str) -> bool:
		return self.i < self.length and self.tokens[self.i].kind != continue_until_kind


# Unitope ::= OPR_TOKEN DAT_TOKEN {, DAT_TOKEN}
# 1-orthotope
class Unitope:
	def __init__(self, opr: Token, dat: List[Token]):
		self.opr = opr
		self.dat = dat
		self.certain = opr.lexeme in ('','=') and len(dat)==1 and dat[0].kind in (LIT_STR_KINDS | LIT_NUM_KINDS)

	@classmethod
	def from_tokens(cls, tokens: TokenStream, implicit_opr: bool = False) -> 'Unitope':
		# NEVER SPACES INSIDE
		
		# 1. OPR_TOKEN
		# NEVER SPACES AROUND OPERATOR
		if tokens.peek_kind('OPR'): opr = tokens.next()
		elif implicit_opr: opr = KEY_EQUALS
		else: raise SyntaxError(f'E_OPR')

		# 2. DAT_TOKEN {, DAT_TOKEN}
		# NEVER WRAP LIST IN QUOTES
		dat: List[Token] = []
		while tokens.continue_until('STATE_SEP'):
			# 2b. DAT_TOKEN
			if tokens.peek_kind(LIT_STR_KINDS | LIT_NUM_KINDS | VAR_KINDS): dat.append(tokens.next())
			else: raise SyntaxError(f'E_LIST')
			
			# 2c. COMMA BEFORE ANOTHER OPTIONAL TOKEN
			if tokens.peek_kind('DAT_SEP'):
				# NEVER SPACES AROUND COMMAS
				# NEVER WILDCARD IN LIST
				if dat[-1].kind=='WILD': raise SyntaxError(f'E_WILD_LIST')
				tokens.next()
			else: break

		if not dat: raise SyntaxError(f'E_DAT')

		# AWALYS GREATER/LESSER SINGLE DAT INT/FLOAT
		if opr.lexeme in {'>','<','>=','<='}:
			# NEVER GREATER/LESSER LIST
			if len(dat)>1: raise SyntaxError(f'E_OPR_LIST')
			# NEVER GREATER/LESSER STRNG
			if dat.kind in LIT_STR_KINDS: raise SyntaxError(f'E_OPR_KIND')

		return cls(opr, dat)

	def first_datum(self) -> Any: return self.dat[0].datum
	def first_kind(self) -> Any: return self.dat[0].kind

	# =DAT1,DAT2
	def __str__(self) -> str: return str(self.opr) + CMA.join(map(str, self.dat))
	__repr__ = __str__

SAME_ORDINATE = Unitope(VAL_EQUALS, [Token('SAME', '_')])


# Orthotope ::= Dimension Unitope {SPC Dimension Unitope}
# n-group of unitopes or n-orthotope
class Orthotope:
	def __init__(self):
		self.unitopes: Dict[int, Unitope] = {}
		self.certain = None

	@classmethod
	def from_tokens(cls, tokens: TokenStream, carry_forward_dimensions:List[int]|None = None, KEYMAP: Dict[str, int]|None = None) -> "Orthotope":
		orthotope = cls()
		if KEYMAP is None: KEYMAP = {}
		prior_val_dimension = None
		certain = True

		while tokens.continue_until('STATE_SEP'):
			token_i_reset = tokens.i
			
			# SKIP SEP BETWEEN PAIRS
			while tokens.peek_kind('PAIR_SEP'): tokens.next()

			# 1. KEY
			# KEY OPERATOR IS: ALMOST ALWAYS '' (EMPTY), SOMETIMES '!' (NOT)
			# NEVER SPACES INSIDE
			key_ordinate = Unitope.from_tokens(tokens, True)
			val_dimension = None
			first_key = key_ordinate.first_datum()

			# 1a. TRY TO MAP first_key TO val_dimension
			if key_ordinate.certain:
				if isinstance(first_key, int): val_dimension = first_key
				elif first_key in KEYMAP: val_dimension = KEYMAP[first_key]

				# RESTART BEFORE HIGHER DIMENSION
				if val_dimension is not None and prior_val_dimension is not None and val_dimension >= prior_val_dimension:
					tokens.i=token_i_reset
					break
			else: certain = False
			
			# 1b. NO MAP, SPlIT KEYx=VALy into 1:KEYx, 0:VALy
			if val_dimension is None:
				orthotope.unitopes[1]=key_ordinate
				val_dimension = 0

			# 2. VAL
			# VAL STRING OPERATOR IS: OFTEN '=', SOMETIMES '!='
			# VAL NUMERIC OPERATORS ARE: '=', '!=', '>', '<', '>=', '<='
			# NEVER SPACES INSIDE
			orthotope.unitopes[val_dimension]=Unitope.from_tokens(tokens, False)

			if not orthotope.unitopes[val_dimension].certain: certain = False

			# RESTART AFTER ZERO DIMENSION
			if val_dimension == 0: break

			prior_val_dimension = val_dimension

		# CARRY FORWARD HIGHER DIMENSIONS FROM PRIOR COORDINATE
		if orthotope.unitopes and carry_forward_dimensions:
			max_dimension = max(dimension for dimension in orthotope.unitopes)
			for dimension in carry_forward_dimensions:
				if dimension>max_dimension: orthotope.unitopes[dimension]=SAME_ORDINATE

		orthotope.certain = certain

		return orthotope


	def verbose(self) -> str:
		out = []
		for dimension in sorted(self.unitopes, reverse=True): out.append(str(dimension) + str(self.unitopes[dimension]))
		return SPC.join(out)


	def __str__(self) -> str:
		out = []
		RASugar = 0 in self.unitopes and 1 in self.unitopes

		for dimension in sorted(self.unitopes, reverse=True):
			if RASugar and dimension<=1: continue
			if self.unitopes[dimension] is SAME_ORDINATE: continue
			out.append(str(dimension) + str(self.unitopes[dimension]))

		if RASugar: out.append(str(self.unitopes[1]) + str(self.unitopes[0]))
		return SPC.join(out)

	__repr__ = __str__


# Hypertope ::= Orthotope {SPC Orthotope}
# n-group of orthotopes
class Hypertope:
	def __init__(self, input_orthotopes: List[Orthotope]):
		self.input_orthotopes: List[Orthotope] = input_orthotopes
		self.output_vectors: List[Vector] = []

	@classmethod
	def from_tokens(cls, tokens: TokenStream) -> "Hypertope":
		input_orthotopes: List[Orthotope] = []
		KEYMAP = {'a':0,'r':1,'b':2}

		while tokens.continue_until('STATE_SEP'):
			carry_forward_dimensions = list(input_orthotopes[-1].unitopes.keys()) if len(input_orthotopes) else []
			orthotope = Orthotope.from_tokens(tokens, carry_forward_dimensions, KEYMAP)
			if orthotope.unitopes: input_orthotopes.append(orthotope)

		return cls(input_orthotopes)

	def write(self):
		write(self)

	def __str__(self) -> str: return SPC.join(map(str, self.input_orthotopes))
	__repr__ = __str__


# Memelang ::= Hypertope {; Hypertope}
# n-group of hypertopes
class Memelang:
	def __init__(self, source: str):
		self.source = source
		self.hypertopes: List[Hypertope] = []

		tokens = TokenStream(source)

		while tokens.peek():
			while tokens.peek_kind('STATE_SEP'): tokens.next()
			hypertope = Hypertope.from_tokens(tokens)
			if hypertope.input_orthotopes: self.hypertopes.append(hypertope)

	def write(self):
		for hypertope in self.hypertopes: write(hypertope)


	def __str__(self) -> str: return END.join(map(str, self.hypertopes))
	__repr__ = __str__


# Store memes as in-memory DB
M_MIN = 1 << 20
M_MAX = 1 << 63
def write(hypertope: Hypertope, pk_dimension:int = 2):

	for i, orthotope in enumerate(hypertope.input_orthotopes):

		if not orthotope.certain: raise ValueError(f'E_UNCERT {orthotope}')
		if not orthotope.unitopes.get(1): raise ValueError(f'E_UNCERT {orthotope}') # KEY/COLUMN
		if not orthotope.unitopes.get(0): raise ValueError(f'E_UNCERT {orthotope}') # VALUE

		for dimension in orthotope.unitopes:
			if orthotope.unitopes[dimension] is SAME_ORDINATE:
				if i==0: raise SyntaxError('E_ZERO_SAME')
				orthotope.unitopes[dimension]=hypertope.input_orthotopes[i-1].unitopes[dimension]

		if not orthotope.unitopes.get(pk_dimension) or orthotope.unitopes[pk_dimension].first_kind() not in ('INT','FLOAT','IDENT','SAME'):
			orthotope.unitopes[pk_dimension] =  Unitope(VAL_EQUALS, [Token('INT', str(random.randrange(M_MIN, M_MAX)))])


def query(query_hypertope: Hypertope, data_hypertope: Hypertope) -> List[Hypertope]:

	output_hypertopes: List[Hypertope] = []

	def dfs(idx:int, output_orthotopes:list[Orthotope]):
		if idx == len(query_hypertope.input_orthotopes):
			output_hypertopes.append(Hypertope(copy.deepcopy(output_orthotopes)))
			return

		query_orthotope = query_hypertope.input_orthotopes[idx]
		for data_orthotope in data_hypertope.input_orthotopes:
			if compare(query_orthotope, data_orthotope):
				output_orthotopes.append(copy.deepcopy(data_orthotope))
				dfs(idx + 1, output_orthotopes)
				output_orthotopes.pop()

	dfs(0, [])

	return output_hypertopes


def compare(query_orthotope: Orthotope, data_orthotope: Orthotope) -> bool:
	for dimension, query_unitope in query_orthotope.unitopes.items():
		data_unitope = data_orthotope.unitopes.get(dimension)
		if data_unitope is None: return False

		data_datum = data_unitope.first_datum()
		query_datum = query_unitope.first_datum()
		query_opr = query_unitope.opr.lexeme

		if query_unitope.first_kind() == 'WILD': success = True
	
		elif query_opr in ("", "="): success = data_datum in {token.datum for token in query_unitope.dat}
		elif query_opr in ("!", "!="): success = data_datum not in {token.datum for token in query_unitope.dat}

		else:
			if data_unitope.first_kind() not in LIT_NUM_KINDS: success = False
			elif query_opr == '>': success = data_datum > query_datum
			elif query_opr == '>=': success = data_datum >= query_datum
			elif query_opr == '<': success = data_datum < query_datum
			elif query_opr == '<=': success = data_datum <= query_datum
			else: raise SyntaxError('E_CMP')

		if not success: return False

	return True