# Memelang v7.09 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
# This script is optimized for teaching LLMs

import random, re, json
from typing import List, Dict, Any

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

DAT_KINDS = {'VAR', 'WILD', 'IDENT', 'INT', 'FLOAT', 'QUOTE', 'SAME'}


class Token:
	def __init__(self, beg: int, kind: str, lexeme: str):
		self.beg: int = beg
		self.kind: str = kind
		self.lexeme: str = lexeme
		self.datum: Any

		if kind == 'QUOTE':		self.datum = json.loads(lexeme)
		elif kind == 'FLOAT':	self.datum = float(lexeme)
		elif kind == 'INT':		self.datum = int(lexeme)
		else:					self.datum = str(lexeme)

	def __iter__(self): yield from (self.beg, self.kind, self.lexeme, self.datum)
	def __str__(self)->str: return self.lexeme
	__repr__ = __str__

DEF_KEY_OPR = Token(-1, 'OPR', '') # INTENTIONALLY EMPTY


# Ordinate ::= OPR_TOKEN DAT_TOKEN {, DAT_TOKEN}
# 1-dimension
class Ordinate:
	def __init__(self, opr: Token, dat: List[Token], beg: int, end: int):
		self.opr = opr
		self.dat = dat
		self.beg = beg
		self.end = end
		self.certain = opr.lexeme in ('','=') and len(dat)==1 and dat[0].kind in ('INT','FLOAT','IDENT','QUOTE')

	@classmethod
	def from_tokens(cls, tokens: List[Token], i: int, implicit_opr: bool = False) -> 'Ordinate':
		# NEVER SPACES INSIDE
		beg = i
		n = len(tokens)
		if i >= n: raise SyntaxError("E_EOF")
		
		# 1. OPR_TOKEN
		# NEVER SPACES AROUND OPERATOR
		if tokens[i].kind == 'OPR': opr = tokens[i]; i+=1
		elif implicit_opr: opr = DEF_KEY_OPR
		else: raise SyntaxError(f'E_OPR')

		# 2. DAT_TOKEN {, DAT_TOKEN}
		# NEVER WRAP LIST IN QUOTES
		dat: List[Token] = []
		while i < n:
			# 2b. DAT_TOKEN
			if i >= n or tokens[i].kind not in DAT_KINDS: raise SyntaxError(f'E_LIST')
			dat.append(tokens[i])
			i += 1

			# 2c. COMMA BEFORE ANOTHER OPTIONAL TOKEN
			if i < n and tokens[i].kind == 'DAT_SEP':
				# NEVER SPACES AROUND COMMAS
				# NEVER WILDCARD IN LIST
				if dat[-1].kind=='WILD': raise SyntaxError(f'E_WILD_LIST')
				i += 1
			else: break

		return cls(opr, dat, beg, i)

	# RETURN FIRST DATUM
	def datum(self) -> Any: return self.dat[0].datum

	# RETURN FIRST KIND
	def kind(self) -> Any: return self.dat[0].kind

	# =DAT1,DAT2
	def __str__(self) -> str: return str(self.opr) + CMA.join(map(str, self.dat))
	__repr__ = __str__

SAME_ORDINATE = Ordinate(Token(-1, 'OPR', '='), [Token(-1, 'SAME', '_')], -1, -1)


# Coordinate ::= Ordinate Ordinate {SPC Ordinate Ordinate}
# n-dimension
class Coordinate:
	def __init__(self):
		self.ordinates: Dict[int, Ordinate] = {}
		self.beg = 0
		self.end = 0
		self.certain = None

	@classmethod
	def from_tokens(cls, tokens: List[Token], i: int, carry_forward_dimensions:List[int]|None = None, KEYMAP: Dict[str, int]|None = None) -> "Coordinate":
		coordinate = cls()
		coordinate.beg = i
		coordinate.end = i
		n = len(tokens)
		if not KEYMAP: KEYMAP = {}
		prior_val_dimension = None
		certain = True

		while i < n and tokens[i].kind not in {'STATE_SEP'}:
			
			# SKIP SEP BETWEEN PAIRS
			if tokens[i].kind == 'PAIR_SEP':
				i += 1
				continue

			# 1. KEY
			# KEY OPERATOR IS: ALMOST ALWAYS '' (EMPTY), SOMETIMES '!' (NOT)
			# NEVER SPACES INSIDE
			key_ordinate = Ordinate.from_tokens(tokens, i, True)
			val_dimension = None
			first_key = key_ordinate.datum()

			# 1a. TRY TO MAP first_key TO val_dimension
			if key_ordinate.certain:
				if isinstance(first_key, int): val_dimension = first_key
				elif first_key in KEYMAP: val_dimension = KEYMAP[first_key]

				# RESTART BEFORE HIGHER DIMENSION
				if val_dimension is not None and prior_val_dimension is not None and val_dimension >= prior_val_dimension: break
			else: certain = False
			
			# 1b. NO MAP, SPlIT KEYx=VALy into 1:KEYx, 0:VALy
			if val_dimension is None:
				coordinate.ordinates[1]=key_ordinate
				val_dimension = 0

			# 2. VAL
			# VAL STRING OPERATOR IS: OFTEN '=', SOMETIMES '!='
			# VAL NUMERIC OPERATORS ARE: '=', '!=', '>', '<', '>=', '<='
			# NEVER SPACES INSIDE
			coordinate.ordinates[val_dimension]=Ordinate.from_tokens(tokens, key_ordinate.end, False)
			i = coordinate.ordinates[val_dimension].end
			coordinate.end = i

			if not coordinate.ordinates[val_dimension].certain: certain = False

			# RESTART AFTER ZERO DIMENSION
			if val_dimension == 0: break

			prior_val_dimension = val_dimension

		# CARRY FORWARD HIGHER DIMENSIONS FROM PRIOR COORDINATE
		if coordinate.ordinates and carry_forward_dimensions:
			max_dimension = max(dimension for dimension in coordinate.ordinates)
			for dimension in carry_forward_dimensions:
				if dimension>max_dimension: coordinate.ordinates[dimension]=SAME_ORDINATE

		coordinate.certain = certain

		return coordinate


	def verbose(self) -> str:
		out = []
		for dimension in sorted(self.ordinates, reverse=True): out.append(str(dimension) + str(self.ordinates[dimension]))
		return SPC.join(out)


	def __str__(self) -> str:
		out = []
		RASugar = 0 in self.ordinates and 1 in self.ordinates

		for dimension in sorted(self.ordinates, reverse=True):
			if RASugar and dimension<=1: continue
			if self.ordinates[dimension] is SAME_ORDINATE: continue
			out.append(str(dimension) + str(self.ordinates[dimension]))

		if RASugar: out.append(str(self.ordinates[1]) + str(self.ordinates[0]))
		return SPC.join(out)


	__repr__ = __str__


# Polyline ::= Coordinate {SPC Coordinate}
# n-group of coordinates
class Polyline:
	def __init__(self, coordinates: List[Coordinate], beg:int, end: int):
		self.coordinates: List[Coordinate] = coordinates
		self.beg = beg
		self.end = end

	@classmethod
	def from_tokens(cls, tokens: List[Token], i: int) -> "Polyline":
		n = len(tokens)
		beg = i
		coordinates: List[Coordinate] = []
		KEYMAP = {'a':0,'r':1,'b':2}

		while i < n and tokens[i].kind != 'STATE_SEP':
			carry_forward_dimensions = list(coordinates[-1].ordinates.keys()) if len(coordinates) else []
			coordinate = Coordinate.from_tokens(tokens, i, carry_forward_dimensions, KEYMAP)
			i = coordinate.end
			if coordinate.ordinates: coordinates.append(coordinate)

		return cls(coordinates, beg, min(i + 1, n))

	def member(self, wherekey:str):
		member(wherekey, self)

	def __str__(self) -> str: return SPC.join(map(str, self.coordinates))
	__repr__ = __str__


# Memelang ::= Polyline {; Polyline}
class Memelang:
	def __init__(self, source: str):
		self.source = source
		self.polylines: List[Polyline] = []

		tokens = self.parse_token(source)
		n = len(tokens)		
		i = 0

		while i < n:
			polyline = Polyline.from_tokens(tokens, i)
			i = polyline.end
			if polyline.coordinates: self.polylines.append(polyline)

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

	def member(self, wherekey:str):
		for polyline in self.polylines: member(wherekey, polyline)


	def __str__(self) -> str: return END.join(map(str, self.polylines))
	__repr__ = __str__


# Store polylines as in-memory DB
MEMEBASE = {}
M_MIN = 1 << 20
M_MAX = 1 << 63
def member(wherekey:str, polyline: Polyline, pk_dimension:int = 2):

	if not wherekey: raise ValueError("wherekey")

	if wherekey not in MEMEBASE: MEMEBASE[wherekey]=[]

	row: Dict[int, Any] = {}
	prior_row: Dict[int, Any] = {}
	pk_datum: Any = None # ROW PRIMARY KEY
	for coordinate in polyline.coordinates:

		if not coordinate.certain: raise ValueError("uncertain: "+str(coordinate))
		if not coordinate.ordinates.get(1): raise ValueError("dim-1: "+str(coordinate)) # KEY/COLUMN
		if not coordinate.ordinates.get(0): raise ValueError("dim-0: "+str(coordinate)) # VALUE

		if coordinate.ordinates.get(pk_dimension) and coordinate.ordinates[pk_dimension].kind() in ('INT','FLOAT','IDENT'):
			pk_datum = coordinate.ordinates[pk_dimension].datum() # INT IS BEST PRACTICE
		elif not pk_datum: pk_datum = random.randrange(M_MIN, M_MAX)

		row: = {}

		for dimension in coordinate.ordinates:
			if dimension==pk_dimension: row[dimension]=pk_datum
			if coordinate.ordinates[dimension] is SAME_ORDINATE: row[dimension]=prior_row[dimension][-1]
			else: row[dimension]=coordinate.ordinates[dimension].datum()

		MEMEBASE[wherekey].append(row)
		prior_row=row
