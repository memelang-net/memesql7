# Memelang v7.12 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
# This script is optimized for training LLMs

# YOU ARE IN BASIC TRAINING MODE, IGNORE NON-BASIC BRANCHES
BASIC_TRAINING_MODE = True

import random, re, json, copy
from typing import List, Dict, Any

Axis = int
NO_AXIS: Axis = -1
NAME_AXIS: Dict[str, Axis] = {
	'v':0, # VAL
	'k':1, # KEY
	'r':2, # ROW
	't':3  # TABLE
}

CMA, SPC, END, WILD, SIGIL_L2R, AXS = ',', ' ', ';', '*', '#', '\u27C2'

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('STAR',		r'"\*"'), # LITERAL ASTERISK, NOT WILDCARD, FOR TRAINING DISTINCTION
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'), # ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS a="John \"Jack\" Kennedy"
	('HYPER_SEP',	r';'),
	('PAIR_SEP',	r'\s+'),
	('DAT_SEP',		r','), # OR LIST
	('OPR',			r'!=|>=|<=|\u27C2|[=><]'),
	('WILD',		r'\*'), # WILDCARD
	('IDENT',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERIC IDENTIFIERS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('VAR',			rf'{SIGIL_L2R}[1-9]\d*'),
	('SAME',		r'_'), # VARIABLE: "SAME VALUE"
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

LIT_STR_KINDS = {'IDENT', 'QUOTE', 'STAR'}
LIT_NUM_KINDS = {'INT', 'FLOAT'}
VAR_KINDS = {'VAR', 'WILD', 'SAME'}


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

EQUALS = Token('OPR', '=')


class TokenStream:
	def __init__(self, src: str):
		self.tokens: List[Token] = []
		self.length = 0
		self.i = 0
		for m in MASTER_PATTERN.finditer(src):
			kind = m.lastgroup
			text = m.group()
			if kind == 'COMMENT': continue
			if kind == 'MISMATCH': raise SyntaxError(f'Unexpected char {text!r} at {m.start()}')
			self.append(Token(kind, text, m.start()))

	def append(self, token: Token):
		self.tokens.append(token)
		self.length+=1

	def peek(self) -> Token | None:
		return self.tokens[self.i] if self.i < self.length else None

	def peek_kind(self) -> str|None:
		token = self.peek()
		return token.kind if token else None

	def next(self) -> Token:
		tok = self.peek()
		if tok is None: raise SyntaxError('E_EOF')
		self.i += 1
		return tok

	def reset(self, i:int):
		self.i = i

	def continue_until(self, continue_until_kinds:List[str]) -> bool:
		return self.i < self.length and self.tokens[self.i].kind not in continue_until_kinds


# Point ::= OPR_TOKEN DAT_TOKEN {, DAT_TOKEN}
# single axis constraint
class Point:
	def __init__(self, opr: Token, dat: List[Token]):
		self.opr = opr
		self.dat = dat
		self.certain = opr.lexeme == '=' and len(dat)==1 and dat[0].kind in (LIT_STR_KINDS | LIT_NUM_KINDS) # CORRECT AND INTENTIONAL

	@classmethod
	def from_tokens(cls, tokens: TokenStream, elided_key_opr: bool = False) -> 'Point':
		# NEVER SPACES INSIDE
		
		# 1. OPR_TOKEN
		# NEVER SPACES AROUND OPERATOR
		if tokens.peek_kind() == 'OPR': opr = tokens.next()
		elif elided_key_opr: opr = EQUALS # EMPTY KEY_OPR MEANS 'EQUALS'
		else: raise SyntaxError(f'E_OPR')

		# 2. DAT_TOKEN {, DAT_TOKEN}
		# NEVER WRAP LIST IN QUOTES
		dat: List[Token] = []
		while tokens.continue_until(['PAIR_SEP','HYPER_SEP']):
			# 2b. DAT_TOKEN
			if tokens.peek_kind() not in (LIT_STR_KINDS | LIT_NUM_KINDS | VAR_KINDS): raise SyntaxError(f'E_LIST')
			dat.append(tokens.next())
			
			# 2c. COMMA BEFORE ANOTHER OPTIONAL TOKEN (OR SEMANTICS)
			if tokens.peek_kind() == 'DAT_SEP':
				# NEVER SPACES AROUND COMMAS
				# NEVER OPR IN LIST
				# NEVER WILDCARD IN LIST
				if dat[-1].kind=='WILD': raise SyntaxError(f'E_WILD_LIST')
				tokens.next()
			else: break

		if not dat: raise SyntaxError(f'E_DAT')

		# ALWAYS GREATER/LESSER SINGLE DAT INT/FLOAT
		if opr.lexeme in {'>','<','>=','<='}:
			# NEVER GREATER/LESSER LIST
			if len(dat)>1: raise SyntaxError(f'E_OPR_LIST')
			# NEVER GREATER/LESSER STRNG
			if dat[0].kind in LIT_STR_KINDS: raise SyntaxError(f'E_OPR_KIND')

		return cls(opr, dat)

	def first_datum(self) -> Any: return self.dat[0].datum
	def first_kind(self) -> Any: return self.dat[0].kind

	def out(self, elided_key_opr:bool=False) -> str:
		# CHANGE PEDANTIC "=KEY=VAL" TO FRIENDLY "KEY=VAL"
		opr = '' if elided_key_opr and self.opr.lexeme=='=' else self.opr.lexeme
		return opr + CMA.join(map(str, self.dat))

	# =DAT1,DAT2
	def __str__(self) -> str: return str(self.opr) + CMA.join(map(str, self.dat))
	__repr__ = __str__

SAME_ORDINATE = Point(EQUALS, [Token('SAME', '_')])


# Path ::= Point [Point] {SPC Point [Point]}
# coordinates of axis constraints
class Path:
	def __init__(self):
		self.points: Dict[Axis, Point] = {}
		self.certain = None
		self.axis_names: Dict[Axis, str] = {}

	@classmethod
	def from_tokens(cls, tokens: TokenStream, carry_forward_axes:List[Axis]|None = None, name_axis: Dict[str, Axis]|None = None) -> 'Path':
		path = cls()
		if name_axis is None: name_axis = {}
		path.axis_names = {v: k for k, v in name_axis.items()}
		prior_min_axis: Axis = NO_AXIS
		path.certain = True
		axis_map: List[int]

		# AXES DESCEND FROM LEFT-TO-RIGHT
		while tokens.continue_until(['HYPER_SEP']):
			starting_i = tokens.i
			points: List[Point] = []
			
			# SKIP PAIR_SEP
			while tokens.peek_kind() == 'PAIR_SEP': tokens.next()

			if BASIC_TRAINING_MODE:
				# 1. KEY=VAL -> KEY POINT, VAL POINT -> ONE PAIR
				# points[0] ::= [KEY_OPR] KEY_DAT {, KEY_DAT}
				# points[1] ::= VAL_OPR VAL_DAT {, VAL_DAT}
				# KEY_OPR IS ALMOST ALWAYS EMPTY (=), RARELY VAL_OPR
				# VAL_OPR STRING IS: OFTEN '=', SOMETIMES '!='
				# VAL_OPR NUMERIC ARE: '=', '!=', '>', '<', '>=', '<='
				# EXAMPLE: role,character!="Luke Skywalker",Joker
				points.append(Point.from_tokens(tokens, elided_key_opr=True)) # KEY
				points.append(Point.from_tokens(tokens, elided_key_opr=False)) # VAL
				if not points[0] or not points[1]: raise SyntaxError('E_PAIR')
				len_points=2

				# 2. ASSIGN AN AXIS TO EACH POINT

				# 2a. DEFAULT CASE (MOST COMMON)
				# KEY points[0] -> axis_map[0] = NAME_AXIS['k'] = 1
				# VAL points[1] -> axis_map[1] = NAME_AXIS['v'] = 0
				axis_map = [NAME_AXIS['k'],NAME_AXIS['v']]

				# 2b. SPECIAL META CASE: KEY MAPS TO VAL'S AXIS (RARE)
				if points[0].first_datum() in name_axis: axis_map = [NO_AXIS, name_axis[points[0].first_datum()]]

			elif not BASIC_TRAINING_MODE:

				# 1. PARSE P1>P2=P3<P4
				while tokens.continue_until(['PAIR_SEP','HYPER_SEP']):
					# NEVER SPACES BETWEEN SEQUENTIAL AXES
					points.append(Point.from_tokens(tokens, len(points)==0))

				len_points=len(points)
				if len_points<2: raise SyntaxError('E_PAIR')

				# 2. ASSIGN AN AXIS TO EACH POINT

				# 2a. DEFAULT CASE, DESCENDING TO ZERO
				axis_map = list(range(len_points - 1, -1, -1))

				# 2b. SPECIAL META CASE: KEY MAPS TO VAL'S AXIS
				if len_points==2 and points[0].first_datum() in name_axis: axis_map = [NO_AXIS, name_axis[points[0].first_datum()]]

				# 2c. FIRST KEY IS AN AXIS
				elif points[0].opr.lexeme==AXS: 
					if points[0].first_datum() in name_axis: first_axis = name_axis[points[0].first_datum()]
					elif points[0].first_kind() == 'INT': first_axis = points[0].first_datum()
					else: raise SyntaxError('E_OPR_AXS')
					if first_axis < len(points) - 1: raise SyntaxError('E_AXIS_NEG')
					axis_map = [NO_AXIS] + [first_axis - i for i in range(len(points) - 1)]


			# 3. PROCESS AXES

			# 3a. ALWAYS AXIS>=0
			if axis_map[-1]<0: raise SyntaxError('E_AXIS_NEG')

			# 3b. HIGHER AXIS DETECTED - START NEW PLANE
			if prior_min_axis>=0 and max(x for x in axis_map) >= prior_min_axis:
				tokens.reset(starting_i)
				break

			# 3c. ASSIGN AXIS TO EACH POINT
			for idx, point in enumerate(points):
				if axis_map[idx] == NO_AXIS: continue
				path.points[axis_map[idx]]=point
				if not point.certain: path.certain = False

			# 4. WHAT NEXT?

			# 4a. ZERO AXIS DETECTED - START NEW PLANE 
			# COMMON AFTER EACH KEY=VAL PAIR
			if axis_map[-1] == NAME_AXIS['v']: break

			# 4b. KEEP DESCENDING INTO AXES ON THIS PLANE
			prior_min_axis = axis_map[-1]

		# CARRY FORWARD HIGHER AXES FROM PRIOR COORDINATE
		# NEVER CARRIES PAST HYPER_SEP SEMICOLON (;)
		if path.points and carry_forward_axes:
			max_axis = max(axis for axis in path.points)
			for axis in carry_forward_axes:
				if axis>max_axis: path.points[axis]=SAME_ORDINATE

		return path


	def __str__(self) -> str:
		out = []
		axis: Axis
		prior_axis: Axis = NO_AXIS

		has_key_value_pair:bool = (NAME_AXIS['k'] in self.points and NAME_AXIS['v'] in self.points)

		for axis in sorted(self.points, reverse=True):
			if self.points[axis] is SAME_ORDINATE: continue

			# KEY=VAL FRIENDLY SYNTAX
			if has_key_value_pair and axis==NAME_AXIS['k']: out.append(self.points[axis].out(True))
			elif has_key_value_pair and axis==NAME_AXIS['v']: out[-1]+=self.points[axis].out()

			# AXIS_NAME=VAL PRETTY SYNTAX
			elif axis in self.axis_names: out.append(str(self.axis_names[axis]) + self.points[axis].out())
			
			elif not BASIC_TRAINING_MODE:
				# APPEND =VAL
				if prior_axis>0 and axis==prior_axis-1: out[-1]+=self.points[axis].out()

				# AXIS=VAL PLAIN SYNTAX
				else: out.append(AXS + str(axis) + self.points[axis].out())

			prior_axis=axis

		return SPC.join(out)

	__repr__ = __str__


# Plane ::= Path {SPC Path}
# matrix of axis constraints
class Plane:
	def __init__(self, input_paths: List[Path]):
		self.input_paths: List[Path] = input_paths

	@classmethod
	def from_tokens(cls, tokens: TokenStream, name_axis: Dict[str,Axis] = NAME_AXIS) -> 'Plane':
		input_paths: List[Path] = []
		
		while tokens.continue_until(['HYPER_SEP']):
			carry_forward_axes = list(input_paths[-1].points.keys()) if input_paths else []
			path = Path.from_tokens(tokens, carry_forward_axes, name_axis)
			if path.points: input_paths.append(path)

		return cls(input_paths)

	def write(self):
		write(self)

	def __str__(self) -> str: return SPC.join(map(str, self.input_paths))
	__repr__ = __str__


# Memelang ::= Plane {; Plane}
# stack of matrices of axis constraints
class Memelang:
	def __init__(self, source: str):
		self.source = source
		self.planes: List[Plane] = []

		tokens = TokenStream(source)

		while tokens.peek():
			while tokens.peek_kind() == 'HYPER_SEP': tokens.next()
			plane = Plane.from_tokens(tokens)
			if plane.input_paths: self.planes.append(plane)

	def write(self):
		for plane in self.planes: write(plane)


	def __str__(self) -> str: return END.join(map(str, self.planes))
	__repr__ = __str__


# Persist Planes in an in-memory collection
M_MIN = 1 << 20
M_MAX = 1 << 53
def write(plane: Plane, primary_key_axis:Axis = NAME_AXIS['r']):

	for i, path in enumerate(plane.input_paths):

		if not path.certain: raise ValueError(f'E_UNCERT {path}')
		if not path.points.get(NAME_AXIS['k']): raise ValueError(f'E_UNCERT {path}') # KEY/COLUMN
		if not path.points.get(NAME_AXIS['v']): raise ValueError(f'E_UNCERT {path}') # VALUE

		for axis in path.points:
			if path.points[axis] is SAME_ORDINATE:
				if i==0: raise SyntaxError('E_ZERO_SAME')
				path.points[axis]=plane.input_paths[i-1].points[axis]

		if not path.points.get(primary_key_axis) or path.points[primary_key_axis].first_kind() not in ('INT','FLOAT','IDENT','SAME'):
			path.points[primary_key_axis] =  Point(EQUALS, [Token('INT', str(random.randrange(M_MIN, M_MAX)))])


def query(query_plane: Plane, data_plane: Plane) -> List[Plane]:

	output_planes: List[Plane] = []

	def dfs(idx:int, output_paths:list[Path]):
		if idx == len(query_plane.input_paths):
			output_planes.append(Plane(copy.deepcopy(output_paths)))
			return

		query_path = query_plane.input_paths[idx]
		for data_path in data_plane.input_paths:
			if intersect(query_path, data_path):
				output_paths.append(copy.deepcopy(data_path))
				dfs(idx + 1, output_paths)
				output_paths.pop()

	dfs(0, [])

	return output_planes


def intersect(query_path: Path, data_path: Path) -> bool:
	for axis, query_point in query_path.points.items():
		data_point = data_path.points.get(axis)
		if data_point is None: return False

		data_datum = data_point.first_datum()
		query_datum = query_point.first_datum()
		query_opr = query_point.opr.lexeme

		if query_opr == '=': success = query_point.first_kind() == 'WILD' or data_datum in {token.datum for token in query_point.dat}
		elif query_opr == '!=': success = query_point.first_kind() != 'WILD' and data_datum not in {token.datum for token in query_point.dat}

		else:
			if data_point.first_kind() not in LIT_NUM_KINDS: success = False
			elif query_point.first_kind() == 'WILD': success = True
			elif query_opr == '>': success = data_datum > query_datum
			elif query_opr == '>=': success = data_datum >= query_datum
			elif query_opr == '<': success = data_datum < query_datum
			elif query_opr == '<=': success = data_datum <= query_datum
			else: raise SyntaxError('E_CMP')

		if not success: return False

	return True