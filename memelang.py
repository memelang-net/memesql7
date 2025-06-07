# Memelang.net | (c) HOLTWORK LLC | Patents Pending
# Encode and decode Memelang string to list
# pair :: k=v :: [k operator, k value, v operator, v value]
# pairs :: pair list :: [pair, ...]
# chain :: pairs list :: pair list list :: [[pair, ...], ...]

import csv, random, re
from collections import defaultdict

MEMEBASE = {}
M_MIN = 1 << 20
M_MAX = 1 << 63

## ENCODE / DECODE

KO, KV, VO, VV = 0, 1, 2, 3
SEQL, SOUT = 0, 1 							# name for each slot in OPR[opr] list
END, SPC, NOT, WLD = ';',' ', '!', '*'		# Special characters
IB, IF = '@', '#'

OPR = { # operator characters and their settings
	None	: (None,	''),
	SPC		: ('=',		SPC),
	NOT		: ('!=',	SPC+NOT),
	'='		: ('=',		'='),
	'!='	: ('!=',	'!='),
	'>'		: ('>',		'>'),
	'<'		: ('<',		'<'),
	'>='	: ('>=',	'>='),
	'<='	: ('<=',	'<='),
}

OPRNUM = {'>', '<', '>=', '<='}
COLSTR = {'r', 'alp'}
SIGIL  = {IB, IF}
RE_NUM = re.compile(r'^[+-]?\d+(\.\d+)?$')
RE_ALP = re.compile(r'[^a-zA-Z0-9_\@\:]') # Complex string must be wrapped in quotes
RE_PAIR = re.compile(r'''
	\s*(!)?								# key operator
	\s*([A-Za-z0-9@#_:*,]+)				# keys
	\s*(!=|>=|<=|=|>|<)					# value operator
	\s*([^"\s;]+|"[^"]*(?:""[^"]*)*")	# values
	(\s|;|$)							# look-ahead
''', re.VERBOSE)


## HELPERS

# What type of value is this?
def typecheck(val, opr:str = None) -> str:
	if isinstance(val, (int, float)) or opr in OPRNUM: return 'num'
	if isinstance(val, str): return 'var' if val[0] in SIGIL else 'str'
	return None


# Variable index
def indexget(n: str) -> int:
	if len(n)<1: return 1
	index=int(n)
	if not index: raise Exception('indexget')
	return index


# Is this value a variable?
def varcheck(tok) -> tuple[str|None, int]:
	if not isinstance(tok, str) or not len(tok): return (None, 0)
	for sigil, mult, offset in (('#',1,-1), ('@',-1,0)):
		if tok.startswith(sigil):
			if tok.startswith(sigil+sigil):	return 'K', mult*indexget(tok[2:])+offset
			if tok[1:].isdigit(): 			return 'V', mult*indexget(tok[1:])+offset
			name, _, p = tok[1:].partition(':')
			return name.lower(), mult*indexget(p)+offset
	return (None, 0)

	
# Load variables for pairs
# Stack K and V
# Stack each certain <key>
def varstack (pairs: list[list]) -> defaultdict:
	vstack=defaultdict(list)
	if not pairs: return vstack

	for _, kv, _, vv in pairs:
		vstack['V'].append(vv)
		vstack['K'].append(kv)
		if len(kv) == 1 and isinstance(kv[0], str): vstack[kv[0].lower()].append(vv)

	return vstack


# Expand variables in current pair with values from prior pairs
def varexpand (vals: list, vstack: dict, lower:bool = False) -> tuple[list, bool]:
	xvals = []
	hasvar = False

	for tok in vals:
		vtyp = typecheck(tok)
		if vtyp != 'var': xvals.append(tok.lower() if lower and isinstance(tok, str) else tok)
		else:
			hasvar = True
			var, idx = varcheck(tok)
			if var not in vstack: raise ValueError(f'Var: {tok}')
			vslen = len(vstack[var])
			if (idx >= 0 and idx >= vslen) or (idx < 0 and abs(idx) > vslen): raise ValueError(f'Index: {tok}')
			xvals.extend([x.lower() if lower and isinstance(x, str) else x for x in vstack[var][idx]])

	return xvals, hasvar


# Do these values match?
def valcmp(c:str, e:str, v1: list, v2) -> bool:
	if not v1 or e is None: return True
	if v2 is None: raise ValueError(f'valcmp none {c} {v1}{e}{v2}')

	if isinstance(v1[0], (int, float)): 
		if not isinstance(v2, (int, float)): raise ValueError('v2 num')
		if len(v1)>1: raise ValueError('v1 len')
		if e == '=':  return (v2 == v1[0])
		elif e == '!=': return (v2 != v1[0])
		elif e == '>':  return (v2 > v1[0])
		elif e == '>=': return (v2 >= v1[0])
		elif e == '<':  return (v2 < v1[0])
		elif e == '<=': return (v2 <= v1[0])
		else: raise ValueError(f'valcmp opr {c} {v1[0]}{e}{v2}')
	
	if e == '=': return (str(v2).lower() in v1)
	elif e == '!=': return (str(v2).lower() not in v1)
	else: raise ValueError(f'valcmp str opr {c} {v1}{e}{v2}')


# Input: W,"X Y",Z
# Output: ['W', 'X Y', 'Z']
def valsplit(commalist: str) -> list:
	vals=[]
	for val in next(csv.reader([commalist], skipinitialspace=True)):
		if not len(val): continue
		elif val[0]!='"' and RE_NUM.fullmatch(val): vals.append(float(val) if '.' in val else int(val))
		else: vals.append(val)
	return vals


# Input: ['W', 'X Y', 'Z']
# Output: W,"X Y",Z
def valjoin(vals: list, opr: str = None) -> str:
	if not vals: return OPR[opr][SOUT]
	out = []
	for val in vals:
		if val is None: continue
		sval = str(val)
		if isinstance(val, str) and RE_ALP.search(sval): sval = '"' + sval.replace('"', '""') + '"'
		out.append(sval)
	return OPR[opr][SOUT] + ','.join(out)


## PARSERS

# Input: Memelang string as 'K=V K,K>V *=V,"x y"; K<* !K=V'
# Output: chain
def decode(memestr: str) -> list[list[list]]:

	# LAZY, FIX LATER FOR QUOTES 
	memestr = re.sub(r'\s*//[^"]+$', '', memestr, flags=re.MULTILINE).strip() # Remove comments
	memestr = memestr.replace(' -> ', ' m!=@m ', memestr) # Desugar join

	if len(memestr) == 0: raise Exception('Empty memestr')
	memestr+=END

	chain, pairs, vdepth = [], [], defaultdict(int)
	pos = 0

	while pos < len(memestr):
		m = RE_PAIR.match(memestr, pos)
		if not m: raise ValueError(f'Parse error near: {memestr[pos:pos+40]!r}')

		pair = [m.group(1) or SPC, valsplit(m.group(2)), m.group(3), valsplit(m.group(4))]

		for slot in (KV,VV):
			for v in pair[slot]:
				var, pop = varcheck(v)
				if var is not None and vdepth[var]<pop: raise Exception(f'Var depth @{var}:{v}')
				if v == WLD and len(pair[slot])>1: raise Exception(f'Wild comma')

			vdepth['V']+=1
			vdepth['K']+=1
			if pair[KO]==SPC and len(pair[KV])==1 and pair[KV][0]!=WLD: vdepth[str(pair[KV][0]).lower()]+=1
			if 'm' not in vdepth: vdepth['m']=1 # auto populates

		if pair[VO] in OPRNUM and len(pair[VV])>1: raise Exception(f'Bad opr-val {pair[VO]}{pair[VV]}')

		pairs.append(pair)

		if m.group(5) == END:
			if pairs: chain.append(pairs)
			pairs = []
			vdepth = defaultdict(int)

		pos = m.end()

	return chain


# Input: memes
# Output: Memelang string "opr1val1opr2val2"
def encode(chain: list[list[list]]) -> str:
	memestr = ''
	for pairs in chain:
		for ko, kv, vo, vv in pairs:
			if len(kv)==1 and len(str(kv[0]))==1: memestr+='\n'
			memestr += valjoin(kv, ko) + valjoin(vv, vo)
		memestr+=END
	return memestr


## IN-MEMORY DB

# Store memes as in-memory DB
def member(pairs: list[list]):
	mval = None

	# b=basekey
	if not pairs or not pairs[0] or not pairs[0][VV]: raise ValueError("member basekey")
	basekey = pairs[0][VV][0]

	if basekey not in MEMEBASE: MEMEBASE[basekey]=[]

	mv = []
	for pair in pairs:
		if not pair: continue
		ko, kv, vo, vv = pair

		if len(kv) != 1 or kv[0] is None: raise ValueError(f'Bad kv, cannot be empty')
		if len(vv) != 1 or vv[0] is None: raise ValueError(f'Bad vv for {kv}')
		if ko != SPC: raise ValueError(f'Bad ko: {ko}{kv}{vo}{vv}')
		if vo != '=': raise ValueError(f'Bad vo: {ko}{kv}{vo}{vv}')

		if typecheck(kv[0]) not in ('str','num') or typecheck(vv[0]) not in ('str','num'): raise ValueError('member variable not allowed')

		if kv[0] == 'b': continue
		if kv[0] == 'm':
			if vv[0]: mv = vv
			else: mv = []
			continue

		if not mv:
			if not mval: mval = random.randrange(M_MIN, M_MAX)
			mval += 1
			mv = [mval]

		row = {'m':mv[0], 'r':kv[0], 'alp':None, 'amt':None}
		acol = 'amt' if typecheck(vv[0])=='num' else 'alp'
		row[acol]=vv[0]
		MEMEBASE[basekey].append(row)


# Evaluate one query-meme meme against rows (full join logic)
# query meme finds new hexes
# input memes contains prior memes
def remember(qpairs: list[list], ichain: list[list[list]]) -> list[list]:

	if not qpairs or not qpairs[0][VV]: raise ValueError("remember basekey")
	basekey = qpairs.pop(0)[VV][0]

	if basekey not in MEMEBASE: raise ValueError('remember basekey')

	ichain = ichain or [[[]]]

	def dfs(idx:int, chosen:list, vstack:dict, mo, mv, newm):
		if idx == len(qpairs):
			results.append(chosen[:])
			return

		ko, kv, vo, vv = qpairs[idx][:]
		vstack2 = defaultdict(list, {k: v[:] for k, v in vstack.items()})

		# Expand variables
		xkv, _ = varexpand(kv, vstack2, True)
		xvv, _ = varexpand(vv, vstack2, True)

		if kv == ['m']:
			dfs(idx + 1, chosen, vstack2, vo, xvv, True)
			return

		acol = 'amt' if xvv and typecheck(xvv[0], vo)=='num' else 'alp'

		mgroups = defaultdict(list)

		for row in MEMEBASE[basekey]:
			if (valcmp('m', OPR[mo][SEQL], mv, row['m'])
			and valcmp('r', OPR[ko][SEQL], xkv, row['r'])
			and valcmp(acol, OPR[vo][SEQL], xvv, row[acol])):
				mgroups[row['m']].append(row)

		for m, rows in mgroups.items():
			vstack3 = defaultdict(list, {k: v[:] for k, v in vstack.items()})

			if newm:
				vstack3['K'].append(['m'])
				vstack3['V'].append([m])
				vstack3['m'].append([m])

			keyvar = None if len(kv)!=1 or typecheck(kv[0])!='str' else kv[0].lower()
			if keyvar: vstack3[keyvar].append([])
			vstack3['V'].append([])
			vstack3['K'].append([])

			for row in rows:
				aval = row['alp'] if row['amt'] is None else row['amt']
				vstack3['V'][-1].append(aval)
				vstack3['K'][-1].append(row['r'])
				if keyvar: vstack3[keyvar][-1].append(aval)

			chosen.extend(row.copy() for row in rows)
			dfs(idx + 1, chosen, vstack3, '=', m, False)
			for _ in rows: chosen.pop()

	ochain = []

	for pairs in ichain:

		results = []
		dfs(0, [], varstack(pairs), None, None, True)

		# Convert satisfying row tuples back into meme structures
		for combo in results:
			opairs = pairs[:]
			mv = None
			for row in combo:
				if mv != row['m']:
					opairs.append([SPC, ['m'], '=', row['m']])
					mv=row['m']
				opairs.append([SPC, row['r'], '=', (row['alp'] if row['alp'] is not None else row['amt'])])
			ochain.append(opairs)

	return ochain


## SQL DB
# 'a' is a value stored as either 'alp' for str or 'amt' for int/float
# 'r' is the relation of 'a' to the meme
# 'm' is a meme identifier that groups r=a pairs

def selectify(pairs: list[list], table: str = 'meme', t:int = 0) -> tuple[str, list]:

	tm1 = 0
	acol = 'alp'
	selects, wheres, joins, groupbys, params = [], [], [], [], []
	vstack = defaultdict(list)
	mo, mv = '=', []

	for ko, kv, vo, vv in pairs:

		# M
		if kv == ['m']:
			mo, mv = vo, vv
			continue

		# R
		kv = [str(v).lower() for v in kv]
		sel = f"""'{ko}' || t{t}.r || '=' """

		# A as ALP/AMT
		acol = 'alp'
		atyp = typecheck(vv[0], vo)
		if atyp == 'num':
			acol = 'amt'
			sel += f"""|| t{t}.amt"""
		elif atyp == 'str':
			vv = [v.lower() if isinstance(v, str) else v for v in vv]
			sel += f"""|| '"' || t{t}.alp || '"'"""
		else:
			sel += f"""|| (CASE WHEN t{t}.amt IS NOT NULL THEN t{t}.amt::text ELSE '"' || t{t}.alp || '"' END)"""

		# Join/Where values
		joint = {'r':None, 'alp':None, 'amt':None, 'm':None}
		for c, o, vals in (('m', mo, mv), ('r', OPR[ko][SEQL], kv), (acol, OPR[vo][SEQL], vv)):
			if not vals or not o: continue

			xvals, hasvar = varexpand(vals, vstack)
			xvallen = len(xvals)

			wherestr = f"LOWER(t{t}.{c})" if c in COLSTR else f"t{t}.{c}"
			if xvallen == 1: wherestr+=o
			elif o == '=': wherestr+=' IN ('
			elif o == '!=': wherestr+=' NOT IN ('
			else: raise Exception('where in')

			for i, val in enumerate(xvals):
				if i>0: wherestr+=','
				wherestr+="%s"
				params.append(val)

			if xvallen > 1: wherestr+=')'

			if len(vals)==1 and hasvar: joint[c] = wherestr # Join on single var
			else: wheres.append(wherestr) # Where for all else

		# JOINING
		# start with from
		if not joins:
			joins.append(f'FROM {table} t{t}')

		# join
		else:
			# new join
			if all(v is None for v in joint.values()): raise Exception('join')
			jands, jors = [], []
			for jcol, cond in joint.items():
				if cond: jands.append(cond)
				else: jors.append(f"t{tm1}.{jcol}!=t{t}.{jcol}")

			jandstr = ' AND '.join(jands)
			jorstr = (' AND (' + ' OR '.join(jors) + ')') if jors else ''
			joins.append(f"LEFT JOIN {table} t{t} ON {jandstr}{jorstr}")

		# new m, group by m and select m=
		if mo != '=' or mv != ['@m']:
			groupbys.append(f"t{t}.m")
			selects.append(f"' m='")
			selects.append(f"t{t}.m")
			vstack['m'].append([f"t{t}.m"])

		# set @rel variable
		if OPR[ko][SEQL] == '=' and len(kv)==1:
			vstack[kv[0]].append([f"LOWER(t{t}.alp)" if acol=='alp' else f"t{t}.amt"])

		vstack['V'].append([f"t{t}.{acol}" if acol == 'amt' else f"LOWER(t{t}.{acol})"])
		vstack['K'].append([f"LOWER(t{t}.r)"])

		# Select
		selects.append(f"""string_agg(DISTINCT {sel}, '')""")

		mo, mv = '=', ['@m']
		tm1=t
		t+=1

	if not joins: raise Exception('no join')

	joinstr = ' '.join(joins)
	selectstr = ','.join(selects)
	groupbystr = ','.join(groupbys)
	wherestr = ('WHERE ' + ' AND '.join(wheres)) if wheres else ''

	return f"SELECT CONCAT({selectstr}, ' ') AS v {joinstr} {wherestr} GROUP BY {groupbystr}", params


def select(imemes: list[list[list]], table: str = 'meme') -> tuple[str, list]:
	selects, params = [], []
	for meme in imemes:
		qry_select, qry_params = selectify(meme, table)
		selects.append(qry_select)
		params.extend(qry_params)
	return f"SELECT string_agg(v, '') AS vv FROM (" + ' UNION '.join(selects) + ")", params


def insert (imemes: list[list[list]], table: str = 'meme') -> tuple[str, list]:
	
	basekey = imemes[0][0][VV][0] # b=basekey
	imemes[0][0][VV][0] = 'insert'+str(random.randrange(M_MIN, M_MAX)) # temporary basekey
	member(imemes)

	rows, params = [], []
	for row in MEMEBASE[basekey]:
		rows.append('(%s,%s,%s,%s)')
		params.extend([row['r'], row['alp'], row['amt'], row['m']])

	MEMEBASE[basekey]=[]

	if rows: return f"INSERT INTO {table} VALUES " + ','.join(rows) + " ON CONFLICT DO NOTHING", params

	return None, []


# INSECURE - display only
def morgify(sql: str, params: list) -> str:
	for param in params: sql = sql.replace("%s", param, 1)
	return sql
