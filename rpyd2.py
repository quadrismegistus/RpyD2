"""
RpyD2
depends:
	rpy2  <http://rpy.sourceforge.net/rpy2.html>
"""
from __future__ import division
from rpy2 import robjects as ro
r = ro.r
from rpy2.robjects.packages import importr
grdevices = importr('grDevices')
rprint = ro.globalenv.get("print")

class InputNotRecognizedError(Exception):
	pass

def load(fn,toprint=True):
	import pickle
	if toprint: print ">> loading:",fn,"...",
	d,df=pickle.load(open(fn))
	r=RpyD2([])
	r.__dict__=d
	for k,v in d.items():
		setattr(r,k,v)
	r.df=ro.DataFrame(df)
	print "done."
	return r

def from_csv(fn,sep='\t',lb='\n',header=True):
	t=open(fn).read()
	header=[]
	ld=[]
	for ln in t.split(lb):
		if not header: header=ln.split(sep); continue
		d={}
		lndat=ln.split(sep)
		if len(lndat)!=len(header): continue
		for i,x in enumerate(lndat):
			if x[0].isdigit():
				if '.' in x:
					x=float(x)
				else:
					x=int(x)
			d[header[i]]=x
		ld.append(d)
	return RpyD2(ld)

def write(fn,data,toprint=False,join_line='\n',join_cell='\t'):
	if type(data)==type([]):
		o=""
		for x in data:
			if type(x)==type([]):
				z=[]
				for y in x:
					if type(y)==type(u''):
						y=y.encode('utf-8')
					z+=[y]
				x=z
				line=join_cell.join(x)
			else:
				try:
					line=str(x)
				except UnicodeEncodeError:
					line=x.encode('utf-8')
			line=line.replace('\r','').replace('\n','')
			o+=line+join_line
	else:
		o=str(data)
	of = open(fn,'w')
	of.write(o)
	of.close()
	if toprint:
		print ">> saved: "+fn

def signcorr(samplesize):
	import numpy as np
	return 1.96/np.sqrt(samplesize-3)

# 
# 
# def mergeDL(dl1,dl2,rstyle=True):
# 	for k in dl2:	
# 		kk=k
# 		if rstyle:
# 			if kk[0].isdigit(): kk='X'+kk
# 			kk=kk.replace("'",".")
# 			kk=kk.replace("-",".")
# 			kk=kk.replace(" ",".")
# 			
# 		try:
# 			dl1[kk].extend(dl2[k])
# 		except KeyError:
# 			print(">> error merging DLs: '"+kk+"' not found in original DL")
# 			continue
# 	return dl1
# 	

def rkey(kk):
	if kk[0].isdigit(): kk='X'+kk
	kk=kk.replace("'",".")
	kk=kk.replace("-",".")
	kk=kk.replace(" ",".")
	kk=kk.replace(",",".")
	return kk

def mergeDL(dl1,dl2,rstyle=True):
	dl={}
	for k in dl1:
		dl[k]=[]
		dl[k].extend(dl1[k])
	
	for k in dl2:	
		kk=k
		if rstyle:
			kk=rkey(kk)
		
		try:
			dl[kk].extend(dl2[k])
		except KeyError:
			print(">> error merging DLs: '"+kk+"' not found in original DL")
			continue
	return dl
			
def printDL(dl):
	for k in dl:
		print k, len(dl[k]), dl[k][0], "..."	
	print
	

def ndian(numericValues,n=2):
	theValues = sorted(numericValues)
	if len(theValues) % n == 1:
		return theValues[int((len(theValues)+1)/n)-1]
	else:
		lower = theValues[int(len(theValues)/n)-1]
		upper = theValues[int(len(theValues)/n)]
		return (float(lower + upper)) / n

def median(numericValues):
	try:
		import numpy as np
		return np.median(numericValues)
	except:
		return ndian(numericValues,n=2)

def lowerq(numericValues):
	return ndian(numericValues,n=4)

def upperq(numericValues):
	return ndian(numericValues,n=(4/3))

def upperthird(numericValues):
	return ndian(numericValues,n=(3/2))

def upperthirdfifth(numericValues):
	return ndian(numericValues,n=(5/3))

def lowereighth(numericValues):
	return ndian(numericValues,n=8)

def lowerfifth(numericValues):
	return ndian(numericValues,n=5)

def lowertenth(numericValues):
	return ndian(numericValues,n=10)

def mean(l):
	return mean_stdev(l)[0]

def correlate(x,y):
	from statlib.stats import pearsonr
	return pearsonr(x,y)


def str_polyfunc(coefficients):
	frmla=[]
	import decimal
	deg=len(coefficients)-1
	for i in range(len(coefficients)):
		term=str(decimal.Decimal(str(coefficients[i])))
		power=deg-i
		if power:
			term+='*x^'+str(deg-i)
		frmla+=[ term ]
	frmla=' + '.join(frmla)
	return frmla

def make_polyfunc(coefficients):
	l=[(coefficients[i],(len(coefficients)-1-i)) for i in range(len(coefficients))]
	return lambda x: sum([t[0]*(x**t[1]) for t in l])



	

class RpyD2():
	def __init__(self,input,**kwargs):
		"""
         input is your data, which can be in the following forms:
           1. LD (List of Dictionaries)
               [ {'hair':'blonde','eyes':'blue'}, {'hair':'black','eyes':'green'}, ... ]
           2. DL (Dictionary of Lists)
               { 'hair':['blonde','blue'], 'eyes':['blue','green] }
           3. Rpy2 DataFrame
           4. Another RpyD2

 
         Keyword arguments will override the following default options:
          self.cols=None                  # specify which columns to build from
          self.rownamecol=None            # specify a column name from which row names should be used
          self.allcols=False              # if False, columns limited to those shared among all rows;
                                            if True, all columns are chosen;
                                            if a positive integer N, columns limited to the 'top' N columns,
                                               where columns are compared numerically by:
          self.trimbyVariance=True             # if trimbyVariance==True, sum of absolute value of Z-scores across column
                                                 otherwise, sum of scores across column

          self.rank=True                  # if rank==True, append 'r'+ranknum to the top N columns
          self.zero=0.0                   # if allcols is True or an integer, what should empty cells be populated with?
          self.z=False                    # if True, Z-score all quantitative columns
          self.factor=True                # if True, treat strings as factors
          self.onlyQuant=False            # if True, only build quantitative columns
          self.onlyCat=False              # if True, only build categorical (string) columns

		  self.toprint=True				  # if True, print R objects using R's summary() before returning them
		  self.colkey={}
		  self.rowcols=[]	# these will be kept even if allcols is set to a number, trimming columns
		"""
				
		## set defaults
		self.cols=None
		self.rownamecol=None
		self.allcols=False
		self.trimbyVariance=True
		self.rank=True
		self.factor=True
		self.z=False
		self.zero=0.0
		self.onlyQuant=False
		self.onlyCat=False
		self.rownames=[]
		self.toprint=True
		self.colkey={}
		self.rowcols=[]
		
		## override defaults with
		for k,v in kwargs.items():
			setattr(self,k,v)
		
		## double override with non-keyword
		#self.input=input
		self.df=None
		self.nrow=0
		self.ncol=0
		self._quantv=None
		self._quantvz=None
		self._tv=None
		self._subv={'cols':{},'rows':{},'cols_rows':{}}
		self._groupv={}
		
		if not input: return
		if not len(input): return
	
		
		if type(input)==type([]) and type(input[0])==type({}):
			self._gen_LD(input)
		elif type(input)==type({}) and type(input.values()[0]==type([])):
			self._gen_DL(input)
		elif type(input)==type(self):
			self._gen_self(input)
		elif type(input)==type(ro.DataFrame({})):
			self._gen_DF(input)
		else:
			raise InputNotRecognizedError("Cannot recognize input of type "+type(input))

	def __str__(self):
		return self.df.__str__()
	
	def __repr__(self):
		loc=object.__repr__(self)[:-1].split(" at ")[1]
		return "<RpyD2 @ "+loc+" storing a "+str(self.nrow)+"x"+str(self.ncol)+" "+self.df.__repr__()[1:-1].replace(" - "," @ ")+">"
	
	def col(self,colname):
		"""Return column 'colname', where colname can be either a string name or an integer position (starting at 0)."""
		try:
			if type(colname)==type(unicode()): colname=colname.encode('utf-8')
			
			if type(colname)==type(''):
				colnum=self.cols.index(colname)
			else:
				colnum=colname
			
			c=self.df[colnum]
			if type(c)==type(ro.FactorVector([])):
				return list([c.levels[c[item_i] - 1] for item_i in range(len(c))])
			else:
				return list(c)
		except KeyError:
			return

	def row(self,rowname):
		"""Return row 'rowname', where rowname can be either a string name or an integer position (starting at 0)."""
		try:
			if type(rowname)==type(''):
				rownum=self.rows.index(rowname)
			else:
				rownum=rowname
				
			l=[]
			for colnum in range(self.ncol):
				c=self.df[colnum]
				if type(c)==type(ro.FactorVector([])):
					l+=[c.levels[c[rownum] - 1]]
				else:
					l+=[c[rownum]]
		
			return l
		except:
			return
	
	def t(self):
		if not self._tv:
			self._tv=RpyD2(r['as.data.frame'](r['t'](self.df)))
			self._tv.rownames=list(self._tv.df.rownames)
		return self._tv
	
	def toLD(self,rownamecol=False):
		ld=[]
		if rownamecol and type(rownamecol)!=type(''): rownamecol='xkey'
		for row in self.rows:
			d=dict(zip(self.cols, self.row(row)))
			if rownamecol: d[rownamecol]=row
			ld.append(d)
		return ld
	
	def toDL(self,cols=None,rows=None,rownamecol=False):
		"""
		Return a dictionary of lists representation of self:
		{'col0':[row0val,row1val,...],
		'col1':[row1val,row2val,...],
		...}
		If rows is a non-empty list, return only these rows.
		If cols is a non-empty list, return only these cols.
		If both are non-empty, return only these rows and only these cols.
		"""
		dl={}
		if rownamecol and type(rownamecol)!=type(''):
			rownamecol='rownamecol'
		
		if not cols and not rows:
			for i in range(self.ncol):
				col=self.col(i)
				colname=self.cols[i]
				dl[colname]=col
			if rownamecol:
				dl[rownamecol]=list(self.df.rownames)
		
		elif cols and not rows:
			for col in cols:
				dl[col]=self.col(col)
			if rownamecol:
				dl[rownamecol]=list(self.df.rownames)
		
		elif cols and rows:
			for col in cols:
				dl[col]=[]
				colnum=self.cols.index(col)
				for row in rows:
					rowdat=self.row(row)
					dl[col].append(rowdat[colnum])
			if rownamecol:
				dl[rownamecol]=[]
				for row in rows:
					if type(row)==type(''):
						dl[rownamecol].append(row)
					else:
						dl[rownamecol].append(self.rows[row])
		
		elif rows and not cols:
			for col in self.cols:
				dl[col]=[]
				colnum=self.cols.index(col)
				for row in rows:
					rowdat=self.row(row)
					dl[col].append(rowdat[colnum])		
			if rownamecol:
				dl[rownamecol]=[]
				for row in rows:
					if type(row)==type(''):
						dl[rownamecol].append(row)
					else:
						dl[rownamecol].append(str(self.rows[row]))
		return dl

	def save(self,fn=None):
		if not fn: 
			import time
			fn=".".join( [ 'rpyd2', time.strftime('%Y%m%d.%H%M.%S', time.gmtime()) , 'pickle' ] )
		import pickle
		pickle.dump((self.__dict__,self.df),open(fn,'wb'))
		print ">> saved:",fn
	
	def get(self,col=None,row=None):
		if col and row:
			return self.col(col)[self.rows.index(row)]
			
		elif col and not row:
			return self.col(col)
			
		elif row and not col:
			return self.row(row)
			
		elif not col and not row:
			return self
		
	def sub(self,cols=[],rows=[],removeCommon=True):
		"""Return an RpyD2 from self, with only those rows and/or columns as specified."""
		
		if cols and not rows:
			if cols==self.cols:
				return self
			keytup=('cols',tuple(sorted(cols)))
		elif cols and rows:
			keytup=('cols',tuple( tuple(sorted(cols)), tuple(sorted(rows)) ))
		elif rows and not cols:
			keytup=('rows',tuple(sorted(rows)))
		elif not cols and not rows:
			return self
		
		try:
			return self._subv[keytup[0]][keytup[1]]
		except KeyError:
			dl=self.toDL(cols,rows,rownamecol=True)
			dlkeys=[k for k in dl.keys() if k!='rownamecol' and k and not k.startswith('index_') and not k.startswith('row_')]
			if len(dlkeys)>1:
				dlkey=dlkeys[0]
				from difflib import SequenceMatcher as SM
				for dlkey2 in dlkeys[1:]:
					s=SM(None,dlkey,dlkey2)
					match=sorted(s.get_matching_blocks(),key=lambda m: -m.size)[0]
					dlkey=dlkey2[match.b:match.b+match.size]
				
				dl=dict(( (k.replace(dlkey,'') if k.replace(dlkey,'') else k),v) for k,v in dl.items())
			
			m=RpyD2(dl,rownamecol='rownamecol')
			self._subv[keytup[0]][keytup[1]]=m
			return m
	
	def q(self,z=False):
		"""Return a version of self of only quantitative columns"""
		if self.onlyQuant:
			return self
		
		
		if not z and self._quantv!=None:
			return self._quantv
		elif z and self._quantvz!=None:
			return self._quantvz
				
		r=RpyD2(self.toDL(),rownames=self.rownames,onlyQuant=True,z=z)
		if z:
			self._quantvz=r
		else:
			self._quantv=r
		return r
	
	
	def _is_quant_num(self,num):
		try:
			num+0
			return True
		except:
			return False
	
	def _is_quant(self,l,FalseAtNone=True):
		if not FalseAtNone:
			return self._is_quant_num(l[0])
		else:
			returnVal=True
			for x in l:
				if not self._is_quant_num(x):
					returnVal=False
					break
			return returnVal
			
		
	
	def _gen_DL(self,dl):
		self.origin='dl'
		if self.rownamecol:
			self.rownames=dl[self.rownamecol]
			del dl[self.rownamecol]
		self._boot_DL(dl)
		
	def _gen_LD(self,ld):
		self.origin='ld'
		dd={}
		
		if not self.cols:
			self.cols=getCols(ld,self.allcols,self.rownamecol)
			if type(self.allcols)==type(2):
				self.cols=trimCols(ld,self.cols,self.allcols,byVariance=self.trimbyVariance,rowcols=self.rowcols)
		
		for i in range(len(self.cols)):
			k=self.cols[i]
			dd[k]=[]
		
		if self.rownamecol:	self.rownames=[]
		for x in ld:
			if self.rownamecol: self.rownames.append(x[self.rownamecol])
			
			for k in self.cols:
				try:
					value=x[k]
				except KeyError:
					try:
						value=x[".".join(k.split(".")[1:])]
					except:
						value=self.zero
				
				dd[k].append(value)
		
		self._boot_DL(dd)
	
	def _gen_DF(self,df):
		self._set_DF(df)

	def _boot_DL(self,dl,rownames=None):
		dd={}
		import datetime
		for k,v in dl.items():
			if type(k)==type(unicode()):
				k=str(k.encode('utf-8','replace'))
			
			#print [type(vv) for vv in v]
			#if type(v[0])==type(''):
			if isinstance(v[0],basestring):
				if self.onlyQuant: continue
				if type(v[0])==type(unicode()): v = [vx.encode('utf-8','replace') for vx in v]
				dd[k]=ro.StrVector(v)
				if self.factor:
					dd[k]=ro.FactorVector(dd[k])
			elif isinstance(v[0],datetime.datetime):
				import time
				dd[k]=ro.vectors.POSIXlt( [ time.struct_time([vv.year,vv.month,vv.day,0,0,0,0,0,0]) for vv in v]  )
			else:
				if self.z:
					v=zfy(v)
				
				try:
					if not '.' in str(v[0]):
						dd[k]=ro.IntVector(v)
					else:
						dd[k]=ro.FloatVector(v)
				except:
					continue
		
		df=ro.DataFrame(dd)
		self._set_DF(df)
		

	def _set_DF(self,df):
		if self.rownames:
			df.rownames=ro.FactorVector(self.rownames)
			#del self.rownames
		self.df=df
		self.nrow=self.df.nrow
		self.ncol=self.df.ncol
		self.cols=list(self.df.colnames)
		self.rows=list(self.df.rownames)

	def rankcols(self,byVariance=False,returnSums=False,rankfunc=None):
		ranks={}
		for colname in self.cols:
			col=self.col(colname)
			if byVariance:
				rankcol=sum([abs(x) for x in zfy(col)])
			else:
				if not rankfunc:
					rankcol=sum(col)
				else:
					rankcol=rankfunc(col)
					
			ranks[colname]=rankcol
		
		keys=sorted(ranks,key=lambda item: -ranks[item])
		if not returnSums:
			return keys
		return (keys,[ranks[k] for k in keys])
		
		
			
	def plots(self,x=None,y=None,n=1):
		if not y: y=self.rankcols()
		if not n or n==1:
			if type(y)==type([]):
				for ykey in y:
					self.plot(x=x,y=y)
			else:
				self.plot(x=x,y=y)
			return
		
		# else, stepwise
		a=None
		lc=len(str(self.ncol))
		#rg=self.group(x=x,ys=y)
		for b in range(0,len(y),n):
			if a==None:
				a=b
				continue
			print a,b
			
			sub=self.group(ys=y[a:b],x=x)
			#sub=self.sub_where(rows={'y_type':y[a:b]})
			
			abk=str(a).zfill(lc)+'-'+str(b).zfill(lc)
			sub.plot(fn='plots.'+abk+'.'+self._get_fn(x,"+".join(y[a:b])) +'.png', x='row',y='y',col='y_type', group='y_type', smooth=False, line=True)
			a=b
		return
		

	def _kwd(self,kwd,**kwarg):
		for k,v in kwarg.items():
			if not k in kwd:
				kwd[k]=v
		return kwd

	def plot(self, fn=None, x=None, y=None, **opt):
		opt=self._kwd(opt,col=None, group=None, w=1100, h=800, size=2, smooth=False, point=True, jitter=False, boxplot=False, boxplot2=False, title=False, flip=False, se=False, density=False, line=False, bar=False, xlab_size=14, ylab_size=14, position='identity', xlab_angle=0, logX=False, logY=False, area=False, text=False, text_size=3, text_angle=45, pdf=False, freqpoly=False)

		if opt['jitter']: opt['position']='jitter'
		
		#df=self.df

		grdevices = importr('grDevices')

		

		import rpy2.robjects.lib.ggplot2 as ggplot2

		#print hasattr(self.df,'x')
		
		if not x:
			if self.df.rownames[0].isdigit():
				self.df.x=ro.FloatVector([float(xx) for xx in list(self.df.rownames)])
			else:
				self.df.x=self.df.rownames
			x='x'
		
		#self.df.y = self.df[self.df.colnames.index(y)]
		#print self.df.y

		gp = ggplot2.ggplot(self.df)
		pp = gp

		if x and y:
			if opt['col'] and opt['group']:
				pp+=ggplot2.aes_string(x=x, y=y,col=opt['col'],group=opt['group'])
			elif opt['col']:
				pp+=ggplot2.aes_string(x=x, y=y,col=opt['col'])
			elif opt['group']:
				pp+=ggplot2.aes_string(x=x, y=y,group=opt['group'])
			else:
				pp+=ggplot2.aes_string(x=x, y=y)
		else:
			if not opt['density'] and not opt['bar'] and not opt['freqpoly']:
				self._call_remaining('plot',fn=fn,x=x,y=y,**opt)
				return
				
		if type(fn)!=type(''): fn=''
		if not fn.endswith('.png'):
			fn+='plot.'+self._get_fn(x,y)+'.png'

		if opt['text']:
			if opt['col']:
				pp+=ggplot2.geom_text(ggplot2.aes_string(label=opt['text'],colour=opt['col']),angle=opt['text_angle'],size=opt['text_size'])
			else:
				pp+=ggplot2.geom_text(ggplot2.aes_string(label=opt['text']),angle=opt['text_angle'],size=opt['text_size'])


		if opt['boxplot']:
			if opt['col']:
				pp+=ggplot2.geom_boxplot(ggplot2.aes_string(fill=opt['col']),color='blue',position=opt['position'])
			else:
				pp+=ggplot2.geom_boxplot(color='blue')	

		if opt['point']:
			if opt['col']:
				pp+=ggplot2.geom_point(ggplot2.aes_string(fill=opt['col'],col=opt['col']),size=opt['size'],position=opt['position'])
			else:
				pp+=ggplot2.geom_point(size=opt['size'],position=opt['position'])


		if opt['boxplot2']:
			if opt['col']:
				pp+=ggplot2.geom_boxplot(ggplot2.aes_string(fill=opt['col']),color='blue',outlier_colour="NA")
			else:
				pp+=ggplot2.geom_boxplot(color='blue')

		if opt['smooth']:
			if opt['smooth']=='lm':
				if opt['col']:
					pp+=ggplot2.stat_smooth(ggplot2.aes_string(col=opt['col']),size=1,method='lm',se=opt['se'])
				else:
					pp+=ggplot2.stat_smooth(col='blue',size=1,method='lm',se=opt['se'])
			else:
				if opt['col']:
					pp+=ggplot2.stat_smooth(ggplot2.aes_string(col=opt['col']),size=1,se=opt['se'])
				else:
					pp+=ggplot2.stat_smooth(col='blue',size=1,se=opt['se'])

		if opt['density']:
			if opt['density']=='h':
				if opt['col'] and opt['group']:
					pp+=ggplot2.geom_histogram(ggplot2.aes_string(x=x,y='..count..',group=opt['group'],fill=opt['col'],col=opt['col'],alpha=0.2))
				else:
					pp+=ggplot2.geom_histogram(ggplot2.aes_string(x=x,y='..scaled..'))
				
			else:
				if opt['col']:
					pp+=ggplot2.geom_density(ggplot2.aes_string(x=x,y='..scaled..',fill=opt['col'],col=opt['col'],alpha=0.2))
				else:
					pp+=ggplot2.geom_density(ggplot2.aes_string(x=x,y='..count..'))

		if opt['line']:
			pp+=ggplot2.geom_line(position=opt['position'])

		if opt['area']:
			pp+=ggplot2.geom_area(ggplot2.aes_string(x=x,y=y,fill=opt['col'],col=opt['col']))
		
		if opt['bar']:
			if y:
				if opt['col']:
					pp+=ggplot2.geom_bar(ggplot2.aes_string(x=x,y=y,fill=opt['col']),position='dodge',stat='identity')
				else:
					pp+=ggplot2.geom_bar(ggplot2.aes_string(x=x,y=y),position='dodge',stat='identity')
			else:
				if opt['col']:
					pp+=ggplot2.geom_bar(ggplot2.aes_string(x=x,fill=opt['col']),position='dodge')
				else:
					pp+=ggplot2.geom_bar(ggplot2.aes_string(x=x),position='dodge')
		
		if opt['freqpoly']:
			if opt['col'] and opt['group']:
				pp+=ggplot2.geom_freqpoly(ggplot2.aes_string(x=x,col=opt['col'],group=opt['group']),position=opt['position'])
			else:
				pp+=ggplot2.geom_bar(ggplot2.aes_string(x=x),position=opt['position'])
		
		if opt['logX']:
			pp+=ggplot2.scale_x_log10()
		if opt['logY']:
			pp+=ggplot2.scale_y_log10()

		
		
		if not opt['title']:
			opt['title']=fn.split("/")[-1]


		pp+=ggplot2.opts(**{'title' : opt['title'], 'axis.text.x': ggplot2.theme_text(size=opt['xlab_size'],angle=opt['xlab_angle']), 'axis.text.y': ggplot2.theme_text(size=opt['ylab_size'],hjust=1)} )
		#pp+=ggplot2.scale_colour_brewer(palette="Set1")
		pp+=ggplot2.scale_colour_hue()

		if opt['flip']:
			pp+=ggplot2.coord_flip()

		if opt['pdf']:
			fn=fn.replace('.png','.pdf')
			grdevices.pdf(file=fn, width=opt['w'], height=opt['h'])
		else:
			grdevices.png(file=fn, width=opt['w'], height=opt['h'])
			
		pp.plot()
		grdevices.dev_off()
		print ">> saved: "+fn
	
	def toVectors(self,xcol='x',ycol='y'):
		vectors={}
		
		ydat=self.col(ycol)
		xdat=self.col(xcol)
		for rownum in range(len(ydat)):
			key=xdat[rownum]
			if type(key)==type(''):
				key=key.strip()
			if not key: continue
			val=ydat[rownum]
			try:
				vectors[key].append(val)
			except KeyError:
				vectors[key]=[]
				vectors[key].append(val)
		
		for k,v in vectors.items():
			try:
				vectors[k]=ro.FloatVector(v)
			except:
				vectors[k]=ro.StrVector(v)
				if self.factor:
					vectors[k]=ro.FactorVector(vectors[k])
				
		return vectors
		
	def _get_fn(self,x,y):
		if y:
			if type(y)==type([]):
				y='+'.join(y)
			elif type(y)!=type(''):
				y=str(y)
			
		if x and y:
			return '-by-'.join([y,x])
		elif x:
			return x
		elif y:
			return y
		else:
			return "_x,y_"
	
	
	def boxplot(self,fn=None,x=None,y=None,main=None,xlab=None,ylab=None,ggplot=False,w=1100,h=800):
		if not (x and y):
			self._call_remaining('boxplot',x=x,y=y)
			return
		
		
		if ggplot:
			self.plot(fn=fn,x=x,y=y,w=w,h=h,title=main,point=False,smooth=False,boxplot2=True,col=x,group=x)
			return
		if fn==None: fn='boxplot.'+self._get_fn(x,y)+'.png'
		if not main: main=fn
		if not xlab: xlab=x
		if not ylab: ylab=y
		
		grdevices = importr('grDevices')
		grdevices.png(file=fn, width=w, height=h)
		frmla=ro.Formula(y+'~'+x)
		r['boxplot'](frmla,data=self.df,main=main,xlab=xlab,ylab=ylab)
		
		grdevices.dev_off()
		print ">> saved: "+fn
		
		
	def _call_remaining(self,function_name,x=None,y=None,fn=None,**opt):
		function=getattr(self,function_name)
		if y and not x:
			for col in self.cols:
				if col==y: continue
				try:
					function(x=col,y=y,fn=fn)
				except:
					pass
			return
		elif x and not y:
			for col in self.cols:
				if col==x: continue
				try:
					function(x=x,y=col,fn=fn,**opt)
				except:
					pass
			return
		elif not x and not y:
			return
	
	def vioplot(self,fn=None,x=None,y=None,w=1100,h=800):
		"""API to the 'vioplot' R package: http://cran.r-project.org/web/packages/vioplot/index.html"""
		if not (x and y):
			self._call_remaining('vioplot',x=x,y=y)
			return
		
		if fn==None:
			fn='vioplot.'+self._get_fn(x,y)+'.png'
			
		vectors=self.toVectors(x,y)

		importr('vioplot')
		
		grdevices.png(file=fn, width=w, height=h)
		vvals=vectors.values()
		vkeys=vectors.keys()

		r['vioplot'](*vvals,**{'names':vkeys,'col':'gold'})
		r['title']( 'Violin (box+density) plot where y='+y+' and x='+x )
		
		grdevices.dev_off()
		print ">> saved: "+fn
		




	def summary(self,obj=None):
		if not obj:
			obj=self.df
		x=r['summary'](obj)
		if self.toprint:
			print x
		return x
		
	def xtabs(self,cols=[]):
		frmla='~'+'+'.join(cols)
		return r['xtabs'](frmla,data=self.df)

	def ca(self,fn,cols=[]):
		importr('ca')
		fit=r['ca'](self.xtabs(cols))
		if self.toprint:
			print r['summary'](fit)
		
		r_plot(fn,fit)
		return fit
	
	
	def pca(self,fn='pca.png',col=None,w=1200,h=1200,title=''):
		stats    = importr('stats')
		graphics = importr('graphics')

		df=self.q().df
		pca = stats.princomp(df)

		grdevices = importr('grDevices')
		ofn=".".join(fn.split(".")[:-1]+["eigens"]+[fn.split(".")[-1]])
		strfacts=str(df.nrow)+" items using "+str(df.ncol)+" features ["+ofn.split("/")[-1]+"]"
		grdevices.png(file=ofn, width=w, height=h)
		graphics.plot(pca, main = title+" [Eigenvalues for "+strfacts+"]")
		# if col:
		# 	graphics.hilight(pca,factors)
		grdevices.dev_off()
		print ">> saved: "+ofn	

		grdevices = importr('grDevices')
		ofn=".".join(fn.split(".")[:-1]+["biplot"]+[fn.split(".")[-1]])
		strfacts=str(df.nrow)+" items using "+str(df.ncol)+" features ["+ofn.split("/")[-1]+"]"
		grdevices.png(file=ofn, width=w, height=h)
		stats.biplot(pca, scale=1,main = title+" [biplot of "+strfacts+"]")
		grdevices.dev_off()
		print ">> saved: "+ofn
	

	def chisq(self,cols=[]):
		fit=r['chisq.test'](self.xtabs(cols))
		if self.toprint:
			print r['summary'](fit)
		return fit

	def _get_frmla(self,formula,joiner='+'):
		if not '~' in formula:
			ykey=formula.strip()
			keys=set(self.df.colnames)
			ykeys=set([ykey])
			xkeys=keys.difference(ykeys)
			return ro.Formula(ykey+" ~ "+joiner.join(xkeys))
		else:
			return ro.Formula(formula)

	# def manova(self,ys=[],xs=[],xjoin='*'):
	# 	Y=r['cbind'](ys)
	# 	#frmla=self._get_frmla(Y+'~'+xjoin.join(xs),joiner='*')
	# 	frmla=ro.Formula(Y+'~'+xjoin.join(xs))
	# 	
	# 	print frmla
	# 	fit=r['manova'](frmla,data=self.df)
	# 	
	# 	if self.toprint:
	# 		print r['summary'](fit)
	# 	
	# 	return fit

	def aov(self, formula, tukey=False, plot=False, fn=None, w=1100, h=800):
		frmla=self._get_frmla(formula,joiner='*')
		fit=r['aov'](frmla,data=self.df)
		
		if tukey:
			tfit=r['TukeyHSD'](fit)
		
		
		if self.toprint:
			print r['summary'](fit)
			if tukey:
				print r['summary'](tfit)
		
		if plot:
			if not fn:
				fn = 'aov.'+str(formula)+'.png'
			grdevices.png(file=fn, width=w, height=h)

			r('layout(matrix(c(1,2,3,4),2,2))') # optional layout 
			r['plot'](tfit) # diagnostic plots

			grdevices.dev_off()
			print ">> saved: "+fn
			
			
		return fit
	
	def polyplot(self,terms):
		pass
	
	def addRowCol(self,name='rowcol'):
		self.addCol(name,self.rows)
		
	
	def addCol(self,name,vals):
		dl=self.toDL()
		#try:
		if type(vals[0])!=type(''):
			dl[name]=ro.FloatVector(vals)
		#except:
		else:
			dl[name]=ro.StrVector(vals)
			if self.factor:
				dl[name]=ro.FactorVector(dl[name])
		
		self.rownames=self.rows
		self._boot_DL(dl,rownames=True)
	
	def removeCol(self,name):
		dl=self.toDL()
		del dl[name]
		self.rownames=self.rows
		self._boot_DL(dl,rownames=True)
		
	def group(self,x=None,ys=[],yname='y',ytype='y_type',otherattrs=[]):
		ld=[]
		if not ys:
			if x:
				ys=[i for i in range(len(self.cols)) if self.cols[i]!=x]
			else:
				ys=self.cols
		
		if type(ys[0])!=type(0):
			ys=[self.cols.index(yk) for yk in ys]
		
		gk=(x,tuple(ys))
		
		try:
			return self._groupv[gk]
		except KeyError:
			pass
		xk='row'
		if x:
			xi=self.cols.index(x)
		for row in self.rows:
			if x:
				xv=self.row(row)[xi]
			else:
				xv=row
			
			if type(xv)==type('') and xv.isdigit():
				xv=float(xv)
			
			for yi in ys:
				d={}
				d[xk]=xv
				d[yname]=self.row(row)[yi]
				d[ytype]=self.cols[yi]
				for oa in otherattrs:
					try:
						d[oa]=self.row(row)[self.cols.index(oa)]
					except:
						pass
				ld.append(d)

		self._groupv[gk]=RpyD2(ld)
		return self._groupv[gk]
	
	def sub_where(self,rows={},removeCol=False):
		r=self.sub(rows=self.rows_where(rows))
		if removeCol:
			for k in rows: r.removeCol(k)
		return r
	
	def polyfits(self,x,y,degs,addCol=True,fn=None,plot=True,onlyBest=False):
		ldn=[]
		if fn==None:
			fn='polyfit.'+self._get_fn(x,y)+'.png'
		for i in degs:
			fit=self.polyfit(x,y,i,addCol=addCol)
			dd={}
			dd['deg']=i
			dd['sum_residuals']=sum(fit['ryanresid'])
			ldn.append(dd)

		rn=RpyD2(ldn)
		if plot: rn.plot('polyfit.'+fn.replace('.png','.residuals.png'), x='deg',y='sum_residuals',line=True,point=True,smooth=False)
		
		if addCol:	# only works if column appended
			if onlyBest:
				res=rn.col('sum_residuals')
				lres=None
				besti=0
				for i in range(len(res)):
					if not lres:
						lres=res[i]
						continue
					if res[i]>=lres:
						break
					#if res[i]<lres:
					besti=i
					lres=res[i]
				best=str(int(rn.col('deg')[besti]))
				#rg=self.group('cnum',['polyfit_'+best.zfill(2),'polyfit_'+best.zfill(2)+'_1drv',y])
				rg=self.group('cnum',['polyfit_'+best.zfill(2),y])
			else:
				rg=self.group('cnum')
			if plot: rg.plot('polyfit.'+fn, x='row', y='y', col='y_type',group='y_type',line=True,point=True,smooth=False)
			return [ (c,self.col(c)) for c in self.cols if c.startswith('polyfit_'+best.zfill(2)) ]
	
	
	def polyfit(self,x,y,deg=3,addCol=True,addDer=True):
		import numpy as np
		#print ">> fitting word: "+word

		xs=self.col(x)
		ys=self.col(y)
		x=np.array(xs)
		y=np.array(ys)

		fit=np.polyfit(x,y,deg,full=True)
		f=np.poly1d(fit[0])
		f2=np.polyder(f)
		if addCol:
			self.addCol('polyfit_'+str(deg).zfill(2),[f(x) for x in xs])
			self.addCol('polyfit_'+str(deg).zfill(2)+'_1drv',[f2(x) for x in xs])
		
		fitd={}
		fitd['coeff']=fit[0]
		fitd['resid']=fit[1]
		fitd['ryanresid']=[ abs(ys[i]-f(xs[i])) for i in range(len(ys)) ]
		fitd['rank']=fit[2]
		fitd['singval']=fit[3]
		fitd['rcond']=fit[4]
		return fitd

	def rows_where(self,qdict):
		rowsIncl=[]
		
		qdict2={}
		for k,v in qdict.items():
			try:
				if type(k)!=type(0):
					colnum=self.cols.index(k)
				else:
					colnum=k
				qdict2[colnum]=v
			except:
				continue
		if not qdict2: return
		for row in self.rows:
			rowdat=self.row(row)
			include=True
			for k,v in qdict2.items():
				if type(v)==type([]):
					if not rowdat[k] in v: include=False
				else:
					if rowdat[k]!=v: include=False
			if include:
				rowsIncl.append(row)
		return rowsIncl


	def lm(self,formula,toprint=False):
		frmla=self._get_frmla(formula)
		fit=r['lm'](frmla,data=self.df)

		if self.toprint:
			print r['summary'](fit)
		return fit
		

	def loess(self,formula,toprint=True):
		frmla=self._get_frmla(formula)
		fit=r['loess'](frmla,data=self.df)
		if self.toprint:
			print r['summary'](fit)
		return fit

	def glm(self,ykey='y',family='gaussian',anovaTest='Chisq'):
		"""
		API to R's glm:
			http://web.njit.edu/all_topics/Prog_Lang_Docs/html/library/base/html/glm.html
		
		Family can be:
			[ref: http://web.njit.edu/all_topics/Prog_Lang_Docs/html/library/base/html/family.html]
			
			
		"""


		keys=set(self.df.colnames)
		ykeys=set([ykey])
		xkeys=keys.difference(ykeys)

		#lm=r['lsfit'](dataframe(ld,xkeys),dataframe(ld,ykeys))

		#df=dataframe(ld)
		frmla=ykey+" ~ "+"+".join(xkeys)
		lm=r['glm'](formula=frmla,data=self.df,family=family)
		#return "\n\n".join(str(x) for x in [ r['anova'](lm,test=anovaTest) ] )
		anova=r['anova'](lm,test=anovaTest)
		return anova
		# try:
		# 			import werwerwe
		# 			from rpy2.robjects.packages import importr
		# 			rjson=importr('rjson')
		# 			import json
		# 			return json.loads(str(r['toJSON'](anova)[0]))
		# 
		# 		except:
		# 			return anova
		
		#r['ls.print'](lm)

	
	
	## DISTANCE MEASURES

	def corrgram(self,fn=None,w=1600,h=1600,title=''):
		"""API to corrgram package:
		"""
		importr('corrgram')
		if not fn:
			fn='corrgram.png'
		grdevices.png(file=fn, width=w, height=h)
		r['corrgram'](self.q().df,lower_panel='panel.shade',upper_panel='panel.pts',order=True,main=title)
		grdevices.dev_off()
		print ">> saved: "+fn

	def cormean(self):
		c=self.cor(returnType='r')
		#print c.mean_stdev()[0]
		return c.mean_stdev()[0]
		#return mean_stdev([x for x in c.flatten() if x > 0])[0]
	
	def cormedian(self):
		c=self.cor(returnType='r')
		#print c.median()
		return c.median()

	def dist(self,z=False,cor=False,returnType=None):
		if not cor:
			x=r['dist'](self.q(z).df)
		else:
			x=self.cordist()
		
		if returnType is None:
			return x
		elif returnType.startswith('d'):
			return r['as.data.frame'](r['as.matrix'](x))
		elif returnType.startswith('r'):
			return RpyD2(r['as.data.frame'](r['as.matrix'](x)))
			
	def corcol(self,l,threshold_pp=0.05):
		if len(l)!=len(self.rows):
			print "!! error: row number mismatch. You gave me a list with",len(l),"items, while I am:\n\t",repr(self)
			return
		
		cors=[]
		for col in self.cols:
			pr,pp=correlate(l,self.col(col))
			if pp>threshold_pp: continue
			cors.append( (col,pr) )
		return cors
	
	# def corgraph(self,fn=None,pr=None):
	# 		if not pr: pr=signcorr(len(self.rows))
	# 		print ">> PR:",pr
	# 		
	# 		corr=self.cor(returnType='r')
	# 		print corr
	
	def corgraph(self,fn='',threshold_pr=None,threshold_pp=0.01,plot=True,plotlim=20,txtlim=1000,justReturn=False,storeDataInGraph=False):
		#if not threshold_pr: threshold_pr=signcorr(len(self.rows))
		
		if fn: fn+='.'
		fn+='corgraph.pdf'
		
		import networkx as nx
		import pystats
		G=nx.Graph()
		for word1 in sorted(self.cols):
			print word1,"..."
			for word2 in sorted(self.cols):
				if word2<=word1: continue
				#print ">>",word1,word2,"...",
				try:
					pr,pp=pystats.correlate(self.col(word1),self.col(word2))
				except:
				#	print
					continue
				if pp>threshold_pp or pr<0.0:
				#	print
					continue
				#print pr,pp,"!"
				G.add_edge(word1,word2,weight=(pr,pp))
		#pyd=nx.to_pydot(G)
		
		if storeDataInGraph:
			for node in G.nodes():
				G.node[node]['data']=self.col(node)
				try:
					G.node[node]['key']=self.colkey[node]
				except KeyError:
					pass
		
		if justReturn: return G
		
		o=""
		o+="\nSIZE: "+str(G.size())
		o+="\nORDER: "+str(G.order())
		o+="\nDENSITY: "+str(nx.density(G))
		o+="\nCLUSTCOEF: "+str(nx.transitivity(G))

		print o
		
		nx.write_edgelist(G,fn.replace('.pdf','.edgelist.txt'))
		print ">> saved:",fn.replace('.pdf','.edgelist.txt')
		
		
		o+="\n\n"
		i=0
		for k,v in sorted(nx.degree_centrality(G).items(),key=lambda item: -item[1]):
			o+="\n"+str(k)+"\t"+str(v)
			if k in self.colkey: o+="\t"+str(self.colkey[k])
			i+=1
			fnp='corgraph.plot1.'+k+'.'+str(v)+'.png'
			if plot:
				#if plotlim and i>plotlim: continue
				self.group(ys=[k]).plot(fn=fnp,x='row',y='y',col='y_type',group='y_type',point=True,size=4,line=True,title=self.colkey[k])
			

		
		o+="\n\n"
		
		i=0	
		numedges=G.size()
		for a,b in sorted(nx.edges(G),key=lambda item: G[item[0]][item[1]]['weight'][1]):
			
			ox=""
			w=G[a][b]['weight']
			if w[0]==1: continue
			if w[1]==0: continue
			i+=1
			
			if txtlim and i>txtlim: break
			
			
			ox+="\n"+str(a)+"\t"+str(b)+"\t"+str(w)
			if a in self.colkey and b in self.colkey:
				ox+="\n"+str(self.colkey[a])
				ox+="\n"+str(self.colkey[b])
				ox+="\n"
			o+=ox
			
			
			if plot:
				if plotlim and i>plotlim: continue
				
				fnp='corgraph.plot.'+ str(i).zfill(len(str(numedges)))+'.'+'.'.join(str(x) for x in [a,b])+'.png'
				self.group(ys=[a,b]).plot(fn=fnp,x='row',y='y',col='y_type',group='y_type',point=True,size=4,line=True,title=ox)
		
		
		
		#pyd.write_pdf(fn)
		#print ">> saved:",fn
		
		write(fn.replace('.pdf','.netstat.txt'),o,toprint=True)
		return G
	
	def clustergram2(self,fn=None,w=800,h=600,kstart=2,kend=10,kstep=1):
		ro.r('''
			library(plyr)
			ks.default <- function(rows) seq(2, max(3, rows %/% 4))

			many_kmeans <- function(x, ks = ks.default(nrow(x)), ...) {
			  ldply(seq_along(ks), function(i) {
			    cl <- kmeans(x, centers = ks[i], ...)
			    data.frame(obs = seq_len(nrow(x)), i = i, k = ks[i], cluster = cl$cluster)
			  })
			}

			all_hclust <- function(x, ks = ks.default(nrow(x)), point.dist = "euclidean", cluster.dist = "ward") {
			  d <- dist(x, method = point.dist)
			  cl <- hclust(d, method = cluster.dist)

			  ldply(seq_along(ks), function(i) {
			    data.frame(
			      obs = seq_len(nrow(x)), i = i, k = ks[i], 
			      cluster = cutree(cl, ks[i])
			    )
			  })  
			}

			center <- function(x) x - mean(range(x))

			#' @param clusters data frame giving cluster assignments as produced by 
			#'   many_kmeans or all_hclust
			#' @param y value to plot on the y-axis.  Should be length
			#'   \code{max(clusters$obs)}
			clustergram <- function(clusters, y, line.width = NULL) {
			  clusters$y <- y[clusters$obs]
			  clusters$center <- ave(clusters$y, clusters$i, clusters$cluster)  

			  if (is.null(line.width)) {
			    line.width <- 0.5 * diff(range(clusters$center, na.rm = TRUE)) / 
			      length(unique(clusters$obs))
			  }
			  clusters$line.width <- line.width

			  # Adjust center positions so that they don't overlap  
			  clusters <- clusters[with(clusters, order(i, center, y, obs)), ]  
			  clusters <- ddply(clusters, c("i", "cluster"), transform, 
			    adj = center + (line.width * center(seq_along(y)))
			  )

			  structure(clusters, 
			    class = c("clustergram", class(clusters)),
			    line.width = line.width)
			}

			plot.clustergram <- function(x) {
			  i_pos <- !duplicated(x$i)

			  means <- ddply(x, c("cluster", "i"), summarise, 
			    min = min(adj), max = max(adj))

			  ggplot(x, aes(i)) +
			    geom_ribbon(aes(y = adj, group = obs, fill = y, ymin = adj - line.width/2, ymax = adj + line.width/2, colour = y)) + 
			    geom_errorbar(aes(ymin = min, ymax = max), data = means, width = 0.1) + 
			    scale_x_continuous("cluster", breaks = x$i[i_pos], labels = x$k[i_pos]) +
			    labs(y = "Cluster average", colour = "Obs\nvalue", fill = "Obs\nvalue")

			}
		''')
		dm=ro.r['as.matrix'](self.q().df)
		ro.globalenv['dm']=dm
		
		if not fn: fn='clustergram.png'
		grdevices.png(file=fn, width=w, height=h)
		
		ro.r('''
			library(ggplot2)
			k_def <- many_kmeans(dm)
			k_10 <- many_kmeans(dm, '''+str(kstart)+''':'''+str(kend)+''')
			k_rep <- many_kmeans(dm, rep(4, 5))
			h_def <- all_hclust(dm)
			h_10 <- all_hclust(dm, 2:10)
			h_5 <- all_hclust(dm, seq(2, 20, by = 4))

			pr <- princomp(dm)
			pr1 <- predict(pr)[, 1]
			pr2 <- predict(pr)[, 2]

			plot(clustergram(k_def, pr1))
			plot(clustergram(k_rep, pr1))
			plot(clustergram(k_rep, pr2))
		
			pr <- princomp(dm)
			pr1 <- predict(pr)[, 1]
			pr2 <- predict(pr)[, 2]
			plot(clustergram(k_def, pr1))
		''')
		"""
		plot(clustergram(k_rep, pr1))
		plot(clustergram(k_rep, pr2))
		
		"""
		grdevices.dev_off()
		print ">> saved:",fn
		
	
	
	
	def clustergram(self,fn=None,w=800,h=600,kstart=2,kend=10,kstep=1,title=''):
		if not title: title='[Clustergram of the PCA-weighted Mean of the clusters k-mean clusters vs number of clusters (k)]'
		ro.r('''
			clustergram.kmeans <- function(Data, k, ...)
			{
				# this is the type of function that the clustergram
				# 	function takes for the clustering.
				# 	using similar structure will allow implementation of different clustering algorithms

				#	It returns a list with two elements:
				#	cluster = a vector of length of n (the number of subjects/items)
				#				indicating to which cluster each item belongs.
				#	centers = a k dimensional vector.  Each element is 1 number that represent that cluster
				#				In our case, we are using the weighted mean of the cluster dimensions by 
				#				Using the first component (loading) of the PCA of the Data.

				cl <- kmeans(Data, k,...)

				cluster <- cl$cluster
				centers <- cl$centers %*% princomp(Data)$loadings[,1]	# 1 number per center
															# here we are using the weighted mean for each

				return(list(
							cluster = cluster,
							centers = centers
						))
			}		

			clustergram.plot.matlines <- function(X,Y, k.range, 
														x.range, y.range , COL, 
														add.center.points , centers.points)
				{
					plot(0,0, col = "white", xlim = x.range, ylim = y.range,
						axes = F,
						xlab = "Number of clusters (k)", ylab = "PCA weighted Mean of the clusters", main = "'''+title+'''")
					axis(side =1, at = k.range)
					axis(side =2)
					abline(v = k.range, col = "grey")

					matlines(t(X), t(Y), pch = 19, col = COL, lty = 1, lwd = 1.5)

					if(add.center.points)
					{
						require(plyr)

						xx <- ldply(centers.points, rbind)
						points(xx$y~xx$x, pch = 19, col = "red", cex = 1.3)

						# add points	
						# temp <- l_ply(centers.points, function(xx) {
												# with(xx,points(y~x, pch = 19, col = "red", cex = 1.3))
												# points(xx$y~xx$x, pch = 19, col = "red", cex = 1.3)
												# return(1)
												# })
									# We assign the lapply to a variable (temp) only to suppress the lapply "NULL" output
					}	
				}



			clustergram <- function(Data, k.range = seq('''+str(kstart)+''','''+str(kend)+''','''+str(kstep)+'''),
										clustering.function = clustergram.kmeans,
										clustergram.plot = clustergram.plot.matlines, 
										line.width = .004, add.center.points = T)
			{
				# Data - should be a scales matrix.  Where each column belongs to a different dimension of the observations
				# k.range - is a vector with the number of clusters to plot the clustergram for
				# clustering.function - this is not really used, but offers a bases to later extend the function to other algorithms 
				#			Although that would  more work on the code
				# line.width - is the amount to lift each line in the plot so they won't superimpose eachother
				# add.center.points - just assures that we want to plot points of the cluster means

				n <- dim(Data)[1]

				PCA.1 <- Data %*% princomp(Data)$loadings[,1]	# first principal component of our data

				if(require(colorspace)) {
						COL <- heat_hcl(n)[order(PCA.1)]	# line colors
					} else {
						COL <- rainbow(n)[order(PCA.1)]	# line colors
						warning('Please consider installing the package "colorspace" for prittier colors')
					}

				line.width <- rep(line.width, n)

				Y <- NULL	# Y matrix
				X <- NULL	# X matrix

				centers.points <- list()

				for(k in k.range)
				{
					k.clusters <- clustering.function(Data, k)

					clusters.vec <- k.clusters$cluster
						# the.centers <- apply(cl$centers,1, mean)
					the.centers <- k.clusters$centers 

					noise <- unlist(tapply(line.width, clusters.vec, cumsum))[order(seq_along(clusters.vec)[order(clusters.vec)])]	
					# noise <- noise - mean(range(noise))
					y <- the.centers[clusters.vec] + noise
					Y <- cbind(Y, y)
					x <- rep(k, length(y))
					X <- cbind(X, x)

					centers.points[[k]] <- data.frame(y = the.centers , x = rep(k , k))	
				#	points(the.centers ~ rep(k , k), pch = 19, col = "red", cex = 1.5)
				}


				x.range <- range(k.range)
				y.range <- range(PCA.1)

				clustergram.plot(X,Y, k.range, 
														x.range, y.range , COL, 
														add.center.points , centers.points)


			}
			''')
		clustergram=ro.r['clustergram']
		if not fn: fn='clustergram.png'
		grdevices.png(file=fn, width=w, height=h)
		clustergram(r['as.matrix'](self.q().df))
		grdevices.dev_off()
		print ">> saved:",fn
		

	def corclust(self,fn=None,pr=None,plot=True,rankfunc=lowerfifth,kstart=2,kend=None,kstep=1,k=None):
		if not pr: pr=signcorr(len(self.rows))
		print ">> PR:",pr
		
		
		if k: return self.kclust(k=k,cor=True,rsplit=True,plot=False)
			
		
		if k:
			kstart=k
			kend=k+1
			kstep=1
		else:
			if not kend: kend=len(self.cols)
		
		for k in range(kstart,kend,kstep):
			print ">> trying:",k
			rs=self.kclust(k=k,cor=True,rsplit=True,plot=False)
			success=True
			for r in rs.values():
				#prnow=r.cormedian()
				prnow=rankfunc(r.cor(returnType='r').flatten())
				print prnow, rankfunc.__name__,
				if prnow<pr:
					success=False
					print 'X!'
					#break
				else:
					print '...'
			if success:
				if plot: self.kclust(fn=fn,k=k,cor=True,rsplit=False,plot=True)
				break
		return rs
			

	def kclust(self,k=4,rsplit=False,cor=False,z=True,plot=True,fn=None,w=1100,h=800,title=''):
		""" Currently set to return self.pam(k) for robust k-means clustering."""
		fit=self.pam(k=k,z=z,cor=cor)
		
		if plot:
			importr('cluster')
			if not fn: fn='kclust'
			if not '.png' in fn: fn+='.'+str(k).zfill(2)+'.z'+str(z)+'.png'
			grdevices.png(file=fn, width=w, height=h)
			if not title: title=fn
			r['clusplot'](fit,color=True, shade=True, labels=2, lines=0, main=title)
			grdevices.dev_off()
			print ">> saved: "+fn



		if rsplit:
			clusterDF=list(fit[2])
			groups={}
			for i in range(len(clusterDF)):
				cnum=clusterDF[i]
				if cor:
					ckey=self.cols[i]
				else:
					ckey=self.rows[i]
				if not cnum in groups:
					groups[cnum]=[]
				groups[cnum]+=[ckey]
						
			for cnum in groups:
				if cor:
					groups[cnum]=self.sub(cols=groups[cnum])
				else:
					groups[cnum]=self.sub(rows=groups[cnum])
			return groups
			
		
		return fit
		
		
		
	def pam(self,k=4,z=True,cor=False):
		"""API to R's pam function: 
			http://stat.ethz.ch/R-manual/R-patched/library/cluster/html/pam.html
		A more robust version of k-means clustering, 'around medoids.'
		"""
		importr('cluster')
		return r['pam'](self.q(z=z).dist(cor=cor),k)


	def kmeans(self,k=4):
		"""API to R's kmeans clustering function: http://stat.ethz.ch/R-manual/R-patched/library/stats/html/kmeans.html"""
		fit=r['kmeans'](self.q().df, k)
		if self.toprint:
			print fit	
		
		return fit

	def cor(self,returnType='rpyd2'):	# returnType options: None, 'df'/'dataframe', 'rpyd2'
		x=r['cor'](self.q().df)
		if not returnType: return x
		df=r['as.data.frame'](x)
		
		if returnType.startswith('d'):
			return df
		return RpyD2(df)
		
	def cordist(self):
		c=self.cor(returnType=None)
		ro.globalenv['c']=c
		return r('as.dist((1 - c)/2)')
	
	
	def distro(self,fn=None,rankfunc=None):
		cols,sums=self.rankcols(returnSums=True,rankfunc=rankfunc)
		lc=len(str(len(cols)))
		
		if not rankfunc:
			sumk='sum'
		else:
			sumk=rankfunc.__name__
		
		dl={}
		dl[sumk]=sums
		dl['cols']=['r'+str(i).zfill(lc)+cols[i] for i in range(len(cols))]
		
		r=RpyD2(dl)
		if not fn: fn='distro.png'
		r.plot(fn=fn.replace('.png','.cols.png'),x=sumk,y='cols',point=True,smooth=False)
		r.plot(fn=fn.replace('.png','.density.png'),x=sumk,density=True)
		
		return r
	
	
	def csv(self,fn='csv.txt',sep='\t'):
		self.df.to_csvfile(fn,sep=sep)
		print ">> saved: "+fn
	

	def hclust(self,cor=False,z=True,plot=True,fn=None,w=1100,h=900,title=''):
		dist=self.dist(z=z,cor=cor)
		
		if not fn: fn='hclust.png'
		hclust=r['hclust'](dist)
		grdevices.png(file=fn, width=w, height=h)
		r['plot'](hclust,main=title)
		grdevices.dev_off()
		print ">> saved: "+fn

		return hclust

	def treepredict(self,y='',fn='treepredict.png',w=1100,h=800):
		importr('rpart')
		grdevices.png(file=fn, width=w, height=h)
		frmla=y+'~'+'+'.join([c for c in self.q().cols if c!=y])
		fit=r['rpart'](ro.Formula(frmla),method="class",data=self.df)
		#r['plot'](r['printcp'](fit)) # display the results


		# plot tree 

		r['plot'](fit, uniform=True,main=frmla)
		r['text'](fit, use_n=True, all=True, cex=.8)
		grdevices.dev_off()
		print ">> saved: "+fn



	def pvclust(self,z=True,fn='pvclust.png',w=1100,h=900):
		"""API to R package pvclust: http://cran.r-project.org/web/packages/pvclust/index.html"""
		importr('pvclust')
		grdevices.png(file=fn, width=w, height=h)
		fit=r['pvclust'](self.q(z=z).df, method_hclust="ward",method_dist="euclidean")
		r['plot'](fit)
		r['pvrect'](fit, alpha=0.95)
		grdevices.dev_off()
		print ">> saved: "+fn
		return fit
	
	def predict(self,y='',z=True,fn='predict.png',w=1100,h=800):
		"""API to pamr.train and pamr.predict:
			http://www-stat.stanford.edu/~tibs/PAM/Rdist/pamr.train.html
			http://rgm2.lab.nig.ac.jp/RGM2/func.php?rd_id=pamr:pamr.predict
			"""
		importr('pamr')
		grdevices.png(file=fn, width=w, height=h)
		y=ro.FactorVector(self.col(y))
		x=self.q(z=z).df
		data=r['list'](x=x,y=y)
		train=r['pamr.train'](data)
		print train
		cv=r['pamr.cv'](train,data)
		fit=r['pamr.predict'](train, x, threshold=1)
		grdevices.dev_off()
		print ">> saved: "+fn
		return fit
	
	
	def mclust(self,z=True,fn='mclust.png',w=1100,h=900):
		
		importr('mclust')
		grdevices.png(file=fn, width=w, height=h)
		

		# plot(fit, mydata) # plot results 
		# print(fit) # display the best model
		
		fit=r['Mclust'](self.q(z=z).df)

		#r['plot'](fit, self.q(z=z).df)
		r('layout(matrix(c(1,2,3,4),3,3))') # optional layout 
		#r['print'](fit)
		grdevices.dev_off()
		print ">> saved: "+fn
		return fit


	def points_3d(self,fn=None,x='x',y='y',z='z',title=False,w=800,h=800):
		if not fn: fn='3d_points.'+'.'.join([x,y,z])+'.png'
		grdevices.png(file=fn, width=w, height=h)
		r = ro.r
		rgl=importr('rgl')
		#dev=r('rgl.open()')
		
		selfdl=self.toDL()
		dl={}
		dl['x']=selfdl[x]
		dl['y']=selfdl[y]
		dl['z']=selfdl[z]
		rp=RpyD2(dl)
		
		print rp.df
		
		xs=self.col(x)
		ys=self.col(y)
		zs=self.col(z)
		#r('open3d()')
		plot=r['points3d'](rp.df)
			
		#rprint(plot)
		
		#r['rgl.snapshot'](fn)
		grdevices.dev_off()
		r_plot(fn,plot)
		#print ">> saved: "+fn


	def cloud(self,fn=None,x='x',y='y',z='z',group=None,title=False,w=800,h=800):
		if not fn: fn='cloud.'+'.'.join([x,y,z])+'.png'
		grdevices.png(file=fn, width=w, height=h)
		r = ro.r
		importr('lattice')
		if groups:
			plot=r['cloud'](ro.Formula(z+'~'+y+'+'+x), data = self.df, scales = r('list(arrows = FALSE)'),groups=groups)
		else:
			plot=r['cloud'](ro.Formula(z+'~'+y+'+'+x), data = self.df, scales = r('list(arrows = FALSE)'))
		rprint(plot)
		grdevices.dev_off()
		print ">> saved: "+fn


	def plot3d(self,fn=None,x='x',y='y',z='z',title=False,w=800,h=800):
		if not fn:
			fn='plot3d.'+'.'.join([x,y,z])+'.png'

		grdevices.png(file=fn, width=w, height=h)

		r = ro.r
		importr('lattice')
		#importr('scatterplot3d')
		#s3d=r['scatterplot3d'](ro.FloatVector(self.col(x)),ro.FloatVector(self.col(y)),ro.FloatVector(self.col(z)),type="h",main=title,highlight_3d=True,xlab=x,ylab=y,zlab=z)
		
		plot=r['wireframe'](ro.Formula(z+'~'+y+'*'+x), self.df, colorkey = True)
		rprint(plot)
		#fit=self.lm(z+'~'+y+'+'+x)
		#s3d.plane3d(fit)

		grdevices.dev_off()
		print ">> saved: "+fn

	def flatten(self,cols=[],rows=[]):
		if rows or cols:
			x=self.sub(cols=cols,rows=rows)
		else:
			x=self
		data=[]
		for icol in range(self.ncol):
			data.extend(self.col(icol))
		return data

	def median(self,cols=[],rows=[]):
		return median(self.flatten(cols=cols,rows=rows))	

	def mean_stdev(self,cols=[],rows=[]):
		return mean_stdev(self.flatten(cols=cols,rows=rows))

	def str_flot(self,x,y=[]):
		"""
		{% for word in wordtfs %}
			"{{ word }}": {
			label: "{{ word}}",
			data: [ {% for year,value in wordtfs[word].items() %} [{{ year }}, {{ value }}], {% endfor %} ]
		},
		{% endfor %}
		"""
		
		o=[]
		index=self.col(x)
		
		for col in self.rankcols():
			if col==x: continue
			if y and (not col in y): continue
			datstr=[(int(index[i]), cx) for i,cx in enumerate(self.col(col))]
			o+=['''
			"'''+col+'''": {
				label: "'''+col+'''",
				data: ['''+', '.join([ '['+str(dx[0])+', '+str(dx[1])+']' for dx in datstr])+''']
			}''']
		return ', \n'.join(o)
	





def trimCols(ld,cols,maxcols,rank=True,byVariance=True,byFrequency=False,printStats=True,rowcols=[]):
	def rank(l):
		if len(l)<2: return 0
		if byVariance or not byFrequency:
			return sum([abs(x) for x in zfy(l)])
		else:
			return sum(l)

	def rankcol(col):
		return (rank([d[col] for d in ld if col in d]), col)

	sofar=[]
	for col in cols:
		if col in rowcols: continue
		colrank=rankcol(col)
		if len(sofar)<maxcols:
			sofar+=[colrank]
			continue

		sofar.sort(reverse=True)
		if sofar[-1][0]<colrank[0]:
			sofar+=[colrank]
			sofar=sorted(sofar,reverse=True)[:maxcols]
	sofar=sorted(sofar,reverse=True)[:maxcols]
	allowedkeys=[x[1] for x in sofar]
	print allowedkeys
	cols=[]
	for i in range(len(allowedkeys)):
		if rank:
			k='r'+str(i+1).zfill(len(str(len(allowedkeys))))+'.'+allowedkeys[i]
		cols.append(k)

	for col in rowcols:
		print col
		cols.append(col)

	return cols





def getCols(ld,allcols=False,rownamecol=None):
	""" Returns a list of columns which,
		if allcols=False, returns only columns which all rows share;
		if allcols"""
	cols=[]
	for row in ld:
		if not type(row)==type({}): continue
		if not len(row.keys()): continue

		if not len(cols):
			cols=row.keys()
			continue

		if allcols:
			for x in set(row.keys())-set(cols):
				cols.append(x)
		else:
			for x in set(cols)-set(row.keys()):
				cols.remove(x)


	if rownamecol:
		if rownamecol in cols:
			cols.remove(rownamecol)

	return cols


def mean_stdev(x):
	try:
		import numpy as np
		return np.mean(x), np.std(x)
	except:
		from math import sqrt
		n, mean, std = len(x), 0, 0
		for a in x:
			mean = mean + a
			mean = mean / float(n)
		for a in x:
			std = std + (a - mean)**2
			std = sqrt(std / float(n-1))
		return mean, std




def r_plot(fn,obj,w=800,h=800,xlab='',ylab='',main=''):
	grdevices.png(file=fn, width=w, height=h)

	r.plot(obj,xlab=xlab,ylab=ylab,main=main)

	grdevices.dev_off()
	print ">> saved: "+fn

	
		

def zfy(tfdict,limit=None):
	if type(tfdict)==type({}):
		mean,stdev=mean_stdev(tfdict.values())
		zdictz={}
		for k,v in tfdict.items():
			if not stdev:
				zdictz[k]=0
			else:
				zdictz[k]=(v-mean)/stdev
			if limit:
				if zdictz[k]>limit:
					del zdictz[k]
		
		return zdictz
	elif type(tfdict)==type([]):
		mean,stdev=mean_stdev(tfdict)
		zdictz=[]
		for v in tfdict:
			if not stdev:
				score=0
			else:
				score=(v-mean)/stdev
			if limit and score>limit:
				continue
			zdictz+=[score]
		return zdictz
	else:
		return tfdict



"""
TODO:
	t-test
		http://www.statmethods.net/stats/ttest.html
		http://stat.ethz.ch/R-manual/R-patched/library/stats/html/t.test.html
		
	hist/density
		http://www.statmethods.net/graphs/density.html
		
	


"""
