"""
RpyD2
depends:
	rpy2  <http://rpy.sourceforge.net/rpy2.html>
"""
from rpy2 import robjects as ro
r = ro.r
from rpy2.robjects.packages import importr
grdevices = importr('grDevices')
rprint = ro.globalenv.get("print")

class InputNotRecognizedError(Exception):
	pass

def load(fn):
	import pickle
	d,df=pickle.load(open(fn))
	r=RpyD2([])
	r.__dict__=d
	for k,v in d.items():
		setattr(r,k,v)
	r.df=ro.DataFrame(df)
	return r
	

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
		if not cols and not rows:
			for i in range(self.ncol):
				col=self.col(i)
				colname=self.cols[i]
				dl[colname]=col
			if rownamecol:
				dl['rownamecol']=self.df.rownames
			
		elif cols and not rows:
			for col in cols:
				dl[col]=self.col(col)
			if rownamecol:
				dl['rownamecol']=self.df.rownames
			
		elif cols and rows:
			for col in cols:
				dl[col]=[]
				colnum=self.cols.index(col)
				for row in rows:
					rowdat=self.row(row)
					dl[col].append(rowdat[colnum])
			
			if rownamecol:
				dl['rownamecol']=[]
				for row in rows:
					if type(row)==type(''):
						dl['rownamecol'].append(row)
					else:
						dl['rownamecol'].append(self.rows[row])
				
			
		elif rows and not cols:
			for col in self.cols:
				dl[col]=[]
				colnum=self.cols.index(col)
				for row in rows:
					rowdat=self.row(row)
					dl[col].append(rowdat[colnum])
					
			if rownamecol:
				dl['rownamecol']=[]
				for row in rows:
					if type(row)==type(''):
						dl['rownamecol'].append(row)
					else:
						dl['rownamecol'].append(str(self.rows[row]))

		return dl

	def save(self,fn=None):
		if not fn: 
			import time
			fn=".".join( [ 'rpyd2', time.strftime('%Y%m%d.%H%M.%S', time.gmtime()) , 'pickle' ] )
		import pickle
		pickle.dump((self.__dict__,self.df),open(fn,'wb'))
		print ">> saved:",fn
	
	def sub(self,cols=[],rows=[]):
		"""Return an RpyD2 from self, with only those rows and/or columns as specified."""
		
		if cols and not rows:
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
			m=RpyD2(self.toDL(cols,rows,rownamecol=True),rownamecol='rownamecol')
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
				self.cols=trimCols(ld,self.cols,self.allcols,byVariance=self.trimbyVariance)
		
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
						value=zero
				dd[k].append(value)
		
		self._boot_DL(dd)
	
	def _gen_DF(self,df):
		self._set_DF(df)

	def _boot_DL(self,dl,rownames=None):
		dd={}
		for k,v in dl.items():
			if type(v[0])==type(''):
				if self.onlyQuant: continue
				dd[k]=ro.StrVector(v)
				if self.factor:
					dd[k]=ro.FactorVector(dd[k])
			else:
				if self.z:
					v=zfy(v)
				
				dd[k]=ro.FloatVector(v)		
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

	def rankcols(self,byVariance=False,returnSums=False):
		ranks={}
		for colname in self.cols:
			col=self.col(colname)
			if byVariance:
				rankcol=sum([abs(x) for x in zfy(col)])
			else:
				rankcol=median(col)
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
			sub.plot(fn='plots.'+abk+'.'+self._get_fn(x,"+".join(y[a:b]))+'.png',x='row',y='y',col='y_type',group='y_type',smooth=False,line=True)
			a=b
		return
		

	def plot(self, fn=None, x=None, y=None, col=None, group=None, w=1100, h=800, size=2, smooth=False, point=True, jitter=False, boxplot=False, boxplot2=False, title=False, flip=False, se=False, density=False, line=False, bar=False, xlab_size=14, ylab_size=14):
		
		if fn==None:
			fn='plot.'+self._get_fn(x,y)+'.png'
		df=self.df
		#import math, datetime
		

		grdevices = importr('grDevices')

		if not title:
			title=fn.split("/")[-1]

		import rpy2.robjects.lib.ggplot2 as ggplot2
		
		grdevices.png(file=fn, width=w, height=h)
		if not x:
			if df.rownames[0].isdigit():
				df.x=ro.FloatVector([float(x) for x in list(df.rownames)])
			else:
				df.x=df.rownames
			x='x'

		gp = ggplot2.ggplot(df)
		pp = gp
		
		if x and y:
			if col and group:
				pp+=ggplot2.aes_string(x=x, y=y,col=col,group=group)
			elif col:
				pp+=ggplot2.aes_string(x=x, y=y,col=col)
			elif group:
				pp+=ggplot2.aes_string(x=x, y=y,group=group)
			else:
				pp+=ggplot2.aes_string(x=x, y=y)
		else:
			boxplot=False
			point=False
			boxplot2=False
			smooth=False
			line=False
			bar=False	

		if boxplot:
			if col:
				pp+=ggplot2.geom_boxplot(ggplot2.aes_string(fill=col),color='blue')
			else:
				pp+=ggplot2.geom_boxplot(color='blue')	

		if point:
			if jitter:
				if col:
					pp+=ggplot2.geom_point(ggplot2.aes_string(fill=col,col=col),size=size,position='jitter')
				else:
					pp+=ggplot2.geom_point(size=size,position='jitter')
			else:
				if col:
					pp+=ggplot2.geom_point(ggplot2.aes_string(fill=col,col=col),size=size)
				else:
					pp+=ggplot2.geom_point(size=size)


		if boxplot2:
			if col:
				pp+=ggplot2.geom_boxplot(ggplot2.aes_string(fill=col),color='blue',outlier_colour="NA")
			else:
				pp+=ggplot2.geom_boxplot(color='blue')

		if smooth:
			if smooth=='lm':
				if col:
					pp+=ggplot2.stat_smooth(ggplot2.aes_string(col=col),size=1,method='lm',se=se)
				else:
					pp+=ggplot2.stat_smooth(col='blue',size=1,method='lm',se=se)
			else:
				if col:
					pp+=ggplot2.stat_smooth(ggplot2.aes_string(col=col),size=1,se=se)
				else:
					pp+=ggplot2.stat_smooth(col='blue',size=1,se=se)

		if density:
			if density=='h':
				pp+=ggplot2.geom_histogram(ggplot2.aes_string(x=x,y='..count..'))
			else:
				if col:
					pp+=ggplot2.geom_density(ggplot2.aes_string(x=x,y='..count..',fill=col,alpha=0.2))
				else:
					pp+=ggplot2.geom_density(ggplot2.aes_string(x=x,y='..count..'))

		if line:
			if jitter:
				pp+=ggplot2.geom_line(position='jitter')
			else:
				pp+=ggplot2.geom_line()
		
		if bar:
			pp+=ggplot2.geom_area(ggplot2.aes_string(x=x,y=y,fill=col))
	


		pp+=ggplot2.opts(**{'title' : title, 'axis.text.x': ggplot2.theme_text(size=xlab_size), 'axis.text.y': ggplot2.theme_text(size=ylab_size,hjust=1)} )
		#pp+=ggplot2.scale_colour_brewer(palette="Set1")
		pp+=ggplot2.scale_colour_hue()
		if flip:
			pp+=ggplot2.coord_flip()



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
		
		
	def _call_remaining(self,function_name,x=None,y=None):
		function=getattr(self,function_name)
		if y and not x:
			for col in self.cols:
				if col==y: continue
				try:
					function(x=col,y=y)
				except:
					pass
			return
		elif x and not y:
			for col in self.cols:
				if col==x: continue
				try:
					function(x=x,y=col)
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
	
	
	def pca(self,fn='pca.png',col=None,w=1200,h=1200):
		stats    = importr('stats')
		graphics = importr('graphics')

		df=self.q().df
		pca = stats.princomp(df)

		grdevices = importr('grDevices')
		ofn=".".join(fn.split(".")[:-1]+["eigens"]+[fn.split(".")[-1]])
		strfacts=str(df.nrow)+" items using "+str(df.ncol)+" features ["+ofn.split("/")[-1]+"]"
		grdevices.png(file=ofn, width=w, height=h)
		graphics.plot(pca, main = "Eigenvalues for "+strfacts)
		# if col:
		# 	graphics.hilight(pca,factors)
		grdevices.dev_off()
		print ">> saved: "+ofn	

		grdevices = importr('grDevices')
		ofn=".".join(fn.split(".")[:-1]+["biplot"]+[fn.split(".")[-1]])
		strfacts=str(df.nrow)+" items using "+str(df.ncol)+" features ["+ofn.split("/")[-1]+"]"
		grdevices.png(file=ofn, width=w, height=h)
		stats.biplot(pca, scale=1,main = "biplot of "+strfacts)
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
	
	def addCol(self,name,vals):
		dl=self.toDL()
		try:
			dl[name]=ro.FloatVector(vals)
		except:
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
		
	def group(self,x=None,ys=[],yname='y',ytype='y_type'):
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
			
			if xv.isdigit():
				xv=float(xv)
			
			for yi in ys:
				d={}
				d[xk]=xv
				d[yname]=self.row(row)[yi]
				d[ytype]=self.cols[yi]
				ld.append(d)

		self._groupv[gk]=RpyD2(ld)
		return self._groupv[gk]
	
	def sub_where(self,rows={},removeCol=False):
		r=self.sub(rows=self.rows_where(rows))
		if removeCol:
			for k in rows: r.removeCol(k)
		return r
	
	def polyfits(self,x,y,degs,addCol=True,fn=None,onlyBest=False):
		ldn=[]
		if fn==None:
			fn='polyfit.'+self._get_fn(x,y)+'.png'
		for i in degs:
			fit=self.polyfit(x,y,i)
			dd={}
			dd['deg']=i
			dd['sum_residuals']=sum(fit['ryanresid'])
			ldn.append(dd)

		rn=RpyD2(ldn)
		rn.plot('polyfit.'+fn.replace('.png','.residuals.png'), x='deg',y='sum_residuals',line=True,point=True,smooth=False)
		
		if addCol:
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
			rg.plot('polyfit.'+fn, x='cnum', y='y', col='y_type',group='y_type',line=True,point=True,smooth=False)
		
	
	
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
			#self.addCol('polyfit_'+str(deg).zfill(2)+'_1drv',[f2(x) for x in xs])
		
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


	def lm(self,formula,toprint=True):
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

	def corrgram(self,fn=None,w=1600,h=1600):
		"""API to corrgram package:
		"""
		importr('corrgram')
		if not fn:
			fn='corrgram.png'
		grdevices.png(file=fn, width=w, height=h)
		r['corrgram'](self.q().df,lower_panel='panel.shade',upper_panel='panel.pts')
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

	def dist(self,z=False,cor=False):
		if not cor:
			return r['dist'](self.q(z).df)
		else:
			return self.cordist()

	def corclust(self,pr=None):
		if not pr: pr=signcorr(len(self.rows))
		print ">> PR:",pr
		
		for k in range(2,len(self.cols)):
			print ">> trying:",k
			rs=self.kclust(k=k,cor=True,rsplit=True,plot=False)
			success=True
			for r in rs.values():
				#prnow=r.cormedian()
				prnow=lowerq(r.cor(returnType='r').flatten())
				print prnow,
				if prnow<pr:
					success=False
					print 'X...'
					break
				print
			if success:
				break
		return rs
			

	def kclust(self,k=4,rsplit=False,cor=False,z=True,plot=True,fn=None,w=1100,h=800):
		""" Currently set to return self.pam(k) for robust k-means clustering."""
		fit=self.pam(k=k,z=z,cor=cor)
		
		if plot:
			importr('cluster')
			if not fn:
				fn='kclust.'+str(k).zfill(2)+'.z'+str(z)+'.png'
			grdevices.png(file=fn, width=w, height=h)
			r['clusplot'](fit,color=True, shade=True, labels=2, lines=0, main=fn)
			grdevices.dev_off()
			print ">> saved: "+fn



		if rsplit:
			clusterDF=list(fit[2])
			groups={}
			for i in range(len(clusterDF)):
				cnum=clusterDF[i]
				ckey=self.cols[i]
				if not cnum in groups:
					groups[cnum]=[]
				groups[cnum]+=[ckey]
						
			for cnum in groups:
				groups[cnum]=self.sub(cols=groups[cnum])
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
	
	
	def distro(self,fn=None):
		cols,sums=self.rankcols(returnSums=True)
		lc=len(str(len(cols)))
		
		dl={}
		dl['sums']=sums
		dl['cols']=['r'+str(i).zfill(lc)+cols[i] for i in range(len(cols))]
		
		r=RpyD2(dl)
		if not fn: fn='distro.png'
		r.plot(fn=fn.replace('.png','.cols.png'),x='sums',y='cols',point=True,smooth=False)
		r.plot(fn=fn.replace('.png','.density.png'),x='sums',density=True)
		
		return r
	
	
	def csv(self,fn='csv.txt',sep='\t'):
		self.df.to_csvfile(fn,sep=sep)
		print ">> saved: "+fn
	

	def hclust(self,cor=False,z=True,plot=True,fn=None,w=1100,h=900):
		dist=self.dist(z=z,cor=cor)
		
		if not fn: fn='hclust.png'
		hclust=r['hclust'](dist)
		grdevices.png(file=fn, width=w, height=h)
		r['plot'](hclust)
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


	def cloud(self,fn=None,x='x',y='y',z='z',title=False,w=800,h=800):
		if not fn: fn='cloud.'+'.'.join([x,y,z])+'.png'
		grdevices.png(file=fn, width=w, height=h)
		r = ro.r
		importr('lattice')
		plot=r['cloud'](ro.Formula(z+'~'+y+'+'+x), self.df, scales = r('list(arrows = FALSE)'))
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








def trimCols(ld,cols,maxcols,rank=True,byVariance=True,byFrequency=False,printStats=True):
	# if self.origin!='ld':
	# 	raise InputNotRecognizedError("Cannot recognize input of type "+type(self.input))
	# ld=self.input


	keysums={}
	for col in cols:
		keysums[col]=[]

	for x in ld:
		for k in x:
			value=x[k]
			if type(value)==type(float()) or type(value)==type(1):
				try:
					keysums[k]+=[value]
				except KeyError:
					continue
	from operator import itemgetter
	i=0
	allowedkeys=[]

	for k in keysums:
		if len(keysums[k])<=1:
			keysums[k]=0
			continue
		if byVariance or not byFrequency:
			keysums[k]=sum([abs(x) for x in zfy(keysums[k])])
		else:
			keysums[k]=sum(keysums[k])


	sumvariances_df=sum(keysums.values())
	sumvariances=0.0

	for key,score in sorted(keysums.items(),key=itemgetter(1),reverse=True):
		print key,"\t",score
		i+=1
		if i>maxcols:
			break
		sumvariances+=score
		allowedkeys.append(key)
	#print allowedkeys

	cols=[]
	for i in range(len(allowedkeys)):
		if rank:
			k='r'+str(i+1).zfill(len(str(len(allowedkeys))))+'.'+allowedkeys[i]
		cols.append(k)

	#cols=allowedkeys
	#print cols
	if byVariance or not byFrequency:
		Zvariances = "Z-variances"
	else:
		Zvariances = "TermFrequencies"

	if printStats:
		print ">> sum of "+Zvariances+" in dataset:", sumvariances_df
		print ">> sum of "+Zvariances+" loaded:", sumvariances
		print ">> (ratio) sum of loaded "+Zvariances+" / sum of possible:", sumvariances/sumvariances_df
		print ">> # features loaded:", len(cols)
		print ">> (ratio) sum of loaded "+Zvariances+" / # of features loaded:", sumvariances/len(cols)
		print
	#print cols
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

def signcorr(samplesize):
	import numpy as np
	return 1.96/np.sqrt(samplesize-3)


def ndian(numericValues,n=2):
	theValues = sorted(numericValues)
	if len(theValues) % n == 1:
		return theValues[(len(theValues)+1)/n-1]
	else:
		lower = theValues[len(theValues)/n-1]
		upper = theValues[len(theValues)/n]
		return (float(lower + upper)) / n

def median(numericValues):
	return ndian(numericValues,n=2)
	
def lowerq(numericValues):
	return ndian(numericValues,n=4)




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