"""
RpyD2
depends:
	rpy2  <http://rpy.sourceforge.net/rpy2.html>
"""


from rpy2 import robjects as ro
r = ro.r
from rpy2.robjects.packages import importr
import rpy2.robjects.lib.ggplot2 as ggplot2



class InputNotRecognizedError(Exception):
	pass




class RpyD2():
	def __init__(self,input,**kwargs):
		"""
		Keyword arguments will override the following default options:
			self.cols=None				# specify which columns to build from
			self.rownamecol=None		# specify a column name from which row names should be used
			self.allcols=False			# if False, columns limited to those shared among all rows;
										  if True, all columns are chosen;
										  if a positive integer N, columns limited to the 'top' N columns,
											where columns are compared numerically by:
			self.trimbyVariance=True		# if trimbyVariance==True, sum of absolute value of Z-scores across column
											  otherwise, sum of scores across column

			self.rank=True					# if rank==True, append 'r'+ranknum to the top N columns
			self.zero=0.0				# if allcols is True or an integer, what should empty cells be populated with?
			self.z=False				# if True, Z-score all quantitative columns
			self.factor=True			# if True, treat strings as factors
			self.onlyQuant=False		# if True, only build quantitative columns
			self.onlyCat=False			# if True, only build categorical (string) columns
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
		
		## override defaults with
		for k,v in kwargs.items():
			setattr(self,k,v)
		
		## double override with non-keyword
		#self.input=input
		self.df=None
		self.nrow=0
		self.ncol=0
		self._quantv=None
		self._subv={'cols':{},'rows':{},'cols_rows':{}}
		
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
		return "<RpyD2 storing a "+str(self.ncol)+"-by-"+str(self.ncol)+"row "+self.df.__repr__()[1:-1].replace(" - "," @ ")+">"
	
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
	
	def q(self):
		"""Return a version of self of only quantitative columns"""
		if self.onlyQuant:
			return self
		
		if self._quantv!=None:
			return self._quantv
				
		self._quantv=RpyD2(self.toDL(),rownames=self.rownames,onlyQuant=True)
		return self._quantv
	
	
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
				self.cols=self.trimCols(ld,self.cols,self.allcols,byVariance=self.trimbyVariance)
		
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
		
		if self.rownames:
			df.rownames=ro.FactorVector(self.rownames)
			#del self.rownames
		self.df=df
		self.nrow=self.df.nrow
		self.ncol=self.df.ncol
		self.cols=list(self.df.colnames)
		self.rows=list(self.df.rownames)




	def plot(self, fn=None, x='x', y='y', col=None, group=None, w=1100, h=800, size=2, smooth=True, point=True, jitter=False, boxplot=False, boxplot2=False, title=False, flip=False, se=False, density=False, line=False , xlab_size=14, ylab_size=24):
		
		if fn==None:
			fn='plot.'+'.'.join([x,y])+'.png'
		df=self.df
		#import math, datetime
		

		grdevices = importr('grDevices')

		if not title:
			title=fn.split("/")[-1]

		grdevices.png(file=fn, width=w, height=h)
		gp = ggplot2.ggplot(df)
		pp = gp	
		if col and group:
			pp+=ggplot2.aes_string(x=x, y=y,col=col,group=group)
		elif col:
			pp+=ggplot2.aes_string(x=x, y=y,col=col)
		elif group:
			pp+=ggplot2.aes_string(x=x, y=y,group=group)
		else:
			pp+=ggplot2.aes_string(x=x, y=y)	

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
			pp+=ggplot2.geom_density(ggplot2.aes_string(x=x,y='..count..'))

		if line:
			pp+=ggplot2.geom_line(position='jitter')


		pp+=ggplot2.opts(**{'title' : title, 'axis.text.x': ggplot2.theme_text(size=xlab_size), 'axis.text.y': ggplot2.theme_text(size=ylab_size,hjust=1)} )
		#pp+=ggplot2.scale_colour_brewer(palette="Set1")
		pp+=ggplot2.scale_colour_hue()
		if flip:
			pp+=ggplot2.coord_flip()



		pp.plot()
		grdevices.dev_off()
		print ">> saved: "+fn

	def summary(self):
		return r['summary'](self.df)
		
	def xtabs(self,cols=[]):
		frmla='~'+'+'.join(cols)
		return r['xtabs'](frmla,data=self.df)

	def ca(self,fn,cols=[],toprint=True):
		importr('ca')
		fit=r['ca'](self.xtabs(cols))
		if toprint:
			print fit
		
		r_plot(fit,fn)
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
	

	def testIndependence(self,cols=[]):
		return r['chisq.test'](self.xtabs(cols))


	def aov(self, formula, toprint = True):
		fit=r['aov'](formula,data=self.df)
		
		if toprint:
			print r['summary'](fit)
		return fit

	def lm(self,ykey='y',toprint=True):

		try:
			keys=set(self.df.colnames)
			ykeys=set([ykey])
			xkeys=keys.difference(ykeys)
			frmla=ykey+" ~ "+"+".join(xkeys)
			fit=r['lm'](frmla,data=self.df)
		except:
			keys=set(self.q().df.colnames)
			ykeys=set([ykey])
			xkeys=keys.difference(ykeys)
			frmla=ykey+" ~ "+"+".join(xkeys)
			fit=r['lm'](frmla,data=self.q().df)
		if toprint:
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
		if i>allcols:
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




def r_plot(obj,fn,w=800,h=800,xlabel="",ylabel="",label=""):
	from rpy2.robjects.packages import importr
	from rpy2.robjects import r
	grdevices = importr('grDevices')
	grdevices.png(file=fn, width=w, height=h)

	r.plot(obj)

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
