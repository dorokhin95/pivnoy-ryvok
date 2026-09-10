"""Original sculpted, weighted character assets. Python + NumPy + SciPy; no external models.
Surface nets extract a continuous skin from sculpt fields. Garments are fused before meshing.
Run from this folder to rebuild ../characters-data.js and the editable mesh metadata.
"""
import numpy as np, json, base64, struct
from pathlib import Path
from scipy.ndimage import map_coordinates
OUT=Path(__file__).resolve().parent.parent
BONES=np.array([[0,.94,0],[0,1.30,0],[0,1.72,0],[-.35,1.48,0],[-.51,1.17,.015],[-.63,.91,.03],[.35,1.48,0],[.51,1.17,.015],[.63,.91,.03],[-.18,.94,0],[-.18,.51,.015],[-.18,.14,.025],[.18,.94,0],[.18,.51,.015],[.18,.14,.025]],dtype='f4')
PARENTS=[-1,0,1,1,3,4,1,6,7,0,9,10,0,12,13]

def sm(a,b,k=.045):
 h=np.maximum(k-np.abs(a-b),0)/k
 return np.minimum(a,b)-h*h*k*.25

def ell(p,c,r):return (np.sqrt(np.sum(((p-np.array(c))/np.array(r))**2,axis=-1))-1)*min(r)
def cap(p,a,b,r1,r2=None):
 a=np.array(a);b=np.array(b);d=b-a;t=np.clip(np.sum((p-a)*d,axis=-1)/np.sum(d*d),0,1)
 r=r1 if r2 is None else r1+(r2-r1)*t
 return np.sqrt(np.sum((p-a-t[...,None]*d)**2,axis=-1))-r

def union(*fs,k=.045):
 a=fs[0]
 for b in fs[1:]:a=sm(a,b,k)
 return a

def mesh_field(fn,bounds,step):
 lo=np.array(bounds[0]);hi=np.array(bounds[1]);axes=[np.arange(lo[i],hi[i]+step,step) for i in range(3)]
 p=np.stack(np.meshgrid(*axes,indexing='ij'),-1).astype('f4');f=fn(p).astype('f4');shape=np.array(f.shape)-1
 corners=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])
 vals=np.stack([f[a:a+shape[0],b:b+shape[1],c:c+shape[2]] for a,b,c in corners],-1)
 active=(vals.min(-1)<0)&(vals.max(-1)>=0);cells=np.argwhere(active);v=vals[active];sums=np.zeros((len(cells),3));counts=np.zeros(len(cells))
 edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.sum(abs(corners[i]-corners[j]))==1]
 for i,j in edges:
  use=(v[:,i]<0)!=(v[:,j]<0);t=np.divide(v[:,i],v[:,i]-v[:,j],out=np.zeros(len(v)),where=abs(v[:,i]-v[:,j])>1e-12)
  sums[use]+=corners[i]+t[use,None]*(corners[j]-corners[i]);counts[use]+=1
 points=(cells+sums/counts[:,None])*step+lo
 ids=np.full(tuple(shape),-1,dtype='i4');ids[tuple(cells.T)]=np.arange(len(cells));faces=[]
 for dim in range(3):
  sl0=[slice(None)]*3;sl1=sl0.copy();sl0[dim]=slice(None,-1);sl1[dim]=slice(1,None)
  crossing=(f[tuple(sl0)]<0)!=(f[tuple(sl1)]<0);loc=np.argwhere(crossing)
  dims=[k for k in range(3) if k!=dim];valid=np.ones(len(loc),bool)
  for d in dims:valid&=(loc[:,d]>0)&(loc[:,d]<shape[d])
  loc=loc[valid];q=[]
  for u,w in [(0,0),(-1,0),(-1,-1),(0,-1)]:
   cc=loc.copy();cc[:,dims[0]]+=u;cc[:,dims[1]]+=w;q.append(ids[tuple(cc.T)])
  q=np.stack(q,-1);q=q[np.all(q>=0,axis=1)];faces.extend([q[:,[0,1,2]],q[:,[0,2,3]]])
 faces=np.concatenate(faces);grad=np.gradient(f,step);coords=((points-lo)/step).T
 normals=np.stack([map_coordinates(g,coords,order=1,mode='nearest') for g in grad],-1);normals/=np.maximum(1e-9,np.linalg.norm(normals,axis=1,keepdims=True))
 cross=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]]);flip=np.sum(cross*normals[faces].mean(1),axis=1)<0;faces[flip]=faces[flip][:,[0,2,1]]
 return points.astype('f4'),normals.astype('f4'),faces.astype('i4')

def weights(p,part):
 n=len(p);w=np.zeros((n,15));x,y=p[:,0],p[:,1];a=np.abs(x)
 if isinstance(part,int):w[:,part]=1
 elif part=='coat':
  arm=np.clip((a-.255)/.115,0,1);arm=arm*arm*(3-2*arm)
  chest=np.clip((y-.94)/.28,0,1);w[:,0]=(1-arm)*(1-chest);w[:,1]=(1-arm)*chest
  low=np.clip((1.26-y)/.20,0,1);low=low*low*(3-2*low)
  for side,up,fore in [(-1,3,4),(1,6,7)]:
   sel=x*side>=0;w[sel,up]=arm[sel]*(1-low[sel]);w[sel,fore]=arm[sel]*low[sel]
 elif part=='legs':
  pelvis=np.clip((y-.80)/.18,0,1);w[:,0]=pelvis
  knee=np.clip((.62-y)/.22,0,1);knee=knee*knee*(3-2*knee)
  for side,thigh,shin in [(-1,9,10),(1,12,13)]:
   sel=x*side>=0;w[sel,thigh]=(1-pelvis[sel])*(1-knee[sel]);w[sel,shin]=(1-pelvis[sel])*knee[sel]
 idx=np.argsort(-w,axis=1)[:,:4];ww=np.take_along_axis(w,idx,axis=1);ww/=ww.sum(1,keepdims=True)
 return idx.astype('f4'),ww.astype('f4')

class Character:
 def __init__(self,name):self.name=name;self.v=[];self.f=[];self.parts=[]
 def add(self,name,fn,bounds,color,part,mat=0,step=.013,paint=None):
  p,n,f=mesh_field(fn,bounds,step*(2.9 if self.name.endswith('_far') else 1.45 if name in ['sculpted face neck ears and nose','inset eye','shaped moustache'] else 1.60));c=np.tile(color,(len(p),1)).astype('f4')
  # Subtle original material wear, not disconnected ornamental meshes.
  grain=(np.sin(p[:,0]*173+p[:,1]*257+p[:,2]*193)*np.sin(p[:,1]*83+p[:,2]*53))*.016
  c*=1+grain[:,None]
  if paint is not None:c=paint(p,c)
  ids,w=weights(p,part);data=np.column_stack((p,n,c,ids,w,np.full(len(p),mat))).astype('f4')
  offset=sum(len(a) for a in self.v);self.v.append(data);self.f.append(f+offset);self.parts.append(dict(name=name,vertices=len(p),triangles=len(f)))
 def export(self):
  v=np.concatenate(self.v);f=np.concatenate(self.f);assert np.isfinite(v).all();assert len(v)<65535,(self.name,len(v));assert np.allclose(v[:,13:17].sum(1),1)
  packed=np.zeros((len(v),28),dtype='u1')
  packed[:,:12]=v[:,:3].astype('<f4').copy().view('u1').reshape(-1,12)
  packed[:,12:15]=np.clip(np.rint(v[:,3:6]*127),-127,127).astype('i1').view('u1')
  packed[:,15:18]=np.clip(np.rint(v[:,6:9]*255),0,255).astype('u1')
  packed[:,18:22]=v[:,9:13].astype('u1')
  qw=np.rint(v[:,13:17]*255).astype('i2');qw[:,0]+=255-qw.sum(1);packed[:,22:26]=qw.astype('u1')
  packed[:,26]=v[:,17].astype('u1')
  raw=struct.pack('<II',len(v),len(f)*3)+packed.tobytes()+f.astype('<u2').tobytes()
  (OUT/'art-source'/f'{self.name}.npz').write_bytes(b'') if False else None
  np.savez_compressed(OUT/'art-source'/f'{self.name}.npz',vertices=v,faces=f)
  print(self.name,len(v),'vertices',len(f),'triangles',flush=True)
  return base64.b64encode(raw).decode(),dict(name=self.name,vertices=len(v),triangles=len(f),parts=self.parts)

def build(kind):
 cop=kind=='police';rider=kind.startswith('rider');ch=Character(kind)
 cloth=np.array([.13,.21,.32] if cop else [.62,.27,.13] if rider else [.34,.39,.23]);skin=[.78,.52,.35] if not rider else [.84,.60,.43]
 def coat(p):
  torso=union(ell(p,[0,1.23,-.015],[.35 if cop else .26 if rider else .31,.36,.225 if cop else .205 if rider else .22]),ell(p,[0,1.42,-.015],[.38 if cop else .29 if rider else .34,.17,.235]),k=.08)
  for side in [-1,1]:
   sleeve=union(cap(p,[side*.31,1.46,0],[side*.51,1.17,.015],.13 if rider else .15,.102 if rider else .122),cap(p,[side*.51,1.17,.015],[side*.625,.96,.03],.101 if rider else .12,.081 if rider else .086),k=.065)
   torso=sm(torso,sleeve,.09)
  # Sewn waist shaping, folds sculpted into the same sleeve/torso surface.
  x,y,z=p[...,0],p[...,1],p[...,2]
  wrinkles=.0014*np.sin(y*83+x*14)*np.exp(-((y-.99)/.045)**2)+.0018*np.sin(y*109-z*25)*np.exp(-((y-1.17)/.050)**2)*np.clip(abs(x)*3,0,1)
  for side in [-1,1]:
   if not rider:
    collar=np.sqrt((np.sqrt(p[...,0]**2+p[...,2]**2)-.136)**2+(p[...,1]-1.564)**2)-.031
    torso=sm(torso,collar,.018)
   if cop:torso=sm(torso,cap(p,[side*.27,1.568,-.04],[side*.385,1.535,-.03],.027,.028),.016)
  return torso+wrinkles
 def fabric(p,c):
  x,y,z=p.T
  front=z>.15
  # Recessed center seam and chest pockets painted onto connected garment surface.
  seam=(abs(x)<.011)&front;c[seam]*=.48
  pocket=(abs(abs(x)-.17)<.095)&(y>1.13)&(y<1.31)&front;c[pocket]*=.84
  stitch=pocket&((abs(y-1.29)<.007)|(abs(abs(x)-.25)<.005));c[stitch]*=1.25
  hem=(y<.97);c[hem]*=.77
  if not cop and not rider:
   patch=(x<-.13)&(x>-.28)&(y>1.01)&(y<1.11)&front;c[patch]=[.50,.39,.20]
  return c
 ch.add('continuous jacket and sleeves',coat,[[-.79,.80,-.34],[.79,1.70,.35]],cloth,'coat',4,.014,fabric)
 def legs(p):
  d=ell(p,[0,.89,0],[.295,.19,.205])
  for side in [-1,1]:d=sm(d,union(cap(p,[side*.16,.87,0],[side*.18,.52,.015],.151,.119),cap(p,[side*.18,.53,.015],[side*.18,.17,.025],.12,.087),k=.055),.07)
  return d+.003*np.sin(p[...,1]*110+p[...,2]*25)*np.exp(-((p[...,1]-.50)/.10)**2)
 def denim(p,c):
  x,y,z=p.T
  c[(abs(abs(x)-.285)<.01)]*=.75
  if not cop:
   patch=(abs(x+.18)<.072)&(abs(y-.51)<.076)&(z>.10);c[patch]=[.35,.32,.23] if not rider else [.17,.20,.24]
  return c
 ch.add('continuous trousers',legs,[[-.37,.06,-.25],[.37,1.09,.27]],[.105,.15,.22] if cop else [.19,.23,.22] if rider else [.23,.245,.19],'legs',0,.013,denim)
 for side,hand,foot in [(-1,5,11),(1,8,14)]:
  hx=side*.637
  def hands(p):
   d=union(ell(p,[hx,.88,.035],[.072,.105,.048]),cap(p,[hx,.96,.028],[hx,.86,.038],.062,.058),k=.03)
   for k in range(4):d=sm(d,cap(p,[hx+(k-1.5)*.027,.855,.05],[hx+(k-1.5)*.027,.786+(abs(k-1.5))*.009,.055],.018,.014),.012)
   return sm(d,cap(p,[hx-side*.047,.90,.045],[hx-side*.082,.847,.063],.023,.017),.022)
  ch.add('sculpted hand',hands,[[hx-.13,.75,-.04],[hx+.13,1,.12]],skin,hand,1,.007)
  def shoe(p):return union(ell(p,[side*.18,.13,.10],[.12,.095,.225]),ell(p,[side*.18,.19,.025],[.095,.16,.115]),k=.04)
  def leather(p,c):
   x,y,z=p.T;c[y<.065]=[.055,.065,.064]
   laces=(y>.193)&(z>.085)&(z<.235)&(abs(x-side*.18)<.075)&(np.mod(z,.037)<.012);c[laces]=[.60,.56,.43] if not rider else [.83,.83,.76]
   return c
  ch.add('shaped lace-up boot',shoe,[[side*.18-.16,.015,-.15],[side*.18+.16,.37,.36]],[.115,.09,.063] if not rider else [.17,.20,.23],foot,0,.009,leather)
 # Anatomical face is one fused surface, including nose, ears, chin and cheek planes.
 def face(p):
  d=union(ell(p,[0,1.955,.012],[.244 if cop else .205 if rider else .237,.278,.211]),ell(p,[0,1.795,.052],[.190 if cop else .15 if rider else .175,.141,.167]),cap(p,[0,1.62,0],[0,1.79,.01],.098,.113),k=.07)
  d=union(d,ell(p,[0,1.935,.219],[.063 if cop else .046 if rider else .067,.080 if rider else .090,.077 if cop else .065 if rider else .096]),ell(p,[0,2.002,.178],[.043,.111,.055]),ell(p,[0,1.765,.144],[.105,.065,.061]),k=.04)
  for side in [-1,1]:
   d=union(d,ell(p,[side*.145,1.898,.117],[.052 if rider else .062,.068,.058 if rider else .069]),ell(p,[side*.228,1.927,.012],[.039,.079,.05]),k=.027)
   d=np.maximum(d,-ell(p,[side*.088,1.994,.200],[.051,.032,.030]))
  return d
 def complexion(p,c):
  x,y,z=p.T
  cheek=np.exp(-((abs(x)-.145)/.06)**2-((y-1.90)/.08)**2)*np.clip(z/.2,0,1)
  c=c*(1-cheek[:,None]*.20)+np.array([.66,.30,.23])*cheek[:,None]*.20
  brow=(y>2.026)&(y<2.043)&(abs(x)>.035)&(abs(x)<.148)&(z>.16);c[brow]=[.23,.18,.125]
  mouth=(abs(x)<.072)&(abs(y-(1.819+2*x*x))<.007)&(z>.18);c[mouth]=[.47,.245,.18]
  nostril=(abs(abs(x)-.029)<.010)&(abs(y-1.903)<.008)&(z>.258);c[nostril]=[.35,.21,.15]
  return c
 ch.add('sculpted face neck ears and nose',face,[[-.30,1.51,-.24],[.30,2.26,.34]],skin,2,1,.008,complexion)
 for side in [-1,1]:
  def eye(p):return ell(p,[side*.088,1.994,.198],[.040,.025,.027])
  def iris(p,c):
   x,y,z=p.T;d=np.sqrt(((x-side*.088)*1.03)**2+((y-1.994)*1.03)**2)
   c[(d<.015)&(z>.20)]=[.21,.32,.33] if cop else [.35,.31,.15]
   c[(d<.008)&(z>.22)]=[.025,.03,.024]
   c[(abs(x-side*.088+.006)<.005)&(abs(y-2.001)<.005)&(z>.22)]=[.97,.98,.91]
   return c
  ch.add('inset eye',eye,[[side*.088-.053,1.954,.163],[side*.088+.053,2.03,.234]],[.91,.88,.78],2,2,.005,iris)
 if not cop and not rider:
  def beard(p):
   d=ell(p,[0,1.795,.108],[.194,.147,.136]);d=np.maximum(d,1.837-p[...,1]) if False else d
   # A shaped beard surface, combed rather than individual spheres.
   d=np.maximum(d,p[...,1]-1.878);d=np.maximum(d,-p[...,2]-.015);d=np.maximum(d,-ell(p,[0,1.828,.24],[.080,.032,.055]))
   d+=.0025*np.sin(p[...,0]*225+p[...,1]*27)
   return d
  def beardpaint(p,c):
   x,y,z=p.T;stripe=(np.sin(x*180+y*13)+np.sin(x*330-y*25))*.055;c*=1+stripe[:,None];return c
  ch.add('combed continuous beard',beard,[[-.21,1.62,-.04],[.21,1.90,.26]],[.37,.345,.265],2,0,.008,beardpaint)
  def beanie(p):
   d=ell(p,[0,2.115,-.025],[.249,.171,.224]);d=np.maximum(d,2.055-p[...,1]);angle=np.arctan2(p[...,0],p[...,2]+.025)
   return d+.0018*np.cos(angle*46)
  def knit(p,c):
   x,y,z=p.T;angle=np.arctan2(x,z+.025);c*=1+(.065*np.cos(angle*46)+.025*np.sin(y*260))[:,None];c[y<2.104]*=.72;return c
  ch.add('ribbed wool beanie',beanie,[[-.27,2.043,-.27],[.27,2.30,.225]],[.40,.285,.15],2,0,.007,knit)
  def bag(p):
   # Superelliptic canvas volume with rounded edges, not a box assembled from primitives.
   q=(p-np.array([0,1.21,-.277]))/np.array([.245,.295,.145]);return (np.sum(abs(q)**3.5,axis=-1)**(1/3.5)-1)*.145
  def canvas(p,c):
   x,y,z=p.T;flap=(y>1.35)&(z<-.35);c[flap]*=1.22
   straps=(abs(abs(x)-.15)<.020)&(z<-.39);c[straps]=[.24,.16,.075]
   buckle=straps&(abs(y-1.13)<.03);c[buckle]=[.70,.57,.26]
   seam=(abs(abs(x)-.207)<.005)&(z<-.35);c[seam]=[.64,.50,.28]
   return c
  ch.add('tailored canvas rucksack',bag,[[-.27,.88,-.44],[.27,1.54,-.10]],[.39,.255,.13],1,0,.011,canvas)
  for side in [-1,1]:
   ch.add('backpack shoulder strap',lambda p:cap(p,[side*.21,1.48,-.12],[side*.25,1.02,-.12],.029),[[side*.25-.08,.97,-.18],[side*.21+.08,1.54,-.06]],[.24,.16,.085],1,0,.012)
 elif cop:
  def moustache(p):
   return union(cap(p,[-.077,1.847,.205],[-.008,1.860,.231],.014,.022),cap(p,[.008,1.86,.231],[.077,1.847,.205],.022,.014),k=.012)
  ch.add('shaped moustache',moustache,[[-.10,1.82,.18],[.10,1.89,.26]],[.205,.16,.115],2,0,.005)
  def hat(p):return union(np.maximum(ell(p,[0,2.153,-.026],[.261,.096,.24]),2.11-p[...,1]),ell(p,[0,2.116,.161],[.232,.028,.191]),k=.017)
  def capcolor(p,c):
   x,y,z=p.T;c[(y<2.135)&(z>.075)]*=.52;c[(y>2.133)&(y<2.145)]=[.51,.10,.08]
   badge=(abs(x)<.028)&(y>2.154)&(y<2.191)&(z>.20);c[badge]=[.86,.69,.27];return c
  ch.add('shaped peaked cap and visor',hat,[[-.28,2.08,-.285],[.28,2.263,.365]],[.105,.17,.25],2,0,.008,capcolor)
  # Back insignia, belt and pocket stitching are surface paint, not glued beads.
  # Modify coat colors directly with a Cyrillic word on the back.
  data=ch.v[0];x,y,z=data[:,:3].T;back=(z<-.19)&(y>1.36)&(y<1.445)&(abs(x)<.215);data[back,6:9]=[.035,.065,.10]
  glyphs=['111101101101101','111101101101111','011101101101101','101101111111101','101101101111001','101101111111101','111101111011101']
  for g,glyph in enumerate(glyphs):
   for k,bit in enumerate(glyph):
    if bit=='1':
     xx=-.203+g*.059+(k%3)*.017;yy=1.431-(k//3)*.013
     pick=back&(abs(x-xx)<.009)&(abs(y-yy)<.007);data[pick,6:9]=[.78,.81,.81]
  def belt(p):
   q=(p-np.array([0,.992,0]))/np.array([.303,.032,.208]);return (np.sqrt(q[...,0]**2+q[...,2]**2)-1)*.208+np.maximum(abs(p[...,1]-.992)-.026,0)*2
  ch.add('duty belt',belt,[[-.33,.957,-.23],[.33,1.028,.235]],[.055,.064,.071],0,0,.011)
  def radio(p):q=(p-np.array([-.266,1.335,-.17]))/np.array([.055,.105,.055]);return (np.sum(abs(q)**4,axis=-1)**.25-1)*.055
  ch.add('radio',radio,[[-.33,1.22,-.24],[-.20,1.45,-.10]],[.045,.061,.069],1,0,.009)
 else:
  # Rider has a distinct sculpt: hoodie/hood, swept hair and an original cycling helmet.
  def hood(p):d=ell(p,[0,1.608,-.075],[.224,.191,.216]);return np.maximum(d,-ell(p,[0,1.67,.06],[.17,.16,.16]))
  ch.add('hood draped over shoulders',hood,[[-.24,1.39,-.31],[.24,1.82,.16]],cloth*.78,1,4,.011)
  def helmet(p):return np.maximum(ell(p,[0,2.117,-.009],[.249,.161,.236]),2.084-p[...,1])
  def vents(p,c):
   x,y,z=p.T;vent=(abs(np.mod(x+.025,.080)-.04)<.012)&(y>2.20)&(abs(z)<.12);c[vent]=[.035,.055,.066];c[(y<2.113)]=[.08,.12,.145];return c
  ch.add('vented bicycle helmet',helmet,[[-.27,2.07,-.26],[.27,2.29,.25]],[.15,.42,.46],2,0,.008,vents)
 return ch

if __name__=='__main__':
 data={};meta=[]
 for name in ['boris','police','rider','rider_far']:
  encoded,m=build(name).export();data[name]=encoded;meta.append(m)
 (OUT/'characters-data.js').write_text('/* Original sculpted meshes. Rebuild with art-source/build_characters.py. */\nwindow.PIVNOY_MODELS='+json.dumps(dict(bones=BONES.tolist(),parents=PARENTS,meshes=data),separators=(',',':'))+';\n')
 (OUT/'art-source'/'models.json').write_text(json.dumps(meta,indent=2))
