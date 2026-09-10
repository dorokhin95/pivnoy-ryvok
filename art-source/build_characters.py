"""Original sculpted, weighted character assets. Python + NumPy + SciPy; no external models.
Surface nets extract a continuous skin from sculpt fields. Garment add-ons (collars, pockets, belts, straps,
soles, hat brims, hair) are solid layers fused into their base field before meshing, so every garment is one
outer surface; each vertex then takes the colour/material of the add-on it lies on. Per-vertex ambient
occlusion is baked against the whole character's union field.

Boris is built once per outfit (skin). His head, beard and hands are shared and exported into
../characters-data.js; each outfit's garments, footwear, headwear and bag go to ../skins/boris-N.js and are
loaded on demand by the game.
Run from this folder to rebuild everything and the editable mesh metadata.
"""
import numpy as np, json, base64, struct
from pathlib import Path
from scipy.ndimage import map_coordinates
OUT=Path(__file__).resolve().parent.parent
# 15 body joints + 3 prop joints driven by characters.js for secondary motion:
# 15 bag / delivery box (child of spine), 16 beard / hood / drawstrings (child of head), 17 belt gear (child of pelvis).
BONES=np.array([[0,.94,0],[0,1.30,0],[0,1.72,0],[-.35,1.48,0],[-.51,1.17,.015],[-.63,.91,.03],[.35,1.48,0],[.51,1.17,.015],[.63,.91,.03],[-.18,.94,0],[-.18,.51,.015],[-.18,.14,.025],[.18,.94,0],[.18,.51,.015],[.18,.14,.025],[0,1.24,-.30],[0,1.78,.03],[0,.99,0]],dtype='f4')
PARENTS=[-1,0,1,1,3,4,1,6,7,0,9,10,0,12,13,1,2,0]
NB=len(BONES)
# Per-character silhouette. Applied post-sculpt in Character.export() -- after skin weights and AO are
# computed in the shared, unscaled sculpting space -- to vertices, normals and a copy of BONES,
# so the FK/skinning math (pure rotation+translation, see characters.js matrix()) stays exact.
CHAR_SCALE={'boris':(1.0,1.0,1.0),'police':(1.14,1.08,1.10),'rider':(0.88,0.95,0.90)}
def scale_for(name):return CHAR_SCALE['rider' if name=='rider_far' else name]
# Materials (fragment shader keys off these): 0 matte, 1 skin, 2 eye, 3 leather, 4 cloth (tinted by outfit),
# 5 quilted cloth (tinted), 6 knit, 7 metal, 8 hair, 9 rubber sole, 10 plastic.
MATTE,SKIN,EYE,LEATHER,CLOTH,QUILT,KNIT,METAL,HAIR,RUBBER,PLASTIC=range(11)
# Heads are sculpted at the original cartoon size and shrunk about the neck base at build time: less toy-like, still stylised.
HS=.80;HC=np.array([0,1.60,0],dtype='f4')
def headpt(p):return HC+(p-HC)/HS
def headify(fn):return lambda p:fn(headpt(p))*HS
# 3x5 bitmap glyphs for painted lettering (rows top to bottom).
FONT={'Б':'111100111101111','О':'111101101101111','М':'101111111101101','Ж':'101111010111101','С':'111100100100111','Т':'111010010010010','А':'111101111101101','В':'111101110101111','К':'101101110101101',
      'П':'111101101101101','Л':'011101101101101','И':'101101111111101','Ц':'101101101111001','Я':'111101111011101','A':'111101111101101','B':'111101110101111','I':'010010010010010','S':'111100111001111'}
def stamp(data,text,x0,y0,cell,mask,color,rowaxis=1):
 """Paint `text` onto already-meshed vertex data (columns 6:9 are colour) inside `mask`, glyph origin at (x0,y0) top-left.
 rowaxis=1 writes on a vertical face (rows down -y), rowaxis=2 on a horizontal face seen from behind (rows toward the camera, -z)."""
 x,y=data[:,0],data[:,rowaxis]
 for g,ch in enumerate(text):
  for k,bit in enumerate(FONT[ch]):
   if bit=='1':
    xx=x0+g*cell*4+(k%3)*cell;yy=y0-(k//3)*cell
    data[mask&(np.abs(x-xx)<cell*.55)&(np.abs(y-yy)<cell*.55),6:9]=color

def sm(a,b,k=.045):
 h=np.maximum(k-np.abs(a-b),0)/k
 return np.minimum(a,b)-h*h*k*.25

def ell(p,c,r):return (np.sqrt(np.sum(((p-np.array(c))/np.array(r))**2,axis=-1))-1)*min(r)
def cap(p,a,b,r1,r2=None):
 a=np.array(a);b=np.array(b);d=b-a;t=np.clip(np.sum((p-a)*d,axis=-1)/np.sum(d*d),0,1)
 r=r1 if r2 is None else r1+(r2-r1)*t
 return np.sqrt(np.sum((p-a-t[...,None]*d)**2,axis=-1))-r
def rbox(p,c,s,r=.008):
 q=np.abs(p-np.array(c))-np.array(s)
 return np.sqrt(np.sum(np.maximum(q,0)**2,axis=-1))+np.minimum(np.max(q,axis=-1),0)-r
def plane_band(p,c,n,half):
 n=np.array(n,dtype='f4');n=n/np.linalg.norm(n);return np.abs(np.sum((p-np.array(c))*n,axis=-1))-half
def yband(p,y0,y1):return np.maximum(y0-p[...,1],p[...,1]-y1)
def ring(p,c,r,thick,y0,y1):
 """Hollow vertical cylinder (collar, neckline) of mid radius r."""
 return np.maximum(np.abs(np.sqrt((p[...,0]-c[0])**2+(p[...,2]-c[2])**2)-r)-thick,yband(p,y0,y1))
def slab(base,region,h=.016):
 """A solid layer raised `h` outside the `base` field, limited to `region` (negative inside). Fused into the
 base with fuse() it adds one outer surface only -- no hidden inner sheet, no z-fighting."""
 return lambda p:np.maximum(base(p)-h,region(p))
def fuse(base,addons,k=.006):
 """Union `base` with add-on fields. Returns (fused field, decals); decals recolor the vertices each add-on owns."""
 fields=[a[0] for a in addons]
 def fused(p):
  d=base(p)
  for f in fields:d=sm(d,f(p),k)
  return d
 return fused,list(addons)
def cut(d,e):return np.maximum(d,-e)

def union(*fs,k=.045):
 a=fs[0]
 for b in fs[1:]:a=sm(a,b,k)
 return a

def mesh_field(fn,bounds,step):
 lo=np.array(bounds[0]);hi=np.array(bounds[1]);axes=[np.arange(lo[i],hi[i]+step,step) for i in range(3)]
 p=np.stack(np.meshgrid(*axes,indexing='ij'),-1).astype('f4');f=fn(p).astype('f4');shape=np.array(f.shape)-1
 corners=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])
 vals=np.stack([f[a:a+shape[0],b:b+shape[1],c:c+shape[2]] for a,b,c in corners],-1)
 active=(vals.min(-1)<0)&(vals.max(-1)>=0);cells=np.argwhere(active)
 empty=(np.zeros((0,3),'f4'),np.zeros((0,3),'f4'),np.zeros((0,3),'i4'))
 if len(cells)==0:return empty
 v=vals[active];sums=np.zeros((len(cells),3));counts=np.zeros(len(cells))
 edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.sum(abs(corners[i]-corners[j]))==1]
 for i,j in edges:
  use=(v[:,i]<0)!=(v[:,j]<0);t=np.divide(v[:,i],v[:,i]-v[:,j],out=np.zeros(len(v)),where=abs(v[:,i]-v[:,j])>1e-12)
  sums[use]+=corners[i]+t[use,None]*(corners[j]-corners[i]);counts[use]+=1
 points=(cells+sums/np.maximum(counts,1)[:,None])*step+lo
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
 faces=np.concatenate(faces) if faces else np.zeros((0,3),'i4')
 if len(faces)==0:return empty
 grad=np.gradient(f,step);coords=((points-lo)/step).T
 normals=np.stack([map_coordinates(g,coords,order=1,mode='nearest') for g in grad],-1);normals/=np.maximum(1e-9,np.linalg.norm(normals,axis=1,keepdims=True))
 cross=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]]);flip=np.sum(cross*normals[faces].mean(1),axis=1)<0;faces[flip]=faces[flip][:,[0,2,1]]
 return points.astype('f4'),normals.astype('f4'),faces.astype('i4')

def weights(p,part):
 n=len(p);w=np.zeros((n,NB));x,y=p[:,0],p[:,1];a=np.abs(x)
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
 def __init__(self,name):self.name=name;self.far=name.endswith('_far');self.v=[];self.f=[];self.fns=[];self.parts=[];self.shared=[]
 def add(self,name,fn,bounds,color,part,mat=MATTE,step=.02,paint=None,fine=False,decals=(),head=False,shared=False):
  if self.far and fine:return
  if head:
   fn0,paint0=fn,paint;fn=headify(fn0);paint=(lambda p,c:paint0(headpt(p),c)) if paint0 else None
   bounds=[list(HC+(np.array(bb,dtype='f4')-HC)*HS) for bb in bounds];decals=[(headify(d[0]),d[1],d[2]) for d in decals];step=step*HS
  st=step*(2.6 if self.far else 1.0);p,n,f=mesh_field(fn,bounds,st)
  if len(p)==0 or len(f)==0:return
  lo,hi=np.array(bounds[0],'f4'),np.array(bounds[1],'f4')
  if (p<lo+st*1.01).any() or (p>hi-st*1.01).any():print('  note:',self.name,repr(name),'surface reaches its sampling bounds',flush=True)
  c=np.tile(color,(len(p),1)).astype('f4');mats=np.full(len(p),mat,'f4')
  # Subtle material wear on every surface; sharper micro-texture is added per material in the fragment shader.
  grain=(np.sin(p[:,0]*173+p[:,1]*257+p[:,2]*193)*np.sin(p[:,1]*83+p[:,2]*53))*.016
  c*=1+grain[:,None]
  if paint is not None:c=paint(p,c)
  if decals:
   # A vertex belongs to the add-on whose own field it lies on; those get the add-on colour and material.
   vals=np.stack([d[0](p) for d in decals],1);owner=np.argmin(vals,1);hit=vals[np.arange(len(p)),owner]<.0045
   for i,(_,col,m) in enumerate(decals):
    sel=hit&(owner==i);c[sel]=np.array(col,'f4')*(1+grain[sel,None]);mats[sel]=m
  ids,w=weights(p,part);data=np.column_stack((p,n,c,ids,w,mats)).astype('f4')
  offset=sum(len(a) for a in self.v);self.v.append(data);self.f.append(f+offset);self.fns.append(fn);self.parts.append(dict(name=name,vertices=len(p),triangles=len(f)));self.shared.append(shared)
 def bake_ao(self,P,N):
  """Screen-space-free AO: probe the union of every part's field along the normal and a tilted cone."""
  def field(q):
   d=np.full(len(q),9.,dtype='f4')
   for fn in self.fns:d=np.minimum(d,fn(q).astype('f4'))
   return d
  up=np.where(np.abs(N[:,1:2])<.9,np.array([[0,1,0]],'f4'),np.array([[1,0,0]],'f4'))
  t1=np.cross(N,up);t1/=np.maximum(1e-6,np.linalg.norm(t1,axis=1,keepdims=True));t2=np.cross(N,t1)
  occ=np.zeros(len(P),'f4')
  for d,w in [(.02,.30),(.045,.25),(.08,.20),(.14,.15),(.22,.10)]:
   s=field(P+N*d);occ+=w*np.clip((d-s)/d,0,1)
  c,sn=np.cos(.85),np.sin(.85)
  for tdir in [t1,-t1,t2,-t2]:
   dirs=N*c+tdir*sn
   for d,w in [(.05,.07),(.12,.08)]:
    s=field(P+dirs*d);occ+=w*np.clip((d-s)/d,0,1)
  return np.clip(1-occ*1.1,0,1)
 def export(self,which='all',suffix=''):
  """which: 'all' every part, 'shared' only shared parts, 'skin' only outfit parts. AO is always baked against everything."""
  if not hasattr(self,'_baked'):
   v=np.concatenate(self.v);ao=self.bake_ao(v[:,:3],v[:,3:6])
   sx,sy,sz=scale_for(self.name)
   v[:,0]*=sx;v[:,1]*=sy;v[:,2]*=sz
   v[:,3]/=sx;v[:,4]/=sy;v[:,5]/=sz;v[:,3:6]/=np.maximum(1e-9,np.linalg.norm(v[:,3:6],axis=1,keepdims=True))
   self._baked=np.column_stack((v,ao)).astype('f4')
   assert np.isfinite(self._baked).all();assert np.allclose(self._baked[:,13:17].sum(1),1)
  v=self._baked
  def pack(vc,fc):
   packed=np.zeros((len(vc),22),dtype='u1')
   packed[:,0:6]=np.clip(np.rint(vc[:,:3]*8192),-32767,32767).astype('<i2').copy().view('u1').reshape(-1,6)
   packed[:,6:9]=np.clip(np.rint(vc[:,3:6]*127),-127,127).astype('i1').view('u1')
   packed[:,9:12]=np.clip(np.rint(vc[:,6:9]*255),0,255).astype('u1')
   packed[:,12:16]=vc[:,9:13].astype('u1')
   qw=np.rint(vc[:,13:17]*255).astype('i2');qw[:,0]+=255-qw.sum(1);packed[:,16:20]=qw.astype('u1')
   packed[:,20]=vc[:,17].astype('u1');packed[:,21]=np.clip(np.rint(vc[:,18]*255),0,255).astype('u1')
   return struct.pack('<II',len(vc),len(fc)*3)+packed.tobytes()+fc.astype('<u2').tobytes()
  counts=[len(a) for a in self.v];offsets=np.cumsum([0]+counts[:-1])
  pick=[i for i in range(len(counts)) if which=='all' or (which=='shared')==self.shared[i]]
  # Index buffers are uint16, so whole parts are grouped into chunks of <65535 vertices.
  chunks=[];cur=[];total=0
  for i in pick:
   if total+counts[i]>65000 and cur:chunks.append(cur);cur=[];total=0
   cur.append(i);total+=counts[i]
  if cur:chunks.append(cur)
  blobs=[]
  for group in chunks:
   vc=np.concatenate([v[offsets[i]:offsets[i]+counts[i]] for i in group]);base=0;fcs=[]
   for i in group:fcs.append(self.f[i]-offsets[i]+base);base+=counts[i]
   fc=np.concatenate(fcs);assert fc.min()>=0 and fc.max()<len(vc)<65535,(self.name,len(vc))
   blobs.append(base64.b64encode(pack(vc,fc)).decode())
  nverts=sum(counts[i] for i in pick);ntris=sum(len(self.f[i]) for i in pick)
  np.savez_compressed(OUT/'art-source'/f'{self.name}{suffix}.npz',vertices=np.concatenate([v[offsets[i]:offsets[i]+counts[i]] for i in pick]))
  print(self.name+suffix,nverts,'vertices',ntris,'triangles',len(blobs),'chunk(s)',flush=True)
  return blobs,dict(name=self.name+suffix,vertices=int(nverts),triangles=int(ntris),chunks=len(blobs),parts=[self.parts[i] for i in pick])

def build(kind,outfit=1):
 cop=kind=='police';rider=kind.startswith('rider');boris=not cop and not rider;ch=Character(kind)
 skin=[.80,.55,.40] if cop else [.86,.63,.47] if rider else [.78,.52,.36]
 gold=[.86,.70,.27];black=[.06,.06,.07];white=[.92,.92,.90]
 # Boris outfits: 0 ragged tee & shorts, 1 quilted vatnik, 2 "Bomzhstavka" courier, 3 Abibas tracksuit.
 if boris:
  cloth=np.array([[.82,.80,.74],[.34,.39,.23],[.98,.78,.10],[.09,.09,.11]][outfit]);pants=[[.36,.35,.38],[.20,.22,.28],[.11,.11,.13],[.09,.09,.11]][outfit]
  coat_mat=QUILT if outfit==1 else CLOTH
 else:
  cloth=np.array([.16,.22,.37] if cop else [.92,.38,.12]);pants=[.14,.18,.30] if cop else [.15,.16,.20];coat_mat=CLOTH
 tw=.35 if cop else .26 if rider else .31;td=.225 if cop else .205 if rider else .22
 su=(.17,.135) if cop else (.125,.098) if rider else (.15,.122)
 sl=(.133,.108) if cop else (.096,.078) if rider else (.12,.086)
 # ---------------------------------------------------------------- torso + sleeves (one garment surface)
 def torso_base(p):
  torso=union(ell(p,[0,1.23,-.015],[tw,.36,td]),ell(p,[0,1.42,-.015],[tw+.03,.17,td+.012]),k=.08)
  if boris:torso=sm(torso,ell(p,[0,1.09,.05],[tw*.86,.16,td*.92]),.07)
  if cop:torso=sm(torso,ell(p,[0,1.12,.03],[tw*.9,.17,td*.96]),.07)
  return torso
 def sleeves(p,rm=1.0,short=False):
  d=None
  for side in [-1,1]:
   if short:s=cap(p,[side*.31,1.46,0],[side*.44,1.27,.01],su[0]*rm,su[1]*1.06*rm)
   else:s=union(cap(p,[side*.31,1.46,0],[side*.51,1.17,.015],su[0]*rm,su[1]*rm),cap(p,[side*.51,1.17,.015],[side*.625,.96,.03],sl[0]*rm,sl[1]*rm),k=.06)
   d=s if d is None else np.minimum(d,s)
  return d
 def wrinkles(p):
  x,y,z=p[...,0],p[...,1],p[...,2]
  wr=.0016*np.sin(y*83+x*14)*np.exp(-((y-.99)/.05)**2)+.002*np.sin(y*109-z*25)*np.exp(-((y-1.17)/.05)**2)*np.clip(np.abs(x)*3,0,1)
  return wr+.0022*np.sin(y*140+x*20)*np.exp(-((y-1.17)/.06)**2)*np.clip((np.abs(x)-.40)*8,0,1)
 def coat0(p):return sm(torso_base(p),sleeves(p),.06)+wrinkles(p)
 hip=[.325,.205,.225] if cop else [.255,.165,.178] if rider else [.295,.19,.205]
 th=(.168,.132) if cop else (.126,.098) if rider else (.151,.119)
 shn=(.133,.096) if cop else (.098,.070) if rider else (.12,.087)
 def mk_legs(rm=1.0):
  def f(p):
   d=ell(p,[0,.89,0],[hip[0]*rm,hip[1],hip[2]*rm])
   for side in [-1,1]:d=sm(d,union(cap(p,[side*.16,.87,0],[side*.18,.52,.015],th[0]*rm,th[1]*rm),cap(p,[side*.18,.53,.015],[side*.18,.17,.025],shn[0]*rm,shn[1]*rm),k=.055),.07)
   y,z=p[...,1],p[...,2]
   return d+.003*np.sin(y*110+z*25)*np.exp(-((y-.50)/.10)**2)+.002*np.sin(y*150)*np.exp(-((y-.25)/.08)**2)
  return f
 legs0=mk_legs()
 body=lambda p:np.minimum(coat0(p),legs0(p))
 def sleeve_frame(p,side):
  """(t along the sleeve, angle around it; 0 = top) for painting stripes and bands."""
  a=np.array([side*.31,1.46,0],'f4');b=np.array([side*.625,.96,.03],'f4');d=b-a;L=float(np.linalg.norm(d));u=d/L
  rel=p-a;t=rel@u;perp=rel-np.outer(t,u);up=np.array([0,1,0],'f4');up=up-up.dot(u)*u;up/=np.linalg.norm(up);sd=np.cross(u,up)
  return t,np.arctan2(perp@sd,perp@up),L
 # ---------------------------------------------------------------- Boris outfit 0: ragged tee, shorts, mismatched boots
 if boris and outfit==0:
  def shirt(p):
   x,y,z=p[...,0],p[...,1],p[...,2]
   d=sm(torso_base(p),sleeves(p,1.0,True),.06)+wrinkles(p)
   d=np.maximum(d,(1.05+.03*np.sin(x*40)+.02*np.sin(z*55+1))-y)
   for hc,hr in [([.12,1.25,.24],.045),([-.16,1.13,.23],.04),([.05,1.36,-.23],.035),([.24,1.40,.10],.03)]:d=cut(d,ell(p,hc,[hr,hr,hr]))
   return d
  def rag(p,c):
   x,y,z=p.T;dirt=np.clip(np.sin(x*23+y*17)*np.sin(z*31-y*9),0,1);c*=1-.28*dirt[:,None];c[y<1.10]*=.82
   for hc,hr in [([.12,1.25,.24],.045),([-.16,1.13,.23],.04),([.05,1.36,-.23],.035),([.24,1.40,.10],.03)]:
    inside=np.sqrt((x-hc[0])**2+(y-hc[1])**2+(z-hc[2])**2)<hr*1.02;c[inside]=[.66,.44,.31]
   c[(np.abs(x)<.006)&(z>.12)&(y>1.28)&(y<1.50)]*=.85
   return c
  ch.add('ragged tee',shirt,[[-.62,1.0,-.30],[.62,1.66,.32]],cloth,'coat',CLOTH,.016,rag)
  ch.add('bare arms',lambda p:sleeves(p,.80),[[-.78,.82,-.16],[.78,1.62,.20]],skin,'coat',SKIN,.012)
  ch.add('belly',lambda p:ell(p,[0,1.03,.02],[tw*.92,.15,td*.86]),[[-.32,.86,-.20],[.32,1.20,.24]],skin,'coat',SKIN,.014)
  def shorts(p):
   x,y,z=p[...,0],p[...,1],p[...,2]
   return np.maximum(mk_legs(1.05)(p),(.57+.03*np.sin(x*50)+.02*np.sin(z*70))-y)
  def shortpaint(p,c):
   x,y,z=p.T;c[(np.abs(np.abs(x)-.29)<.01)]*=.8;c*=1-.18*np.clip(np.sin(x*19+z*23)*np.sin(y*31),0,1)[:,None];c[(y>1.0)]*=.75;return c
  ch.add('ragged shorts',shorts,[[-.38,.50,-.26],[.38,1.09,.28]],pants,'legs',MATTE,.017,shortpaint)
  ch.add('bare legs',mk_legs(.90),[[-.36,.06,-.24],[.36,1.0,.26]],skin,'legs',SKIN,.016,lambda p,c:(c.__setitem__((p[:,1]<.30),np.array(skin)*.82) or c))
 elif boris and outfit==1:
  addons=[]
  for side in [-1,1]:
   addons.append((slab(coat0,lambda p,s=side:ell(p,[s*.625,.96,.03],[.11,.11,.11]),.013),cloth*.84,QUILT))
   addons.append((slab(coat0,lambda p,s=side:rbox(p,[s*.165,1.245,.25],[.075,.055,.2],.01),.013),cloth*.96,QUILT))
   addons.append((slab(coat0,lambda p,s=side:rbox(p,[s*.165,1.315,.25],[.082,.016,.2],.008),.02),cloth*.84,QUILT))
  addons.append((lambda p:ring(p,[0,0,0],.153,.024,1.535,1.605),cloth*.9,QUILT))
  addons.append((slab(body,lambda p:yband(p,.975,1.025),.013),[.52,.42,.25],LEATHER))
  coat,cdec=fuse(coat0,addons)
  def fabric(p,c):
   x,y,z=p.T;front=z>.12;c[(np.abs(x)<.010)&front&(y>1.0)&(y<1.5)]*=.55;c[y<.96]*=.80;c[(x<-.36)&(np.abs(y-1.17)<.06)]*=.82;return c
  ch.add('jacket',coat,[[-.80,.80,-.34],[.80,1.70,.36]],cloth,'coat',QUILT,.016,fabric,decals=cdec)
  laddons=[(slab(legs0,lambda p,s=side:ell(p,[s*.18,.50,.14],[.085,.10,.09]),.012),[.36,.31,.22],MATTE) for side in [-1,1]]
  legs,ldec=fuse(legs0,laddons)
  ch.add('trousers',legs,[[-.38,.06,-.26],[.38,1.09,.28]],pants,'legs',MATTE,.017,lambda p,c:(c.__setitem__((np.abs(np.abs(p[:,0])-.285)<.009),np.array(pants)*.78) or c.__setitem__((p[:,1]<.30),np.array(pants)*.9) or c),decals=ldec)
 elif boris and outfit==2:
  # Courier: yellow jacket, black yoke, collar and cuffs, reflective band, chest strap for the box.
  addons=[(slab(coat0,lambda p,s=side:ell(p,[s*.625,.96,.03],[.11,.11,.11]),.013),black,CLOTH) for side in [-1,1]]
  addons.append((slab(coat0,lambda p:np.maximum(1.52-p[...,1],np.abs(p[...,0])-.48),.010),black,CLOTH))
  addons.append((lambda p:ring(p,[0,0,.01],.150,.022,1.535,1.60),black,CLOTH))
  addons.append((slab(coat0,lambda p:yband(p,1.12,1.165),.012),[.86,.88,.84],PLASTIC))
  addons.append((slab(coat0,lambda p:np.maximum(plane_band(p,[0,1.40,0],[.82,.57,0],.03),-p[...,2]-.05),.013),[.12,.12,.14],MATTE))
  coat,cdec=fuse(coat0,addons)
  def fabric(p,c):
   x,y,z=p.T;front=z>.12;c[(np.abs(x)<.010)&front&(y>1.0)&(y<1.5)]*=.45;c[y<.96]*=.85;return c
  ch.add('courier jacket',coat,[[-.80,.80,-.34],[.80,1.70,.36]],cloth,'coat',CLOTH,.016,fabric,decals=cdec)
  laddons=[(slab(legs0,lambda p,s=side:np.maximum(yband(p,.17,.24),np.abs(p[...,0]-s*.18)-.16),.011),[.24,.24,.27],MATTE) for side in [-1,1]]
  legs,ldec=fuse(legs0,laddons)
  ch.add('joggers',legs,[[-.38,.06,-.26],[.38,1.09,.28]],pants,'legs',MATTE,.017,lambda p,c:(c.__setitem__((np.abs(np.abs(p[:,0])-.285)<.011)&(p[:,1]<.86)&(p[:,1]>.25),[.98,.78,.10]) or c),decals=ldec)
 elif boris and outfit==3:
  # Abibas: black tracksuit, four white stripes down every sleeve and leg, white zip, stand collar, gold chain.
  addons=[(slab(coat0,lambda p,s=side:ell(p,[s*.625,.96,.03],[.11,.11,.11]),.011),[.14,.14,.16],CLOTH) for side in [-1,1]]
  addons.append((lambda p:ring(p,[0,0,0],.150,.022,1.535,1.61),[.14,.14,.16],CLOTH))
  coat,cdec=fuse(coat0,addons)
  def stripes(p,c):
   x,y,z=p.T
   c[(np.abs(x)<.010)&(z>.12)&(y>1.0)&(y<1.52)]=[.90,.90,.90]
   for side in [-1,1]:
    t,ang,L=sleeve_frame(p,side);on=(t>.06)&(t<L-.02)&(np.abs(x)>.34)
    for k in range(4):c[on&(np.abs(ang-(-.36+k*.24))<.055)]=[.92,.92,.92]
   return c
  ch.add('tracksuit jacket',coat,[[-.80,.80,-.34],[.80,1.70,.36]],cloth,'coat',CLOTH,.015,stripes,decals=cdec)
  data=ch.v[-1];x,y,z=data[:,:3].T;stamp(data,'ABIBAS',-.205,1.44,.017,(z<-.19)&(np.abs(x)<.22)&(y>1.34)&(y<1.46),[.94,.94,.94])
  def legstripes(p,c):
   x,y,z=p.T
   for side in [-1,1]:
    ang=np.arctan2(z-.015,(x-side*.18)*side);on=(y<.86)&(y>.20)&(x*side>0)
    for k in range(4):c[on&(np.abs(ang-(-.36+k*.24))<.065)]=[.92,.92,.92]
   return c
  ch.add('track pants',legs0,[[-.38,.06,-.26],[.38,1.09,.28]],pants,'legs',MATTE,.016,legstripes)
  ch.add('gold chain',lambda p:ring(p,[0,0,.05],.135,.011,1.505,1.53),[[-.17,1.48,-.12],[.17,1.56,.22]],gold,'coat',METAL,.006,None,True)
 else:
  addons=[(slab(coat0,lambda p,s=side:ell(p,[s*.625,.96,.03],[.11,.11,.11]),.013),cloth*.84,CLOTH) for side in [-1,1]]
  if cop:
   for side in [-1,1]:
    addons.append((slab(coat0,lambda p,s=side:rbox(p,[s*.30,1.60,-.01],[.09,.06,.05],.012),.017),[.14,.19,.33],CLOTH))      # epaulettes
    addons.append((slab(coat0,lambda p,s=side:rbox(p,[s*.16,1.29,.25],[.075,.05,.2],.01),.013),cloth*.97,CLOTH))          # chest pockets
    addons.append((slab(coat0,lambda p,s=side:rbox(p,[s*.16,1.355,.25],[.082,.016,.2],.008),.02),cloth*.86,CLOTH))        # pocket flaps
    addons.append((slab(coat0,lambda p,s=side:rbox(p,[s*.21,1.02,.25],[.08,.06,.2],.012),.013),cloth*.95,CLOTH))          # hip pockets
    addons.append((slab(coat0,lambda p,s=side:ell(p,[s*.065,1.53,.24],[.06,.045,.2]),.012),[.66,.72,.82],CLOTH))         # shirt collar
   addons.append((slab(coat0,lambda p:rbox(p,[0,1.41,.25],[.022,.09,.2],.008),.012),[.38,.10,.13],CLOTH))                  # tie
   addons.append((slab(coat0,lambda p:rbox(p,[.15,1.38,.25],[.022,.028,.2],.006),.014),gold,METAL))                      # badge
   for k in range(4):addons.append((slab(coat0,lambda p,yy=1.06+k*.095:ell(p,[0,yy,.25],[.015,.015,.2]),.016),[.80,.66,.26],METAL))
   addons.append((slab(body,lambda p:yband(p,.965,1.025),.017),black,LEATHER))                                              # duty belt
  if rider:
   addons.append((lambda p:ring(p,[0,0,.02],.145,.02,1.50,1.56),cloth*.8,CLOTH))                                          # hoodie neckline
   addons.append((slab(coat0,lambda p:rbox(p,[0,1.06,.25],[.15,.055,.2],.02),.013),cloth*.9,CLOTH))                         # kangaroo pocket
   addons.append((slab(coat0,lambda p:np.maximum(plane_band(p,[0,1.40,0],[.82,.57,0],.03),-p[...,2]-.05),.013),[.12,.12,.14],MATTE))   # bag strap
   addons.append((slab(coat0,lambda p:np.maximum(plane_band(p,[0,1.40,0],[.82,.57,0],.009),-p[...,2]-.05),.016),[.86,.90,.55],PLASTIC)) # reflective stripe
  coat,cdec=fuse(coat0,addons)
  def fabric(p,c):
   x,y,z=p.T;front=z>.12
   c[(np.abs(x)<.010)&front&(y>1.0)&(y<1.5)]*=.55;c[y<.96]*=.80
   if cop:c[(np.abs(np.abs(x)-.09)<.006)&front&(y>1.02)&(y<1.5)]*=.7
   return c
  ch.add('jacket',coat,[[-.80,.80,-.34],[.80,1.70,.36]],cloth,'coat',CLOTH,.0155 if cop else .016,fabric,decals=cdec)
  laddons=[]
  if rider:
   for side in [-1,1]:laddons.append((slab(legs0,lambda p,s=side:np.maximum(yband(p,.17,.24),np.abs(p[...,0]-s*.18)-.16),.011),[.28,.29,.33],MATTE))
  legs,ldec=fuse(legs0,laddons)
  def denim(p,c):
   x,y,z=p.T
   c[(np.abs(np.abs(x)-.285*hip[0]/.295)<.009)]*=.78
   if cop:c[(np.abs(np.abs(x)-(hip[0]-.02))<.010)&(y<.85)&(y>.2)]=[.50,.55,.66]
   if rider:c[(np.abs(np.abs(x)-.255)<.011)&(y<.84)&(y>.25)]=[.82,.82,.84]
   return c
  ch.add('trousers',legs,[[-.38,.06,-.26],[.38,1.09,.28]],pants,'legs',MATTE,.017,denim,decals=ldec)
 # ---------------------------------------------------------------- hands (shared for Boris) and footwear
 for side,hand,foot in [(-1,5,11),(1,8,14)]:
  hx=side*.637;hs=1.08 if cop else .94 if rider else 1.0
  def make_hand(hx=hx,side=side,hs=hs):
   def fn(p):
    d=union(ell(p,[hx,.88,.035],[.072*hs,.105*hs,.05*hs]),cap(p,[hx,.96,.028],[hx,.86,.038],.062*hs,.058*hs),k=.03)
    for k in range(4):d=sm(d,cap(p,[hx+(k-1.5)*.027,.855,.05],[hx+(k-1.5)*.027,.786+abs(k-1.5)*.009,.055],.018*hs,.014*hs),.012)
    return sm(d,cap(p,[hx-side*.047,.90,.045],[hx-side*.082,.847,.063],.023*hs,.017*hs),.022)
   return fn
  def glove(p,c):
   x,y,z=p.T;c[y>.935]=[.20,.20,.23];c[(y<.86)&(z>.05)]*=1.10;return c
  def knuckles(p,c):
   x,y,z=p.T;c[(np.abs(y-.86)<.012)&(z>.04)]*=.90;return c
  ch.add('hand',make_hand(),[[hx-.13,.75,-.05],[hx+.13,1.0,.13]],[.12,.12,.14] if rider else skin,hand,MATTE if rider else SKIN,.0075,glove if rider else knuckles,shared=boris)
  sneaker=rider or (boris and outfit>=2)
  m=(1.08,1.05,1.06) if cop else (.90,.80,.92) if rider else (1,.84,.96) if sneaker else (1,1,1);n_=(1.08,1.02,1.06) if cop else (.86,.78,.90) if rider else (.94,.80,.92) if sneaker else (1,1,1)
  def make_shoe(side=side,m=m,n_=n_):
   def fn(p):
    d=union(ell(p,[side*.18,.13,.10],[.12*m[0],.095*m[1],.225*m[2]]),ell(p,[side*.18,.19,.025],[.095*n_[0],.16*n_[1],.115*n_[2]]),k=.04)
    if cop:d=sm(d,cap(p,[side*.18,.19,.02],[side*.18,.31,.02],.112,.118),.03)
    if boris and outfit==0 and side==-1:d=cut(d,ell(p,[side*.18,.16,.30],[.055,.05,.06]))
    return d
   return fn
  shoe0=make_shoe();saddons=[(slab(shoe0,lambda p:p[...,1]-(.075 if sneaker or rider else .062),.014),white if (sneaker or rider) else [.10,.10,.09],RUBBER)]
  if not (rider or sneaker):saddons.append((slab(shoe0,lambda p:np.maximum(.20-p[...,2],p[...,1]-.16),.009),[.30,.20,.12] if boris else [.10,.10,.12],LEATHER))
  shoe,sdec=fuse(shoe0,saddons,k=.005)
  def leather(p,c,side=side):
   x,y,z=p.T
   if cop:c[(np.abs(y-.29)<.012)]*=.8
   laces=(y>.19)&(z>.09)&(z<.24)&(np.abs(x-side*.18)<.07)&(np.mod(z,.036)<.012)
   if boris and outfit==0:
    c[laces&(side==1)]=[.40,.36,.30];c*=1-.30*np.clip(np.sin(x*40+z*33)*np.sin(y*57),0,1)[:,None]
    hole=np.sqrt((x-side*.18)**2+(y-.16)**2+(z-.30)**2)<.065;c[hole&(side==-1)]=[.60,.42,.30]
   elif sneaker:
    c[laces]=[.95,.95,.93]
    if outfit==3:
     for k in range(4):c[(np.abs(x-side*.18)>.075)&(np.abs((z-.02)-(y-.13)*1.1-k*.045)<.011)&(y>.09)&(y<.30)]=[.08,.08,.10]
   else:c[laces]=[.62,.58,.45] if boris else [.90,.90,.88] if rider else [.30,.30,.32]
   return c
  bootcol=[.07,.07,.08] if cop else [.16,.18,.24] if rider else ([.36,.22,.12] if side==-1 else [.12,.11,.12]) if (boris and outfit==0) else (white if sneaker else [.22,.14,.08])
  ch.add('boot' if not sneaker else 'sneaker',shoe,[[side*.18-.18,.0,-.16],[side*.18+.18,.43 if cop else .37,.38]],bootcol,foot,MATTE if (rider or sneaker) else LEATHER,.011,leather,decals=sdec)
 # ---------------------------------------------------------------- head: one fused face surface, hair fused as a layer
 sk=[.244,.278,.211] if cop else [.205,.262,.205] if rider else [.237,.278,.211]
 jaw=[.19,.141,.167] if cop else [.15,.13,.16] if rider else [.175,.141,.167]
 neck=(.11,.128) if cop else (.088,.100) if rider else (.098,.113)
 nose=[.063,.09,.077] if cop else [.046,.08,.065] if rider else [.07,.092,.098]
 def skull(p):return ell(p,[0,1.955,.012],sk)
 def neckf(p):return cap(p,[0,1.53,0],[0,1.71,.01],neck[0],neck[1]*.92)
 def face0(p):
  d=union(skull(p),ell(p,[0,1.795,.052],jaw),cap(p,[0,1.66,0],[0,1.79,.01],neck[0]*1.15,neck[1]),k=.07)
  d=union(d,ell(p,[0,1.935,.219],nose),ell(p,[0,2.002,.178],[.043,.111,.055]),ell(p,[0,1.765,.144],[.105,.065,.061]),k=.04)
  d=sm(d,ell(p,[0,2.045,.185],[.15,.022,.05]),.03)
  for side in [-1,1]:
   d=union(d,ell(p,[side*.145,1.898,.117],[.07 if cop else .052 if rider else .062,.068,.076 if cop else .058 if rider else .069]),ell(p,[side*.228,1.927,.012],[.039,.079,.05]),k=.027)
   d=cut(d,ell(p,[side*.088,1.994,.200],[.052,.034,.032]))
   d=cut(d,ell(p,[side*.028,1.90,.29],[.012,.010,.02]))
  return cut(d,cap(p,[-.052,1.822,.228],[.052,1.822,.228],.011))
 hy=(2.04,2.12) if cop else (2.00,2.10)
 haircol=[.30,.30,.31] if cop else [.14,.10,.08] if rider else [.40,.30,.18]
 # Boris goes bare-headed in some outfits: a full short haircut with a receding hairline instead of a band under the hat.
 if boris:hairreg=lambda p:np.maximum(np.maximum((2.07-.7*np.maximum(0,.10-p[...,2]))-p[...,1],p[...,2]-.14),1.88-p[...,1])
 else:hairreg=lambda p:np.maximum(yband(p,*hy),p[...,2]-(.14 if rider else .10))
 face_h,fdec_h=fuse(face0,[(slab(skull,hairreg,.017),haircol,HAIR)],k=.004)
 face=lambda p:sm(headify(face_h)(p),neckf(p),.05);fdec=[(headify(d[0]),d[1],d[2]) for d in fdec_h]
 def complexion(p,c):
  x,y,z=p.T
  cheek=np.exp(-((np.abs(x)-.145)/.06)**2-((y-1.90)/.08)**2)*np.clip(z/.2,0,1)
  c=c*(1-cheek[:,None]*.22)+np.array([.70,.30,.24])*cheek[:,None]*.22
  c[(np.abs(x)<.06)&(np.abs(y-1.822)<.006)&(z>.19)]=[.40,.20,.16]
  c[(np.abs(x)<.045)&(y>1.80)&(y<1.818)&(z>.20)]=[.62,.34,.28]
  c[(np.abs(np.abs(x)-.088)<.055)&(y>2.02)&(y<2.036)&(z>.17)]*=.86
  if boris:
   for yy in [2.075,2.10]:c[(np.abs(y-yy)<.004)&(np.abs(x)<.11)&(z>.15)]*=.84
   c[(np.abs(np.abs(x)-.15)<.03)&(y>1.95)&(y<1.985)&(z>.12)]*=.90
  if cop:c[(y<1.86)&(z>.02)&(np.abs(x)<.2)&(np.sin(x*300)*np.sin(y*300)>0)]*=.93
  if rider:c[(y<1.83)&(z>.05)]*=1.03
  return c
 ch.add('face',face,[[-.26,1.48,-.22],[.26,2.14,.32]],skin,2,SKIN,.0105,lambda p,c:complexion(headpt(p),c),decals=fdec,shared=boris)
 for side in [-1,1]:
  def iris(p,c,side=side):
   x,y,z=p.T;d=np.sqrt(((x-side*.088)*1.03)**2+((y-1.994)*1.03)**2)
   c[(d<.016)&(z>.20)]=[.20,.34,.36] if cop else [.30,.42,.20] if rider else [.36,.30,.14]
   c[(d<.0085)&(z>.22)]=[.02,.02,.02]
   c[(np.abs(x-side*.088+.006)<.005)&(np.abs(y-2.001)<.005)&(z>.22)]=[.98,.98,.94]
   return c
  ch.add('eye',lambda p,s=side:ell(p,[s*.088,1.994,.198],[.040,.025,.027]),[[side*.088-.053,1.954,.163],[side*.088+.053,2.03,.234]],[.95,.94,.90],2,EYE,.0048,iris,head=True,shared=boris)
 def brows(p):
  d=np.full(p.shape[:-1],9.,'f4')
  for side in [-1,1]:
   d=np.minimum(d,ell(p,[side*.092,2.046,.214],[.052,.015 if cop else .010,.017]))
   if cop:d=sm(d,ell(p,[side*.05,2.036,.222],[.022,.013,.015]),.01)
  return d
 ch.add('eyebrows',brows,[[-.17,2.0,.17],[.17,2.09,.25]],[.16,.12,.08] if not cop else [.20,.16,.12],2,HAIR,.006,None,True,head=True,shared=boris)
 # ---------------------------------------------------------------- character-specific kit
 if boris:
  def beard(p):
   d=ell(p,[0,1.79,.11],[.20,.16,.15]);d=np.maximum(d,p[...,1]-1.88);d=np.maximum(d,-p[...,2]-.01)
   d=cut(d,ell(p,[0,1.83,.25],[.078,.028,.06]))
   d=sm(d,ell(p,[0,1.858,.238],[.10,.026,.03]),.02)
   return d+.003*np.sin(p[...,0]*225+p[...,1]*27)
  def beardpaint(p,c):
   x,y,z=p.T;stripe=(np.sin(x*180+y*13)+np.sin(x*330-y*25))*.06;c*=1+stripe[:,None];c[y>1.85]*=.9;return c
  ch.add('beard',beard,[[-.22,1.61,-.04],[.22,1.90,.28]],[.44,.40,.33],16,HAIR,.012,beardpaint,head=True,shared=True)
  def baseball_cap(name,crown_col,peak_col):
   crown0=lambda p:np.maximum(ell(p,[0,2.14,-.02],[.252,.15,.235]),2.07-p[...,1])
   peak=lambda p:np.maximum(ell(p,[0,2.085,.20],[.20,.016,.12]),.09-p[...,2])
   button=lambda p:ell(p,[0,2.29,-.02],[.02,.012,.02])
   capf,dec=fuse(crown0,[(peak,peak_col,CLOTH),(button,crown_col,CLOTH)],k=.006)
   def seams(p,c):
    x,y,z=p.T;ang=np.arctan2(x,z+.02);c[(np.abs(np.mod(ang+.4,1.05)-.5)<.03)&(y>2.09)]*=.8;return c
   ch.add(name,capf,[[-.28,2.05,-.28],[.28,2.32,.34]],crown_col,2,CLOTH,.0115,seams,decals=dec,head=True)
  if outfit==1:
   def beanie0(p):return np.maximum(ell(p,[0,2.125,-.02],[.252,.175,.228]),2.06-p[...,1])
   beanie,bdec=fuse(beanie0,[(slab(skull,lambda p:yband(p,2.03,2.105),.042),[.40,.24,.12],KNIT)],k=.008)
   ch.add('wool beanie',beanie,[[-.30,2.02,-.28],[.30,2.31,.27]],[.46,.28,.14],2,KNIT,.0125,None,decals=bdec,head=True)
   def bag0(p):
    q=(p-np.array([0,1.21,-.277]))/np.array([.245,.295,.145]);return (np.sum(np.abs(q)**3.5,axis=-1)**(1/3.5)-1)*.145
   bag,gdec=fuse(bag0,[(slab(bag0,lambda p:yband(p,1.42,1.52),.012),[.36,.24,.12],MATTE)],k=.006)
   def canvas(p,c):
    x,y,z=p.T;c[(np.abs(np.abs(x)-.207)<.006)&(z<-.35)]=[.62,.50,.28];c[(np.abs(y-1.16)<.006)&(z<-.40)]*=.8;return c
   ch.add('rucksack',bag,[[-.28,.88,-.46],[.28,1.56,-.10]],[.42,.28,.14],15,MATTE,.0145,canvas,decals=gdec)
   for side in [-1,1]:
    ch.add('flap strap',lambda p,s=side:rbox(p,[s*.13,1.30,-.435],[.018,.11,.012],.005),[[side*.13-.05,1.17,-.47],[side*.13+.05,1.43,-.38]],[.24,.16,.08],15,LEATHER,.0065,None,True)
    ch.add('buckle',lambda p,s=side:rbox(p,[s*.13,1.20,-.44],[.024,.016,.014],.004),[[side*.13-.05,1.17,-.48],[side*.13+.05,1.23,-.40]],[.72,.60,.28],15,METAL,.005,None,True)
    ch.add('shoulder strap',lambda p,s=side:cap(p,[s*.21,1.49,-.11],[s*.25,1.02,-.13],.03),[[side*.25-.08,.97,-.19],[side*.21+.08,1.55,-.05]],[.24,.16,.085],1,LEATHER,.011)
   ch.add('rolled blanket',lambda p:cap(p,[-.24,1.55,-.28],[.24,1.55,-.28],.065),[[-.32,1.47,-.36],[.32,1.63,-.20]],[.36,.42,.40],15,CLOTH,.012,lambda p,c:(c.__setitem__((np.abs(np.abs(p[:,0])-.12)<.012),[.22,.18,.12]) or c))
   ch.add('bottle',lambda p:union(cap(p,[.27,1.06,-.25],[.27,1.26,-.25],.038),cap(p,[.27,1.26,-.25],[.27,1.34,-.25],.018),k=.02),[[.20,1.0,-.32],[.34,1.37,-.18]],[.16,.46,.24],15,PLASTIC,.0065,None,True)
   ch.add('strap buckle',lambda p:rbox(p,[.0,1.0,td+.03],[.028,.02,.012],.004),[[-.05,.96,td-.02],[.05,1.04,td+.06]],[.72,.60,.28],'coat',METAL,.005,None,True)
  elif outfit==2:
   baseball_cap('courier cap',[.09,.09,.11],[.98,.78,.10])
   # Big delivery thermal box. rbox half-extents are s+r, so this one spans x +-.26, y 1.05..1.55, z -.59..-.19: the sampling
   # bounds must enclose all of it (a face on the bounds is not meshed) and the front wall sits inside the jacket, so no gap shows.
   # Black lettering goes on the back wall and on the lid, which is what the chase camera sees.
   def boxpaint(p,c):
    x,y,z=p.T;c[(np.abs(y-1.47)<.006)&(z<-.50)]*=.6;c[(np.abs(np.abs(x)-.22)<.006)&(z<-.50)]*=.7;return c
   ch.add('thermal box',lambda p:rbox(p,[0,1.30,-.39],[.23,.22,.17],.03),[[-.30,1.01,-.63],[.30,1.59,-.15]],[.98,.80,.12],15,PLASTIC,.011,boxpaint)
   data=ch.v[-1];x,y,z=data[:,:3].T;ink=[.08,.08,.10];back=(z<-.56)&(np.abs(x)<.228)
   stamp(data,'БОМЖ',-.165,1.44,.022,back&(y>1.32)&(y<1.46),ink);stamp(data,'СТАВКА',-.201,1.28,.0175,back&(y>1.18)&(y<1.30),ink)
   top=(y>1.52)&(np.abs(x)<.228)
   stamp(data,'БОМЖ',-.165,-.285,.022,top&(z<-.27)&(z>-.41),ink,rowaxis=2);stamp(data,'СТАВКА',-.201,-.415,.0175,top&(z<-.40)&(z>-.52),ink,rowaxis=2)
   for side in [-1,1]:
    ch.add('box strap',lambda p,s=side:cap(p,[s*.15,1.50,-.14],[s*.17,1.36,-.34],.02),[[side*.17-.06,1.30,-.40],[side*.15+.06,1.54,-.08]],[.12,.12,.14],1,MATTE,.0105)
   ch.add('strap clip',lambda p:rbox(p,[-.18,1.19,.20],[.03,.024,.012],.005),[[-.24,1.15,.16],[-.12,1.24,.25]],[.25,.25,.28],'coat',PLASTIC,.0055,None,True)
  elif outfit==3:
   baseball_cap('black cap',[.10,.10,.12],[.10,.10,.12])
   data=ch.v[-1];x,y,z=data[:,:3].T;front=(z>.08)&(y>2.016)&(y<2.08)&(np.abs(x)<.08);stamp(data,'A',-.02,2.072,.0128,front,[.92,.92,.92])
 elif cop:
  ch.add('moustache',lambda p:union(cap(p,[-.08,1.845,.205],[-.008,1.862,.232],.016,.024),cap(p,[.008,1.862,.232],[.08,1.845,.205],.024,.016),k=.012),[[-.11,1.81,.17],[.11,1.90,.27]],[.22,.16,.11],2,HAIR,.0052,head=True)
  def crown0(p):return np.maximum(union(ell(p,[0,2.165,-.03],[.268,.105,.248]),ell(p,[0,2.235,.02],[.20,.05,.19]),k=.03),2.10-p[...,1])
  capf,capdec=fuse(crown0,[(slab(skull,lambda p:yband(p,2.085,2.13),.041),[.55,.10,.08],CLOTH),(lambda p:np.maximum(ell(p,[0,2.11,.175],[.24,.02,.19]),.06-p[...,2]),[.05,.05,.06],LEATHER)],k=.005)
  ch.add('peaked cap',capf,[[-.30,2.06,-.30],[.30,2.31,.39]],[.12,.17,.30],2,CLOTH,.0115,None,decals=capdec,head=True)
  ch.add('cockade',lambda p:ell(p,[0,2.175,.245],[.03,.03,.012]),[[-.05,2.13,.22],[.05,2.22,.27]],gold,2,METAL,.0045,None,True,head=True)
  # Back insignia is surface paint on the tunic (part 0).
  data=ch.v[0];x,y,z=data[:,:3].T;back=(z<-.19)&(y>1.36)&(y<1.445)&(np.abs(x)<.215);data[back,6:9]=[.035,.065,.10]
  stamp(data,'ПОЛИЦИЯ',-.20,1.431,.015,back,[.80,.83,.83])
  for side in [-1,1]:
   ch.add('epaulette star',lambda p,s=side:ell(p,[s*.30,1.655,-.005],[.02,.008,.02]),[[side*.30-.05,1.62,-.04],[side*.30+.05,1.69,.03]],gold,'coat',METAL,.0045,None,True)
  ch.add('radio',lambda p:rbox(p,[-.15,1.40,.215],[.03,.055,.025],.006),[[-.20,1.33,.17],[-.10,1.47,.26]],[.05,.06,.07],1,PLASTIC,.007)
  ch.add('antenna',lambda p:cap(p,[-.15,1.455,.215],[-.15,1.56,.20],.006),[[-.18,1.44,.18],[-.12,1.58,.24]],[.05,.05,.06],1,PLASTIC,.005,None,True)
  ch.add('belt buckle',lambda p:rbox(p,[0,.995,.215],[.032,.024,.012],.004),[[-.06,.95,.17],[.06,1.04,.26]],[.80,.78,.70],'coat',METAL,.005,None,True)
  ch.add('holster',lambda p:rbox(p,[tw+.005,.90,.04],[.04,.095,.05],.012),[[tw-.06,.78,-.04],[tw+.08,1.02,.12]],black,17,LEATHER,.0095)
  ch.add('pouch',lambda p:rbox(p,[-(tw-.005),.935,.06],[.05,.05,.04],.012),[[-(tw+.08),.86,-.01],[-(tw-.08),1.01,.13]],black,17,LEATHER,.0095)
 else:
  # Delivery courier: hoodie with hood and drawstrings, kangaroo pocket, helmet, thermal box on a chest strap.
  ch.add('hood',lambda p:cut(ell(p,[0,1.62,-.07],[.228,.20,.222]),ell(p,[0,1.68,.06],[.175,.165,.165])),[[-.25,1.38,-.32],[.25,1.84,.17]],cloth*.80,16,CLOTH,.013,head=True)
  for side in [-1,1]:
   ch.add('drawstring',lambda p,s=side:union(cap(p,[s*.045,1.52,.17],[s*.06,1.30,.205],.011),ell(p,[s*.06,1.285,.206],[.014,.02,.014]),k=.01),[[side*.06-.05,1.25,.13],[side*.06+.05,1.56,.24]],[.94,.93,.88],16,MATTE,.0055,None,True)
  def helmet0(p):
   d=np.maximum(ell(p,[0,2.12,-.01],[.25,.165,.24]),2.08-p[...,1])
   for k in range(-2,3):d=cut(d,rbox(p,[k*.085,2.305,-.03],[.014,.03,.11],.006))
   return d
  helmet,hdec=fuse(helmet0,[(lambda p:np.maximum(ell(p,[0,2.105,.19],[.21,.016,.10]),.10-p[...,2]),[.09,.13,.16],PLASTIC),(slab(skull,lambda p:yband(p,2.07,2.10),.03),[.09,.13,.16],PLASTIC)],k=.005)
  def vents(p,c):
   x,y,z=p.T;c[(y>2.26)&(np.abs(np.mod(x+.0425,.085)-.0425)<.017)&(np.abs(z+.03)<.12)]*=.7;return c
  ch.add('helmet',helmet,[[-.28,2.05,-.27],[.28,2.31,.31]],[.15,.44,.48],2,PLASTIC,.0115,vents,decals=hdec,head=True)
  ch.add('chin strap',lambda p:union(cap(p,[-.25,2.07,.02],[-.115,1.735,.20],.011),cap(p,[.25,2.07,.02],[.115,1.735,.20],.011),k=.01),[[-.29,1.70,-.02],[.29,2.10,.24]],[.10,.10,.12],2,MATTE,.006,None,True,head=True)
  def boxpaint(p,c):
   x,y,z=p.T;c[(np.abs(y-1.40)<.006)&(z<-.45)]*=.6;c[(np.sqrt(x**2+(y-1.25)**2)<.07)&(z<-.5)]=[.96,.96,.94];c[(np.sqrt(x**2+(y-1.25)**2)<.045)&(z<-.5)]=[.20,.20,.22];return c
  ch.add('thermal box',lambda p:rbox(p,[0,1.29,-.37],[.20,.19,.15],.03),[[-.27,1.03,-.59],[.27,1.55,-.15]],[.96,.78,.18],15,PLASTIC,.012,boxpaint)
  for side in [-1,1]:
   ch.add('box strap',lambda p,s=side:cap(p,[s*.15,1.50,-.14],[s*.16,1.32,-.36],.02),[[side*.16-.06,1.28,-.40],[side*.15+.06,1.54,-.08]],[.12,.12,.14],1,MATTE,.0105)
  ch.add('strap clip',lambda p:rbox(p,[-.18,1.19,.20],[.03,.024,.012],.005),[[-.24,1.15,.16],[-.12,1.24,.25]],[.25,.25,.28],'coat',PLASTIC,.0055,None,True)
 return ch

if __name__=='__main__':
 import sys
 (OUT/'skins').mkdir(exist_ok=True)
 if len(sys.argv)>1:   # e.g. `python build_characters.py skin2` rebuilds one Boris outfit file only
  mp=OUT/'art-source'/'models.json';meta=json.loads(mp.read_text()) if mp.exists() else []
  cd=OUT/'characters-data.js';head,_,body=cd.read_text(encoding='utf-8').partition('window.PIVNOY_MODELS=');models=json.loads(body.strip().rstrip(';'))
  for arg in sys.argv[1:]:
   if not arg.startswith('skin'):   # police / rider / rider_far: replace that mesh inside characters-data.js
    blobs,m=build(arg).export();models['meshes'][arg]=blobs;meta=[e for e in meta if e['name']!=m['name']]+[m];continue
   i=int(arg.replace('skin',''));blobs,m=build('boris',i).export('skin',f'_skin{i}')
   meta=[e for e in meta if e['name']!=m['name']]+[m]
   (OUT/'skins'/f'boris-{i}.js').write_text(f'/* Boris outfit {i}. Rebuild with art-source/build_characters.py. */\nwindow.PIVNOY_SKINS=window.PIVNOY_SKINS||{{}};window.PIVNOY_SKINS[{i}]='+json.dumps(blobs)+';\n')
  mp.write_text(json.dumps(meta,indent=2));cd.write_text(head+'window.PIVNOY_MODELS='+json.dumps(models,separators=(',',':'))+';'+chr(10),encoding='utf-8')
  sys.exit(0)
 data={};meta=[]
 for name in ['police','rider','rider_far']:
  blobs,m=build(name).export();data[name]=blobs;meta.append(m)
 for i in range(4):
  ch=build('boris',i)
  if i==1:
   blobs,m=ch.export('shared','_shared');data['boris']=blobs;meta.append(m)
  blobs,m=ch.export('skin',f'_skin{i}');meta.append(m)
  (OUT/'skins'/f'boris-{i}.js').write_text(f'/* Boris outfit {i}. Rebuild with art-source/build_characters.py. */\nwindow.PIVNOY_SKINS=window.PIVNOY_SKINS||{{}};window.PIVNOY_SKINS[{i}]='+json.dumps(blobs)+';\n')
 bones={k:(BONES*np.array(CHAR_SCALE[k],dtype='f4')).tolist() for k in ['boris','police','rider']}
 (OUT/'characters-data.js').write_text('/* Original sculpted meshes. Rebuild with art-source/build_characters.py. Vertex layout: 22 bytes (i16 pos/8192, i8 normal, u8 rgb, u8 joints x4, u8 weights x4, u8 material, u8 ao). Boris here is the shared head/beard/hands; outfits live in skins/boris-N.js. */\nwindow.PIVNOY_MODELS='+json.dumps(dict(bones=bones,parents=PARENTS,meshes=data),separators=(',',':'))+';\n')
 (OUT/'art-source'/'models.json').write_text(json.dumps(meta,indent=2))
