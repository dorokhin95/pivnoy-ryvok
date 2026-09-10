"""Original sculpted, weighted character assets. Python + NumPy + SciPy; no external models.
Surface nets extract a continuous skin from sculpt fields. Garment add-ons (collars, pockets, belts, straps,
soles, hat brims, hair) are solid layers fused into their base field before meshing, so every garment is one
outer surface; each vertex then takes the colour/material of the add-on it lies on. Per-vertex ambient
occlusion is baked against the whole character's union field.
Run from this folder to rebuild ../characters-data.js and the editable mesh metadata.
"""
import numpy as np, json, base64, struct
from pathlib import Path
from scipy.ndimage import map_coordinates
OUT=Path(__file__).resolve().parent.parent
# 15 body joints + 3 prop joints driven by characters.js for secondary motion:
# 15 bag / delivery box (child of spine), 16 beard / hood / drawstrings (child of head), 17 belt gear (child of pelvis).
BONES=np.array([[0,.94,0],[0,1.30,0],[0,1.72,0],[-.35,1.48,0],[-.51,1.17,.015],[-.63,.91,.03],[.35,1.48,0],[.51,1.17,.015],[.63,.91,.03],[-.18,.94,0],[-.18,.51,.015],[-.18,.14,.025],[.18,.94,0],[.18,.51,.015],[.18,.14,.025],[0,1.24,-.30],[0,1.84,.04],[0,.99,0]],dtype='f4')
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
 def __init__(self,name):self.name=name;self.far=name.endswith('_far');self.v=[];self.f=[];self.fns=[];self.parts=[]
 def add(self,name,fn,bounds,color,part,mat=MATTE,step=.02,paint=None,fine=False,decals=()):
  if self.far and fine:return
  p,n,f=mesh_field(fn,bounds,step*(2.6 if self.far else 1.0))
  if len(p)==0 or len(f)==0:return
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
  offset=sum(len(a) for a in self.v);self.v.append(data);self.f.append(f+offset);self.fns.append(fn);self.parts.append(dict(name=name,vertices=len(p),triangles=len(f)))
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
 def export(self):
  v=np.concatenate(self.v);f=np.concatenate(self.f)
  ao=self.bake_ao(v[:,:3],v[:,3:6])
  sx,sy,sz=scale_for(self.name)
  v[:,0]*=sx;v[:,1]*=sy;v[:,2]*=sz
  v[:,3]/=sx;v[:,4]/=sy;v[:,5]/=sz;v[:,3:6]/=np.maximum(1e-9,np.linalg.norm(v[:,3:6],axis=1,keepdims=True))
  v=np.column_stack((v,ao)).astype('f4')
  assert np.isfinite(v).all();assert np.allclose(v[:,13:17].sum(1),1)
  def pack(vc,fc):
   packed=np.zeros((len(vc),22),dtype='u1')
   packed[:,0:6]=np.clip(np.rint(vc[:,:3]*8192),-32767,32767).astype('<i2').copy().view('u1').reshape(-1,6)
   packed[:,6:9]=np.clip(np.rint(vc[:,3:6]*127),-127,127).astype('i1').view('u1')
   packed[:,9:12]=np.clip(np.rint(vc[:,6:9]*255),0,255).astype('u1')
   packed[:,12:16]=vc[:,9:13].astype('u1')
   qw=np.rint(vc[:,13:17]*255).astype('i2');qw[:,0]+=255-qw.sum(1);packed[:,16:20]=qw.astype('u1')
   packed[:,20]=vc[:,17].astype('u1');packed[:,21]=np.clip(np.rint(vc[:,18]*255),0,255).astype('u1')
   return struct.pack('<II',len(vc),len(fc)*3)+packed.tobytes()+fc.astype('<u2').tobytes()
  # Index buffers are uint16, so whole parts are grouped into chunks of <65535 vertices.
  counts=[len(a) for a in self.v];offsets=np.cumsum([0]+counts[:-1]);chunks=[];cur=[];total=0
  for i,c in enumerate(counts):
   if total+c>65000 and cur:chunks.append(cur);cur=[];total=0
   cur.append(i);total+=c
  chunks.append(cur);blobs=[]
  for group in chunks:
   base=offsets[group[0]];n=sum(counts[i] for i in group)
   vc=v[base:base+n];fc=np.concatenate([self.f[i] for i in group])-base
   assert fc.min()>=0 and fc.max()<n<65535,(self.name,n)
   blobs.append(base64.b64encode(pack(vc,fc)).decode())
  np.savez_compressed(OUT/'art-source'/f'{self.name}.npz',vertices=v,faces=f)
  print(self.name,len(v),'vertices',len(f),'triangles',len(blobs),'chunk(s)',flush=True)
  return blobs,dict(name=self.name,vertices=len(v),triangles=len(f),chunks=len(blobs),parts=self.parts)

def build(kind):
 cop=kind=='police';rider=kind.startswith('rider');boris=not cop and not rider;ch=Character(kind)
 skin=[.80,.55,.40] if cop else [.86,.63,.47] if rider else [.78,.52,.36]
 cloth=np.array([.16,.22,.37] if cop else [.92,.38,.12] if rider else [.34,.39,.23])
 pants=[.14,.18,.30] if cop else [.15,.16,.20] if rider else [.20,.22,.28]
 coat_mat=CLOTH if not boris else QUILT;gold=[.86,.70,.27];black=[.06,.06,.07]
 tw=.35 if cop else .26 if rider else .31;td=.225 if cop else .205 if rider else .22
 su=(.17,.135) if cop else (.125,.098) if rider else (.15,.122)
 sl=(.133,.108) if cop else (.096,.078) if rider else (.12,.086)
 # ---------------------------------------------------------------- torso + sleeves (one garment surface)
 def coat0(p):
  x,y,z=p[...,0],p[...,1],p[...,2]
  torso=union(ell(p,[0,1.23,-.015],[tw,.36,td]),ell(p,[0,1.42,-.015],[tw+.03,.17,td+.012]),k=.08)
  if boris:torso=sm(torso,ell(p,[0,1.09,.05],[tw*.86,.16,td*.92]),.07)
  if cop:torso=sm(torso,ell(p,[0,1.12,.03],[tw*.9,.17,td*.96]),.07)
  for side in [-1,1]:
   sleeve=union(cap(p,[side*.31,1.46,0],[side*.51,1.17,.015],su[0],su[1]),cap(p,[side*.51,1.17,.015],[side*.625,.96,.03],sl[0],sl[1]),k=.06)
   torso=sm(torso,sleeve,.06)
  wr=.0016*np.sin(y*83+x*14)*np.exp(-((y-.99)/.05)**2)+.002*np.sin(y*109-z*25)*np.exp(-((y-1.17)/.05)**2)*np.clip(np.abs(x)*3,0,1)
  wr+=.0022*np.sin(y*140+x*20)*np.exp(-((y-1.17)/.06)**2)*np.clip((np.abs(x)-.40)*8,0,1)
  return torso+wr
 hip=[.325,.205,.225] if cop else [.255,.165,.178] if rider else [.295,.19,.205]
 th=(.168,.132) if cop else (.126,.098) if rider else (.151,.119)
 shn=(.133,.096) if cop else (.098,.070) if rider else (.12,.087)
 def legs0(p):
  d=ell(p,[0,.89,0],hip)
  for side in [-1,1]:d=sm(d,union(cap(p,[side*.16,.87,0],[side*.18,.52,.015],*th),cap(p,[side*.18,.53,.015],[side*.18,.17,.025],*shn),k=.055),.07)
  y,z=p[...,1],p[...,2]
  return d+.003*np.sin(y*110+z*25)*np.exp(-((y-.50)/.10)**2)+.002*np.sin(y*150)*np.exp(-((y-.25)/.08)**2)
 body=lambda p:np.minimum(coat0(p),legs0(p))
 addons=[]
 for side in [-1,1]:
  addons.append((slab(coat0,lambda p,s=side:ell(p,[s*.625,.96,.03],[.11,.11,.11]),.013),cloth*.84,coat_mat))            # cuffs
 if boris:
  addons.append((lambda p:ring(p,[0,0,0],.153,.024,1.535,1.605),cloth*.9,QUILT))                                          # stand collar
  for side in [-1,1]:
   addons.append((slab(coat0,lambda p,s=side:rbox(p,[s*.165,1.245,.25],[.075,.055,.2],.01),.013),cloth*.96,QUILT))      # chest pockets
   addons.append((slab(coat0,lambda p,s=side:rbox(p,[s*.165,1.315,.25],[.082,.016,.2],.008),.02),cloth*.84,QUILT))       # pocket flaps
  addons.append((slab(body,lambda p:yband(p,.975,1.025),.013),[.52,.42,.25],LEATHER))                                      # waist strap
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
  c[(np.abs(x)<.010)&front&(y>1.0)&(y<1.5)]*=.55
  c[y<.96]*=.80
  if boris:c[(x<-.36)&(np.abs(y-1.17)<.06)]*=.82
  if cop:c[(np.abs(np.abs(x)-.09)<.006)&front&(y>1.02)&(y<1.5)]*=.7
  return c
 ch.add('jacket',coat,[[-.80,.80,-.34],[.80,1.70,.36]],cloth,'coat',coat_mat,.0155 if cop else .016,fabric,decals=cdec)
 # ---------------------------------------------------------------- trousers
 laddons=[]
 if boris:
  for side in [-1,1]:laddons.append((slab(legs0,lambda p,s=side:ell(p,[s*.18,.50,.14],[.085,.10,.09]),.012),[.36,.31,.22],MATTE))   # knee patches
 if rider:
  for side in [-1,1]:laddons.append((slab(legs0,lambda p,s=side:np.maximum(yband(p,.17,.24),np.abs(p[...,0]-s*.18)-.16),.011),[.28,.29,.33],MATTE))  # jogger cuffs
 legs,ldec=fuse(legs0,laddons)
 def denim(p,c):
  x,y,z=p.T
  c[(np.abs(np.abs(x)-.285*hip[0]/.295)<.009)]*=.78
  if cop:c[(np.abs(np.abs(x)-(hip[0]-.02))<.010)&(y<.85)&(y>.2)]=[.50,.55,.66]
  if rider:c[(np.abs(np.abs(x)-.255)<.011)&(y<.84)&(y>.25)]=[.82,.82,.84]
  if boris:c[(y<.30)]*=.9
  return c
 ch.add('trousers',legs,[[-.38,.06,-.26],[.38,1.09,.28]],pants,'legs',MATTE,.017,denim,decals=ldec)
 # ---------------------------------------------------------------- hands / gloves and footwear
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
  ch.add('hand',make_hand(),[[hx-.13,.75,-.05],[hx+.13,1.0,.13]],[.12,.12,.14] if rider else skin,hand,MATTE if rider else SKIN,.0075,glove if rider else knuckles)
  m=(1.08,1.05,1.06) if cop else (.90,.80,.92) if rider else (1,1,1);n_=(1.08,1.02,1.06) if cop else (.86,.78,.90) if rider else (1,1,1)
  def make_shoe(side=side,m=m,n_=n_):
   def fn(p):
    d=union(ell(p,[side*.18,.13,.10],[.12*m[0],.095*m[1],.225*m[2]]),ell(p,[side*.18,.19,.025],[.095*n_[0],.16*n_[1],.115*n_[2]]),k=.04)
    if cop:d=sm(d,cap(p,[side*.18,.19,.02],[side*.18,.31,.02],.112,.118),.03)
    return d
   return fn
  shoe0=make_shoe();saddons=[(slab(shoe0,lambda p:p[...,1]-(.075 if rider else .062),.014),[.90,.90,.88] if rider else [.10,.10,.09],RUBBER)]
  if not rider:saddons.append((slab(shoe0,lambda p:np.maximum(.20-p[...,2],p[...,1]-.16),.009),[.30,.20,.12] if boris else [.10,.10,.12],LEATHER))
  shoe,sdec=fuse(shoe0,saddons,k=.005)
  def leather(p,c):
   x,y,z=p.T
   if cop:c[(np.abs(y-.29)<.012)]*=.8
   laces=(y>.19)&(z>.09)&(z<.24)&(np.abs(x-side*.18)<.07)&(np.mod(z,.036)<.012);c[laces]=[.62,.58,.45] if boris else [.90,.90,.88] if rider else [.30,.30,.32]
   return c
  ch.add('boot',shoe,[[side*.18-.18,.0,-.16],[side*.18+.18,.43 if cop else .37,.38]],[.07,.07,.08] if cop else [.16,.18,.24] if rider else [.22,.14,.08],foot,LEATHER if not rider else MATTE,.011,leather,decals=sdec)
 # ---------------------------------------------------------------- head: one fused face surface, hair fused as a layer
 sk=[.244,.278,.211] if cop else [.205,.262,.205] if rider else [.237,.278,.211]
 jaw=[.19,.141,.167] if cop else [.15,.13,.16] if rider else [.175,.141,.167]
 neck=(.11,.128) if cop else (.088,.100) if rider else (.098,.113)
 nose=[.063,.09,.077] if cop else [.046,.08,.065] if rider else [.07,.092,.098]
 def skull(p):return ell(p,[0,1.955,.012],sk)
 def face0(p):
  d=union(skull(p),ell(p,[0,1.795,.052],jaw),cap(p,[0,1.62,0],[0,1.79,.01],*neck),k=.07)
  d=union(d,ell(p,[0,1.935,.219],nose),ell(p,[0,2.002,.178],[.043,.111,.055]),ell(p,[0,1.765,.144],[.105,.065,.061]),k=.04)
  d=sm(d,ell(p,[0,2.045,.185],[.15,.022,.05]),.03)
  for side in [-1,1]:
   d=union(d,ell(p,[side*.145,1.898,.117],[.07 if cop else .052 if rider else .062,.068,.076 if cop else .058 if rider else .069]),ell(p,[side*.228,1.927,.012],[.039,.079,.05]),k=.027)
   d=cut(d,ell(p,[side*.088,1.994,.200],[.052,.034,.032]))
   d=cut(d,ell(p,[side*.028,1.90,.29],[.012,.010,.02]))
  return cut(d,cap(p,[-.052,1.822,.228],[.052,1.822,.228],.011))
 hy=(2.04,2.12) if cop else (2.00,2.10) if rider else (1.99,2.07)
 haircol=[.30,.30,.31] if cop else [.14,.10,.08] if rider else [.36,.26,.14]
 face,fdec=fuse(face0,[(slab(skull,lambda p:np.maximum(yband(p,*hy),p[...,2]-(.14 if rider else .10)),.017),haircol,HAIR)],k=.004)
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
 ch.add('face',face,[[-.30,1.50,-.26],[.30,2.26,.36]],skin,2,SKIN,.012,complexion,decals=fdec)
 for side in [-1,1]:
  def iris(p,c,side=side):
   x,y,z=p.T;d=np.sqrt(((x-side*.088)*1.03)**2+((y-1.994)*1.03)**2)
   c[(d<.016)&(z>.20)]=[.20,.34,.36] if cop else [.30,.42,.20] if rider else [.36,.30,.14]
   c[(d<.0085)&(z>.22)]=[.02,.02,.02]
   c[(np.abs(x-side*.088+.006)<.005)&(np.abs(y-2.001)<.005)&(z>.22)]=[.98,.98,.94]
   return c
  ch.add('eye',lambda p,s=side:ell(p,[s*.088,1.994,.198],[.040,.025,.027]),[[side*.088-.053,1.954,.163],[side*.088+.053,2.03,.234]],[.95,.94,.90],2,EYE,.0048,iris)
 def brows(p):
  d=np.full(p.shape[:-1],9.,'f4')
  for side in [-1,1]:
   d=np.minimum(d,ell(p,[side*.092,2.046,.214],[.052,.015 if cop else .010,.017]))
   if cop:d=sm(d,ell(p,[side*.05,2.036,.222],[.022,.013,.015]),.01)
  return d
 ch.add('eyebrows',brows,[[-.17,2.0,.17],[.17,2.09,.25]],[.16,.12,.08] if not cop else [.20,.16,.12],2,HAIR,.006,None,True)
 # ---------------------------------------------------------------- character-specific kit
 if boris:
  def beard(p):
   d=ell(p,[0,1.79,.11],[.20,.16,.15]);d=np.maximum(d,p[...,1]-1.88);d=np.maximum(d,-p[...,2]-.01)
   d=cut(d,ell(p,[0,1.83,.25],[.078,.028,.06]))
   d=sm(d,ell(p,[0,1.858,.238],[.10,.026,.03]),.02)
   return d+.003*np.sin(p[...,0]*225+p[...,1]*27)
  def beardpaint(p,c):
   x,y,z=p.T;stripe=(np.sin(x*180+y*13)+np.sin(x*330-y*25))*.06;c*=1+stripe[:,None];c[y>1.85]*=.9;return c
  ch.add('beard',beard,[[-.22,1.61,-.04],[.22,1.90,.28]],[.44,.40,.33],16,HAIR,.012,beardpaint)
  def beanie0(p):return np.maximum(ell(p,[0,2.125,-.02],[.252,.175,.228]),2.06-p[...,1])
  beanie,bdec=fuse(beanie0,[(slab(skull,lambda p:yband(p,2.03,2.105),.042),[.40,.24,.12],KNIT)],k=.008)
  ch.add('wool beanie',beanie,[[-.30,2.02,-.28],[.30,2.31,.27]],[.46,.28,.14],2,KNIT,.0125,None,decals=bdec)
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
 elif cop:
  ch.add('moustache',lambda p:union(cap(p,[-.08,1.845,.205],[-.008,1.862,.232],.016,.024),cap(p,[.008,1.862,.232],[.08,1.845,.205],.024,.016),k=.012),[[-.11,1.81,.17],[.11,1.90,.27]],[.22,.16,.11],2,HAIR,.0052)
  def crown0(p):return np.maximum(union(ell(p,[0,2.165,-.03],[.268,.105,.248]),ell(p,[0,2.235,.02],[.20,.05,.19]),k=.03),2.10-p[...,1])
  capf,capdec=fuse(crown0,[(slab(skull,lambda p:yband(p,2.085,2.13),.041),[.55,.10,.08],CLOTH),(lambda p:np.maximum(ell(p,[0,2.11,.175],[.24,.02,.19]),.06-p[...,2]),[.05,.05,.06],LEATHER)],k=.005)
  ch.add('peaked cap',capf,[[-.30,2.06,-.30],[.30,2.31,.39]],[.12,.17,.30],2,CLOTH,.0115,None,decals=capdec)
  ch.add('cockade',lambda p:ell(p,[0,2.175,.245],[.03,.03,.012]),[[-.05,2.13,.22],[.05,2.22,.27]],gold,2,METAL,.0045,None,True)
  # Back insignia is surface paint on the tunic (part 0).
  data=ch.v[0];x,y,z=data[:,:3].T;back=(z<-.19)&(y>1.36)&(y<1.445)&(np.abs(x)<.215);data[back,6:9]=[.035,.065,.10]
  glyphs=['111101101101101','111101101101111','011101101101101','101101111111101','101101101111001','101101111111101','111101111011101']
  for g,glyph in enumerate(glyphs):
   for k,bit in enumerate(glyph):
    if bit=='1':
     xx=-.203+g*.059+(k%3)*.017;yy=1.431-(k//3)*.013
     pick=back&(np.abs(x-xx)<.009)&(np.abs(y-yy)<.007);data[pick,6:9]=[.80,.83,.83]
  for side in [-1,1]:
   ch.add('epaulette star',lambda p,s=side:ell(p,[s*.30,1.655,-.005],[.02,.008,.02]),[[side*.30-.05,1.62,-.04],[side*.30+.05,1.69,.03]],gold,'coat',METAL,.0045,None,True)
  ch.add('radio',lambda p:rbox(p,[-.15,1.40,.215],[.03,.055,.025],.006),[[-.20,1.33,.17],[-.10,1.47,.26]],[.05,.06,.07],1,PLASTIC,.007)
  ch.add('antenna',lambda p:cap(p,[-.15,1.455,.215],[-.15,1.56,.20],.006),[[-.18,1.44,.18],[-.12,1.58,.24]],[.05,.05,.06],1,PLASTIC,.005,None,True)
  ch.add('belt buckle',lambda p:rbox(p,[0,.995,.215],[.032,.024,.012],.004),[[-.06,.95,.17],[.06,1.04,.26]],[.80,.78,.70],'coat',METAL,.005,None,True)
  ch.add('holster',lambda p:rbox(p,[tw+.005,.90,.04],[.04,.095,.05],.012),[[tw-.06,.78,-.04],[tw+.08,1.02,.12]],black,17,LEATHER,.0095)
  ch.add('pouch',lambda p:rbox(p,[-(tw-.005),.935,.06],[.05,.05,.04],.012),[[-(tw+.08),.86,-.01],[-(tw-.08),1.01,.13]],black,17,LEATHER,.0095)
 else:
  # Delivery courier: hoodie with hood and drawstrings, kangaroo pocket, helmet, thermal box on a chest strap.
  ch.add('hood',lambda p:cut(ell(p,[0,1.62,-.07],[.228,.20,.222]),ell(p,[0,1.68,.06],[.175,.165,.165])),[[-.25,1.38,-.32],[.25,1.84,.17]],cloth*.80,16,CLOTH,.013)
  for side in [-1,1]:
   ch.add('drawstring',lambda p,s=side:union(cap(p,[s*.045,1.52,.17],[s*.06,1.30,.205],.011),ell(p,[s*.06,1.285,.206],[.014,.02,.014]),k=.01),[[side*.06-.05,1.25,.13],[side*.06+.05,1.56,.24]],[.94,.93,.88],16,MATTE,.0055,None,True)
  def helmet0(p):
   d=np.maximum(ell(p,[0,2.12,-.01],[.25,.165,.24]),2.08-p[...,1])
   for k in range(-2,3):d=cut(d,rbox(p,[k*.085,2.305,-.03],[.014,.03,.11],.006))
   return d
  helmet,hdec=fuse(helmet0,[(lambda p:np.maximum(ell(p,[0,2.105,.19],[.21,.016,.10]),.10-p[...,2]),[.09,.13,.16],PLASTIC),(slab(skull,lambda p:yband(p,2.07,2.10),.03),[.09,.13,.16],PLASTIC)],k=.005)
  def vents(p,c):
   x,y,z=p.T;c[(y>2.26)&(np.abs(np.mod(x+.0425,.085)-.0425)<.017)&(np.abs(z+.03)<.12)]*=.7;return c
  ch.add('helmet',helmet,[[-.28,2.05,-.27],[.28,2.31,.31]],[.15,.44,.48],2,PLASTIC,.0115,vents,decals=hdec)
  ch.add('chin strap',lambda p:union(cap(p,[-.25,2.07,.02],[-.115,1.735,.20],.011),cap(p,[.25,2.07,.02],[.115,1.735,.20],.011),k=.01),[[-.29,1.70,-.02],[.29,2.10,.24]],[.10,.10,.12],2,MATTE,.006,None,True)
  def boxpaint(p,c):
   x,y,z=p.T;c[(np.abs(y-1.40)<.006)&(z<-.45)]*=.6;c[(np.sqrt(x**2+(y-1.25)**2)<.07)&(z<-.5)]=[.96,.96,.94];c[(np.sqrt(x**2+(y-1.25)**2)<.045)&(z<-.5)]=[.20,.20,.22];return c
  ch.add('thermal box',lambda p:rbox(p,[0,1.29,-.40],[.20,.19,.125],.03),[[-.24,1.08,-.55],[.24,1.50,-.25]],[.96,.78,.18],15,PLASTIC,.012,boxpaint)
  for side in [-1,1]:
   ch.add('box strap',lambda p,s=side:cap(p,[s*.15,1.50,-.14],[s*.16,1.32,-.36],.02),[[side*.16-.06,1.28,-.40],[side*.15+.06,1.54,-.08]],[.12,.12,.14],1,MATTE,.0105)
  ch.add('strap clip',lambda p:rbox(p,[-.18,1.19,.20],[.03,.024,.012],.005),[[-.24,1.15,.16],[-.12,1.24,.25]],[.25,.25,.28],'coat',PLASTIC,.0055,None,True)
 return ch

if __name__=='__main__':
 data={};meta=[]
 for name in ['boris','police','rider','rider_far']:
  blobs,m=build(name).export();data[name]=blobs;meta.append(m)
 bones={k:(BONES*np.array(CHAR_SCALE[k],dtype='f4')).tolist() for k in ['boris','police','rider']}
 (OUT/'characters-data.js').write_text('/* Original sculpted meshes. Rebuild with art-source/build_characters.py. Vertex layout: 22 bytes (i16 pos/8192, i8 normal, u8 rgb, u8 joints x4, u8 weights x4, u8 material, u8 ao). */\nwindow.PIVNOY_MODELS='+json.dumps(dict(bones=bones,parents=PARENTS,meshes=data),separators=(',',':'))+';\n')
 (OUT/'art-source'/'models.json').write_text(json.dumps(meta,indent=2))
