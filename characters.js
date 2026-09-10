/* Original GPU-skinned characters. 18-joint rig (15 body + 3 prop joints with lagged secondary motion),
   baked ambient occlusion, per-material shading with procedural micro-texture, squash & stretch. */
(()=>{'use strict';
const STRIDE=22;
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o;}
function matrix(t,rx=0,ry=0,rz=0){const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry),cz=Math.cos(rz),sz=Math.sin(rz);return new Float32Array([cz*cy,sz*cy,-sy,0,cz*sy*sx-sz*cx,sz*sy*sx+cz*cx,cy*sx,0,cz*sy*cx+sz*sx,sz*sy*cx-cz*sx,cy*cx,0,...t,1]);}
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
// Gait personality: Semyonych runs upright with a controlled, on-duty stride; Boris is the scrappy baseline.
const GAIT={police:{amp:.82,freq:.90,arm:.60,sway:.4},boris:{amp:1,freq:1,arm:1,sway:1}};
function rig(t,slide,pose,kind){
 const data=window.PIVNOY_MODELS,rest=data.bones[kind==='rider_far'?'rider':kind]||data.bones.boris,nb=rest.length;
 const rot=rest.map(()=>[0,0,0]),off=rest.map(()=>[0,0,0]),idle=pose.idle,G=GAIT[kind]||GAIT.boris;
 const gait=idle?0:Math.sin(t*G.freq)*G.amp,lag=idle?0:Math.sin(t*G.freq-.42)*G.amp,air=pose.air||0,lean=pose.lean||0,trip=pose.trip||0,vy=pose.vy||0;
 const bob=idle?0:Math.sin(2*t*G.freq-1.1);
 rot[0]=[trip*.20,-gait*.05,-lean*.13];off[0][1]=idle?Math.sin(t)*.014:Math.abs(Math.cos(t*G.freq))*.033;
 rot[1]=[.045+air*.07,Math.sin(t)*.035*(idle?.2:1)*G.sway+gait*.06,0];rot[2]=[-.035,Math.sin(t*.32)*.055*G.sway,lean*.09];
 for(const [side,ua,fa,th,sh,ft]of [[-1,3,4,9,10,11],[1,6,7,12,13,14]]){
  const swing=gait*side,armSwing=swing*G.arm,armLag=lag*side*G.arm;
  rot[ua]=[armSwing*.54,0,-side*.07];rot[fa]=[-.46-Math.max(0,-armLag)*.45,0,0];
  rot[th]=[-swing*.64,0,side*.025];rot[sh]=[.10+Math.max(0,swing)*.92,0,0];rot[ft]=[-Math.max(0,swing)*.28,0,0];
  if(air>0){rot[th]=[-.52+(side===1?.23:0),0,0];rot[sh]=[.9,0,0];rot[ua]=[-.63,0,-side*.12];rot[fa]=[-.60,0,0];}
 }
 // Prop joints: bag/box lags the torso bounce and reacts to vertical speed; beard/hood/strings swing after the head; belt gear jiggles.
 if(nb>15){
  off[15][1]=-.020*bob+clamp(-vy*.010,-.045,.045);rot[15]=[.06*bob+clamp(vy*.02,-.14,.14),0,lean*.18];
  rot[16]=[.05*Math.sin(2*t*G.freq-1.5)+clamp(-vy*.03,-.22,.22)-trip*.15,-rot[2][1]*.6,lean*.12];
  rot[17]=[0,0,.05*Math.sin(2*t*G.freq-.9)];off[17][1]=-.006*bob;
 }
 if(kind==='rider'){
  const kick=Math.sin(t*.62),bump=Math.sin(t*1.3);
  off[0][1]=-.055+bump*.018;rot[1]=[-.13+kick*.03,0,0];rot[2]=[.13,0,0];
  for(const [side,ua,fa]of [[-1,3,4],[1,6,7]]){rot[ua]=[-1.2794,0,-side*1.20];rot[fa]=[-.0075+bump*.02,0,-side*.223];}
  // Left leg plants on the deck with a light suspension flex; right leg kicks back to push off and recovers forward.
  rot[9]=[-.15+Math.max(0,kick)*.07,0,0];rot[10]=[.20+Math.max(0,kick)*.14,0,0];
  rot[12]=[.15+kick*.42,0,0];rot[13]=[.22+Math.max(0,-kick)*.50,0,0];
  if(nb>15){off[15][1]=.012*Math.sin(t*1.3-1.0);rot[15]=[.04*Math.sin(t*1.3-1.0)-kick*.03,0,0];rot[16]=[.07*Math.sin(t*1.3-1.2),0,0];}
 }
 if(slide>0){off[0][1]-=.46*slide;rot[1][0]=-.65*slide;rot[2][0]=.43*slide;for(const [ua,fa,th,sh]of [[3,4,9,10],[6,7,12,13]]){rot[ua]=[-1.0*slide,0,0];rot[fa]=[-1.15*slide,0,0];rot[th]=[-1.30*slide,0,0];rot[sh]=[2.1*slide,0,0];}}
 if(pose.arrest){rot[1][0]=-.12;rot[3][0]=rot[6][0]=-.60;rot[4][0]=rot[7][0]=-.65;}
 const world=[],out=new Float32Array(nb*16);
 for(let i=0;i<nb;i++){const parent=data.parents[i],p=rest[i],rel=p.map((v,k)=>v-(parent<0?0:rest[parent][k])+off[i][k]);world[i]=matrix(rel,...rot[i]);if(parent>=0)world[i]=mul(world[parent],world[i]);const inverse=matrix(p.map(v=>-v));out.set(mul(world[i],inverse),i*16);}
 return out;
}
class CharacterRenderer{
 constructor(gl){this.gl=gl;this.queue=[];this.meshes={};const NB=window.PIVNOY_MODELS.parents.length;const vs=`precision highp float;
attribute vec3 position;attribute vec3 normal;attribute vec3 albedo;attribute vec4 joints;attribute vec4 weights;attribute float material;attribute float ao;
uniform mat4 bones[${NB}];uniform vec3 origin;uniform float aspect;uniform float roll;uniform float yaw;uniform vec3 tint;uniform float stretch;
varying vec3 vNormal;varying vec3 vColor;varying vec3 vWorld;varying vec3 vRest;varying float vMaterial;varying float vFog;varying float vAO;
void main(){vec3 pos=position*(1./8192.);mat4 skin=bones[int(joints.x)]*weights.x+bones[int(joints.y)]*weights.y+bones[int(joints.z)]*weights.z+bones[int(joints.w)]*weights.w;vec3 p=(skin*vec4(pos,1.)).xyz;vec3 n=mat3(skin)*normal;
p.y*=1.+stretch;p.xz*=1.-stretch*.5;
if(roll>0.){float r=clamp((roll-.12)/.76,0.,1.)*6.2831853;vec2 q=vec2(p.y-.59,p.z);float radius=length(q);float tuck=min(1.,min(roll/.10,(1.-roll)/.10));if(radius>.54)q*=mix(1.,.54/radius,tuck);p.y=.59+q.x*cos(r)-q.y*sin(r);p.z=q.x*sin(r)+q.y*cos(r);n.yz=mat2(cos(r),sin(r),-sin(r),cos(r))*n.yz;}
mat2 turn=mat2(cos(yaw),-sin(yaw),sin(yaw),cos(yaw));p.xz=turn*p.xz;n.xz=turn*n.xz;p+=origin;
vWorld=p;vRest=pos;vNormal=normalize(n);vAO=ao;vMaterial=material;
vColor=albedo*((material>3.5&&material<5.5)?tint:vec3(1.));
vec3 v=p-vec3(0.,5.2,-8.);float y=v.y*.951+v.z*.309;float z=-v.y*.309+v.z*.951;gl_Position=vec4(v.x*1.64/aspect,y*1.64,z*1.002-.2002,z);vFog=clamp((z-38.)/55.,0.,1.);}`;
 const fs=`precision mediump float;varying vec3 vNormal;varying vec3 vColor;varying vec3 vWorld;varying vec3 vRest;varying float vMaterial;varying float vFog;varying float vAO;
float hash(vec3 q){return fract(sin(dot(q,vec3(12.9898,78.233,37.719)))*43758.5453);}
float band(float lo,float hi){return step(lo,vMaterial)*(1.-step(hi,vMaterial));}
void main(){
 vec3 n=normalize(vNormal),L=normalize(vec3(-.42,.78,-.50)),eye=normalize(vec3(0.,5.2,-8.)-vWorld),r=vRest;
 float isSkin=band(.5,1.5),isEye=band(1.5,2.5),isLeather=band(2.5,3.5),isCloth=band(3.5,4.5),isQuilt=band(4.5,5.5),isKnit=band(5.5,6.5),isMetal=band(6.5,7.5),isHair=band(7.5,8.5),isRubber=band(8.5,9.5),isPlastic=step(9.5,vMaterial);
 // Procedural micro-texture in rest space: fabric grain, quilting seams, knit ribs, hair strands.
 float grain=hash(floor(r*150.))-.5;
 float tex=1.+grain*(.06*isCloth+.07*isQuilt+.04*isHair+.05*isLeather+.03*isSkin+.04*isRubber+.025*(1.-isEye-isMetal-isPlastic));
 float qa=abs(fract((r.x+r.y)*8.5)-.5),qb=abs(fract((r.x-r.y)*8.5)-.5),qd=min(qa,qb);
 float seam=1.-smoothstep(.025,.075,qd);
 tex+=isQuilt*(-.20*seam+(1.-seam)*(qd-.25)*.16);
 float ang=atan(r.x,r.z+.02);
 tex+=isKnit*(.11*cos(ang*46.)+.05*sin(r.y*300.));
 tex+=isHair*(.12*sin(r.x*230.+r.y*40.)+.06*sin(r.x*380.-r.y*30.+r.z*90.));
 vec3 base=vColor*tex;
 // Lighting: key with soft wrap, hemisphere ambient, baked AO, fill, per-material specular, skin warmth, rim, fog.
 float ndl=dot(n,L),diff=max(0.,ndl),wrapd=clamp(ndl*.55+.45,0.,1.);
 float lit=mix(diff,wrapd,.30+.40*isSkin+.20*isHair);
 vec3 amb=mix(vec3(.34,.30,.26),vec3(.62,.74,.84),n.y*.5+.5)*.88;
 float ao=mix(1.,vAO,.70-isEye*.55-isMetal*.25);
 float fill=max(0.,dot(n,normalize(vec3(.7,.3,.6))))*.26;
 vec3 color=base*(amb*ao+vec3(1.04,.98,.90)*lit*1.02*ao+fill*vec3(.62,.72,.82));
 color+=base*vec3(.24,.07,.02)*(1.-lit)*isSkin*ao;
 vec3 h=normalize(L+eye);float ndh=max(0.,dot(n,h));
 float specPow=18.+isEye*70.+isMetal*60.+isPlastic*34.+isLeather*10.+isHair*8.;
 float specAmt=.03+isSkin*.08+isEye*.45+isLeather*.24+isMetal*.75+isPlastic*.36+isHair*.10+isRubber*.04;
 color+=vec3(1.,.95,.86)*pow(ndh,specPow)*specAmt*ao;
 color+=vec3(.9,.92,.9)*pow(ndh,5.)*.05*(isCloth+isQuilt+isKnit);
 float rim=pow(1.-max(0.,dot(n,eye)),3.);
 color+=vec3(.30,.44,.50)*rim*.30*ao;
 color=color*1.08/(1.+color*.09);
 color=mix(color,vec3(.48,.67,.72),vFog);
 gl_FragColor=vec4(color,1.);
}`;
 const compile=(type,source)=>{const sh=gl.createShader(type);gl.shaderSource(sh,source);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));return sh;};
 this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,vs));gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,fs));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
 this.attrs=[['position',3,0,gl.SHORT,false],['normal',3,6,gl.BYTE,true],['albedo',3,9,gl.UNSIGNED_BYTE,true],['joints',4,12,gl.UNSIGNED_BYTE,false],['weights',4,16,gl.UNSIGNED_BYTE,true],['material',1,20,gl.UNSIGNED_BYTE,false],['ao',1,21,gl.UNSIGNED_BYTE,true]].map(([n,size,offset,type,normalized])=>({loc:gl.getAttribLocation(this.program,n),size,offset,type,normalized}));
 this.uniforms=Object.fromEntries(['bones[0]','origin','aspect','roll','yaw','tint','stretch'].map(n=>[n,gl.getUniformLocation(this.program,n)]));
 // Each character is a list of chunks (uint16 index limit); all chunks share the same bone matrices.
 for(const [kind,chunks]of Object.entries(window.PIVNOY_MODELS.meshes)){this.meshes[kind]=(Array.isArray(chunks)?chunks:[chunks]).map(encoded=>{const str=atob(encoded),raw=new Uint8Array(str.length);for(let i=0;i<str.length;i++)raw[i]=str.charCodeAt(i);const header=new DataView(raw.buffer),nv=header.getUint32(0,true),count=header.getUint32(4,true);if(raw.byteLength!==8+nv*STRIDE+count*2)throw Error('Invalid character asset');const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(raw.buffer,8,nv*STRIDE),gl.STATIC_DRAW);const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(raw.buffer.slice(8+nv*STRIDE,8+nv*STRIDE+count*2)),gl.STATIC_DRAW);return {vb,ib,count};});}
 }
 add(kind,x,y,z,t,c,slide,pose){this.queue.push({kind,x,y,z,t,c,slide,pose});}
 draw(aspect){const gl=this.gl,u=this.uniforms;gl.useProgram(this.program);gl.uniform1f(u.aspect,aspect);const base=[.34,.39,.23];
  for(const a of this.queue){const parts=this.meshes[a.kind==='rider'&&a.z>28?'rider_far':a.kind];
   gl.uniformMatrix4fv(u['bones[0]'],false,rig(a.t,a.slide,a.pose,a.kind));gl.uniform3f(u.origin,a.x,a.y,a.z);gl.uniform1f(u.roll,a.pose.roll||0);gl.uniform1f(u.yaw,a.pose.yaw||0);gl.uniform1f(u.stretch,a.pose.roll>0?0:(a.pose.stretch||0));gl.uniform3f(u.tint,...(a.kind==='boris'?a.c.map((v,i)=>v/base[i]):[1,1,1]));
   for(const m of parts){gl.bindBuffer(gl.ARRAY_BUFFER,m.vb);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.ib);for(const at of this.attrs){gl.enableVertexAttribArray(at.loc);gl.vertexAttribPointer(at.loc,at.size,at.type,at.normalized,STRIDE,at.offset);}gl.drawElements(gl.TRIANGLES,m.count,gl.UNSIGNED_SHORT,0);}}
  for(const a of this.attrs)gl.disableVertexAttribArray(a.loc);this.queue.length=0;}
}
window.CharacterRenderer=CharacterRenderer;window.PivnoyRig=rig;
})();
