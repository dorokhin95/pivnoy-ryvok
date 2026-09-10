/* Original GPU-skinned characters. 15-joint rig, smooth normals, material lighting. */
(()=>{'use strict';
const identity=()=>new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o;}
function matrix(t,rx=0,ry=0,rz=0){const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry),cz=Math.cos(rz),sz=Math.sin(rz);return new Float32Array([cz*cy,sz*cy,-sy,0,cz*sy*sx-sz*cx,sz*sy*sx+cz*cx,cy*sx,0,cz*sy*cx+sz*sx,sz*sy*cx-cz*sx,cy*cx,0,...t,1]);}
function rig(t,slide,pose,kind){const data=window.PIVNOY_MODELS,rest=data.bones,rot=rest.map(()=>[0,0,0]),off=rest.map(()=>[0,0,0]),idle=pose.idle,gait=idle?0:Math.sin(t),air=pose.air||0,roll=pose.roll||0;
 rot[0]=[(pose.trip||0)*.20,0,-(pose.lean||0)*.13];off[0][1]=idle?Math.sin(t)*.014:Math.abs(Math.cos(t))*.033;
 rot[1]=[.045+air*.07,Math.sin(t)*.035*(idle?.2:1),0];rot[2]=[-.035,Math.sin(t*.32)*.055,0];
 for(const [side,ua,fa,th,sh,ft]of [[-1,3,4,9,10,11],[1,6,7,12,13,14]]){
  const swing=gait*side;rot[ua]=[swing*.54,0,-side*.07];rot[fa]=[-.46-Math.max(0,-swing)*.45,0,0];rot[th]=[-swing*.64,0,side*.025];rot[sh]=[.10+Math.max(0,swing)*.92,0,0];rot[ft]=[-Math.max(0,swing)*.28,0,0];
  if(air>0){rot[th]=[-.52+(side===1?.23:0),0,0];rot[sh]=[.9,0,0];rot[ua]=[-.63,0,-side*.12];rot[fa]=[-.60,0,0];}
 }
 if(kind==='rider'){
  off[0][1]=-.055+Math.sin(t*1.3)*.018;rot[1]=[-.13,0,0];rot[2]=[.13,0,0];
  for(const [side,ua,fa,th,sh]of [[-1,3,4,9,10],[1,6,7,12,13]]){rot[ua]=[-1.2794,0,-side*1.20];rot[fa]=[-.0075,0,-side*.223];rot[th]=[side*.13,0,0];rot[sh]=[.18,0,0];}
 }
 if(slide>0){off[0][1]-=.46*slide;rot[1][0]=-.65*slide;rot[2][0]=.43*slide;for(const [ua,fa,th,sh]of [[3,4,9,10],[6,7,12,13]]){rot[ua]=[-1.0*slide,0,0];rot[fa]=[-1.15*slide,0,0];rot[th]=[-1.30*slide,0,0];rot[sh]=[2.1*slide,0,0];}}
 if(pose.arrest){rot[1][0]=-.12;rot[3][0]=rot[6][0]=-.60;rot[4][0]=rot[7][0]=-.65;}
 const world=[],out=new Float32Array(15*16);
 for(let i=0;i<15;i++){const parent=data.parents[i],p=rest[i],rel=p.map((v,k)=>v-(parent<0?0:rest[parent][k])+off[i][k]);world[i]=matrix(rel,...rot[i]);if(parent>=0)world[i]=mul(world[parent],world[i]);const inverse=matrix(p.map(v=>-v));out.set(mul(world[i],inverse),i*16);}
 return out;
}
class CharacterRenderer{
 constructor(gl){this.gl=gl;this.queue=[];this.meshes={};const vs=`precision highp float;
attribute vec3 position;attribute vec3 normal;attribute vec3 albedo;attribute vec4 joints;attribute vec4 weights;attribute float material;
uniform mat4 bones[15];uniform vec3 origin;uniform float aspect;uniform float roll;uniform float yaw;uniform vec3 tint;
varying vec3 vNormal;varying vec3 vColor;varying vec3 vWorld;varying float vMaterial;varying float vFog;
void main(){mat4 skin=bones[int(joints.x)]*weights.x+bones[int(joints.y)]*weights.y+bones[int(joints.z)]*weights.z+bones[int(joints.w)]*weights.w;vec3 p=(skin*vec4(position,1.)).xyz;vec3 n=mat3(skin)*normal;
if(roll>0.){float r=clamp((roll-.12)/.76,0.,1.)*6.2831853;vec2 q=vec2(p.y-.59,p.z);float radius=length(q);float tuck=min(1.,min(roll/.10,(1.-roll)/.10));if(radius>.54)q*=mix(1.,.54/radius,tuck);p.y=.59+q.x*cos(r)-q.y*sin(r);p.z=q.x*sin(r)+q.y*cos(r);n.yz=mat2(cos(r),sin(r),-sin(r),cos(r))*n.yz;}
mat2 turn=mat2(cos(yaw),-sin(yaw),sin(yaw),cos(yaw));p.xz=turn*p.xz;n.xz=turn*n.xz;p+=origin;
vWorld=p;vNormal=normalize(n);vColor=albedo*(material>3.5?tint:vec3(1.));vMaterial=material;
vec3 v=p-vec3(0.,5.2,-8.);float y=v.y*.951+v.z*.309;float z=-v.y*.309+v.z*.951;gl_Position=vec4(v.x*1.64/aspect,y*1.64,z*1.002-.2002,z);vFog=clamp((z-38.)/55.,0.,1.);}`;
 const fs=`precision mediump float;varying vec3 vNormal;varying vec3 vColor;varying vec3 vWorld;varying float vMaterial;varying float vFog;
void main(){vec3 n=normalize(vNormal),light=normalize(vec3(-.42,.78,-.50)),eye=normalize(vec3(0.,5.2,-8.)-vWorld);float diffuse=max(0.,dot(n,light)),fill=max(0.,dot(n,normalize(vec3(.7,.3,.6))));vec3 color=vColor*(.40+.66*diffuse+.19*fill);float skin=step(.5,vMaterial)*(1.-step(1.5,vMaterial));float shiny=step(1.5,vMaterial)*(1.-step(2.5,vMaterial));float spec=pow(max(0.,dot(n,normalize(light+eye))),mix(16.,65.,shiny));color+=vec3(.98,.85,.65)*spec*(.035+skin*.09+shiny*.30);color+=vec3(.25,.37,.40)*pow(1.-max(0.,dot(n,eye)),3.)*.20;color=mix(color,vec3(.48,.67,.72),vFog);gl_FragColor=vec4(color,1.);}`;
 const compile=(type,source)=>{const sh=gl.createShader(type);gl.shaderSource(sh,source);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));return sh;};
 this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,vs));gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,fs));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
 this.attrs=[['position',3,0,gl.FLOAT,false],['normal',3,12,gl.BYTE,true],['albedo',3,15,gl.UNSIGNED_BYTE,true],['joints',4,18,gl.UNSIGNED_BYTE,false],['weights',4,22,gl.UNSIGNED_BYTE,true],['material',1,26,gl.UNSIGNED_BYTE,false]].map(([n,size,offset,type,normalized])=>({loc:gl.getAttribLocation(this.program,n),size,offset,type,normalized}));
 this.uniforms=Object.fromEntries(['bones[0]','origin','aspect','roll','yaw','tint'].map(n=>[n,gl.getUniformLocation(this.program,n)]));
 for(const [kind,encoded]of Object.entries(window.PIVNOY_MODELS.meshes)){const str=atob(encoded),raw=new Uint8Array(str.length);for(let i=0;i<str.length;i++)raw[i]=str.charCodeAt(i);const header=new DataView(raw.buffer),nv=header.getUint32(0,true),count=header.getUint32(4,true);if(raw.byteLength!==8+nv*28+count*2)throw Error('Invalid character asset');const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(raw.buffer,8,nv*28),gl.STATIC_DRAW);const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(raw.buffer,8+nv*28,count),gl.STATIC_DRAW);this.meshes[kind]={vb,ib,count};}
 }
 add(kind,x,y,z,t,c,slide,pose){this.queue.push({kind,x,y,z,t,c,slide,pose});}
 draw(aspect){const gl=this.gl,u=this.uniforms;gl.useProgram(this.program);gl.uniform1f(u.aspect,aspect);for(const a of this.queue){const m=this.meshes[a.kind==='rider'&&a.z>28?'rider_far':a.kind];gl.bindBuffer(gl.ARRAY_BUFFER,m.vb);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.ib);for(const at of this.attrs){gl.enableVertexAttribArray(at.loc);gl.vertexAttribPointer(at.loc,at.size,at.type,at.normalized,28,at.offset);}gl.uniformMatrix4fv(u['bones[0]'],false,rig(a.t,a.slide,a.pose,a.kind));gl.uniform3f(u.origin,a.x,a.y,a.z);gl.uniform1f(u.roll,a.pose.roll||0);gl.uniform1f(u.yaw,a.pose.yaw||0);const base=[.34,.39,.23];gl.uniform3f(u.tint,...(a.kind==='boris'?a.c.map((v,i)=>v/base[i]):[1,1,1]));gl.drawElements(gl.TRIANGLES,m.count,gl.UNSIGNED_SHORT,0);}for(const a of this.attrs)gl.disableVertexAttribArray(a.loc);this.queue.length=0;}
}
window.CharacterRenderer=CharacterRenderer;window.PivnoyRig=rig;
})();
