/* Original GPU-skinned characters. 18-joint rig (15 body + 3 prop joints with lagged secondary motion),
   baked ambient occlusion, per-material shading with procedural micro-texture, squash & stretch. */
(()=>{'use strict';
const STRIDE=22;
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o;}
function matrix(t,rx=0,ry=0,rz=0){const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry),cz=Math.cos(rz),sz=Math.sin(rz);return new Float32Array([cz*cy,sz*cy,-sy,0,cz*sy*sx-sz*cx,sz*sy*sx+cz*cx,cy*sx,0,cz*sy*cx+sz*sx,sz*sy*cx-cz*sx,cy*cx,0,...t,1]);}
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
// Gait personality: Semyonych runs upright with a controlled, on-duty stride; Boris is the scrappy baseline.
const GAIT={police:{amp:.82,freq:.90,arm:.60,sway:.4},boris:{amp:1,freq:1,arm:1,sway:1}};
const lerp=(a,b,k)=>a+(b-a)*k,mix3=(a,b,k)=>[lerp(a[0],b[0],k),lerp(a[1],b[1],k),lerp(a[2],b[2],k)];
const SIDES=[[-1,3,4,9,10,11],[1,6,7,12,13,14]];
function rig(t,slide,pose,kind){
 const data=window.PIVNOY_MODELS,rest=data.bones[kind==='rider_far'?'rider':kind]||data.bones.boris,nb=rest.length;
 const rot=rest.map(()=>[0,0,0]),off=rest.map(()=>[0,0,0]),idle=pose.idle,G=GAIT[kind]||GAIT.boris;
 const speed=clamp(pose.speed||0,0,1),stride=G.amp*(1+.18*speed),w=t*G.freq;
 const gait=idle?0:Math.sin(w)*stride,lag=idle?0:Math.sin(w-.42)*stride,air=pose.air||0,lean=pose.lean||0,trip=pose.trip||0,vy=pose.vy||0,up=clamp(vy/9.8,-1,1),pr=pose.roll||0;
 // Two bounces per stride: lowest at mid-stance (legs together), highest in flight (legs apart).
 const ph=idle?0:Math.sin(w),bounce=ph*ph,bob=idle?0:Math.sin(2*w-1.1);
 // Pelvis: lean grows with speed, hips yaw and roll with the stride, bank into a lane change.
 rot[0]=[trip*.20+.04*speed,-gait*.06+lean*.10,-lean*.13+gait*.05];
 off[0][1]=idle?Math.sin(t)*.014:bounce*.036;off[0][2]=idle?Math.sin(t*.7)*.006:0;
 // Chest counters the hips and turns into the lane change; the head stays level, nods with the bounce and leads the turn.
 rot[1]=[.045+.10*speed+air*.07,Math.sin(t)*.035*(idle?.2:1)*G.sway+gait*.08+lean*.22,-gait*.03];
 rot[2]=[-.035-.05*speed-bounce*.03,Math.sin(t*.32)*.055*G.sway+lean*.15,lean*.09-gait*.02];
 // Glance over a shoulder (pose.look, signed, + = right): the chest twists a little, the head a lot and tilts into the turn.
 const look=pose.look||0;if(look){const a=Math.abs(look),s=Math.sign(look);rot[1][1]+=s*a*.60;rot[2][1]+=s*a*1.20;rot[2][0]-=a*.08;rot[2][2]-=s*a*.09;}
 for(const [side,ua,fa,th,sh,ft]of SIDES){
  const swing=gait*side,armSwing=swing*G.arm,armLag=lag*side*G.arm,fwd=Math.max(0,swing),back=Math.max(0,-swing);
  // Elbows flex as the arm comes forward; a stumble throws the arms up and out.
  rot[ua]=[armSwing*.58-trip*.9,0,-side*.07+side*trip*.7];rot[fa]=[-.50-Math.max(0,-armLag)*.55-trip*.4,0,0];
  // Knee lifts on the forward swing, toe points on the push-off behind.
  rot[th]=[-swing*.66,0,side*.025];rot[sh]=[.10+fwd*.95,0,0];rot[ft]=[-fwd*.28+back*.35,0,0];
  if(air>0){
   // Rising: arms swing up, knees tuck. Falling: legs reach for the ground, arms spread for balance.
   const rise=Math.max(0,up),fall=Math.max(0,-up);
   rot[th]=[-.52+(side===1?.23:0)+fall*.32-rise*.10,0,0];rot[sh]=[.9-fall*.5,0,0];rot[ft]=[-.2+fall*.35,0,0];
   rot[ua]=[-.63-rise*.9+fall*.15,0,-side*.12+side*fall*.6];rot[fa]=[-.60-rise*.3,0,0];
  }
 }
 // Prop joints: bag/box lags the torso bounce and reacts to vertical speed; beard/hood/strings swing after the head; belt gear jiggles.
 if(nb>15){
  off[15][1]=-.020*bob+clamp(-vy*.010,-.045,.045);rot[15]=[.06*bob+clamp(vy*.02,-.14,.14),0,lean*.18];
  rot[16]=[.05*Math.sin(2*w-1.5)+clamp(-vy*.03,-.22,.22)-trip*.15,-rot[2][1]*.6,lean*.12];
  rot[17]=[0,0,.05*Math.sin(2*w-.9)];off[17][1]=-.006*bob;
 }
 if(kind==='rider'||pose.ride){
  const kick=Math.sin(t*.62),bump=Math.sin(t*1.3);
  off[0][1]=-.055+bump*.018;rot[1]=[-.13+kick*.03,kick*.04,0];rot[2]=[.13,-kick*.03,0];
  for(const [side,ua,fa]of [[-1,3,4],[1,6,7]]){rot[ua]=[-1.2794,0,-side*1.20];rot[fa]=[-.0075+bump*.02,0,-side*.223];}
  // Left leg plants on the deck with a light suspension flex; right leg kicks back to push off and recovers forward.
  rot[9]=[-.15+Math.max(0,kick)*.07,0,0];rot[10]=[.20+Math.max(0,kick)*.14,0,0];
  rot[12]=[.15+kick*.42,0,0];rot[13]=[.22+Math.max(0,-kick)*.50,0,0];
  if(nb>15){off[15][1]=.012*Math.sin(t*1.3-1.0);rot[15]=[.04*Math.sin(t*1.3-1.0)-kick*.03,0,0];rot[16]=[.07*Math.sin(t*1.3-1.2),0,0];}
 }
 if(slide>0&&pr>0&&pose.ride){
  // Ducking on the scooter instead of rolling: knees deep, chest down, hands stay on the bar.
  const k=slide;off[0][1]-=.34*k;rot[1][0]+=.42*k;rot[2][0]-=.25*k;for(const [side,ua,fa,th,sh,ft]of SIDES){rot[th]=mix3(rot[th],[-1.05,0,side*.08],k);rot[sh]=mix3(rot[sh],[1.7,0,0],k);rot[ft]=mix3(rot[ft],[-.4,0,0],k);}
 }else if(slide>0&&pr>0){
  // Forward roll done by the skeleton: a real tuck (chin to chest, knees to chest, arms around the knees) and the whole
  // body turning about the centre of the curled-up ball, which sits 0.40 above the ground. No geometry is squashed.
  const k=slide,ang=clamp((pr-.12)/.76,0,1)*6.2831853,cx=Math.cos(ang),sx=Math.sin(ang);
  rot[0]=[rot[0][0]*(1-k)+ang,rot[0][1]*(1-k),rot[0][2]*(1-k)];
  off[0]=[0,lerp(off[0][1],-.54-.02*cx+.30*sx,k),lerp(off[0][2],-.02*sx-.30*cx,k)];
  rot[1][0]=lerp(rot[1][0],.95,k);rot[2][0]=lerp(rot[2][0],.72,k);
  for(const [side,ua,fa,th,sh,ft]of SIDES){rot[ua]=mix3(rot[ua],[-1.15,0,side*.15],k);rot[fa]=mix3(rot[fa],[-1.35,0,0],k);rot[th]=mix3(rot[th],[-1.45,0,side*.05],k);rot[sh]=mix3(rot[sh],[2.25,0,0],k);rot[ft]=mix3(rot[ft],[.30,0,0],k);}
 }else if(slide>0){
  // Landing: absorb the impact with a short crouch, arms out.
  const k=slide;off[0][1]-=.14*k;rot[1][0]+=.28*k;rot[2][0]-=.15*k;
  for(const [side,ua,fa,th,sh]of SIDES){rot[th][0]-=.45*k;rot[sh][0]+=.75*k;rot[ua][0]-=.2*k;rot[ua][2]+=side*.35*k;}
 }
 // Knocked down: pitch forward onto the ground, arms out to break the fall, legs trailing, gear flopping.
 const fall=pose.fall||0,rise=(pose.arrest&&kind!=='police')?1:(pose.rise||0);
 if(fall>0){
  rot[0]=mix3(rot[0],[1.52,0,Math.sin(t*9)*.03*(1-fall)],fall);off[0]=[0,lerp(off[0][1],-.72,fall),lerp(off[0][2],.30,fall)];
  rot[1]=mix3(rot[1],[-.25,0,0],fall);rot[2]=mix3(rot[2],[-.62,.15,0],fall);
  for(const [side,ua,fa,th,sh,ft]of SIDES){rot[ua]=mix3(rot[ua],[-2.3,0,side*.55],fall);rot[fa]=mix3(rot[fa],[-.35,0,0],fall);rot[th]=mix3(rot[th],[.25,0,side*.18],fall);rot[sh]=mix3(rot[sh],[.45,0,0],fall);rot[ft]=mix3(rot[ft],[.30,0,0],fall);}
  if(nb>15){rot[15]=mix3(rot[15],[-.45,0,0],fall);rot[16]=mix3(rot[16],[-.30,0,0],fall);rot[17]=mix3(rot[17],[0,0,.1],fall);}
 }
 // Hauled up onto his knees: hands up, head hung, shoulders heaving.
 if(rise>0){
  const sway=Math.sin(t*1.6)*.02;
  rot[0]=mix3(rot[0],[.08,0,0],rise);off[0]=[0,lerp(off[0][1],-.41,rise),lerp(off[0][2],0,rise)];
  rot[1]=mix3(rot[1],[.14+sway,0,0],rise);rot[2]=mix3(rot[2],[.40,Math.sin(t*.7)*.08,0],rise);
  for(const [side,ua,fa,th,sh,ft]of SIDES){rot[ua]=mix3(rot[ua],[-2.35,0,side*.5],rise);rot[fa]=mix3(rot[fa],[-.55,0,0],rise);rot[th]=mix3(rot[th],[0,0,side*.08],rise);rot[sh]=mix3(rot[sh],[1.57,0,0],rise);rot[ft]=mix3(rot[ft],[.9,0,0],rise);}
  if(nb>15){rot[15]=mix3(rot[15],[0,0,0],rise);rot[16]=mix3(rot[16],[.05,0,0],rise);rot[17]=mix3(rot[17],[0,0,0],rise);}
 }
 // Jetpack flight: hanging from the straps, knees up a little, feet dangling.
 const fly=pose.fly||0;
 if(fly>0){rot[0]=mix3(rot[0],[.10,0,0],fly);rot[1][0]=lerp(rot[1][0],-.05,fly);rot[2][0]=lerp(rot[2][0],-.10,fly);for(const [side,ua,fa,th,sh,ft]of SIDES){rot[ua]=mix3(rot[ua],[.20,0,-side*.12],fly);rot[fa]=mix3(rot[fa],[-2.05,0,-side*.25],fly);rot[th]=mix3(rot[th],[-.40+(side===1?.15:0),0,side*.07],fly);rot[sh]=mix3(rot[sh],[.85,0,0],fly);rot[ft]=mix3(rot[ft],[.40,0,0],fly);}}
 // Semyonych's street repertoire. knock: clotheslined by a bar, head and chest thrown back, arms flung up.
 const knock=pose.knock||0;
 if(knock>0){rot[0][0]-=.10*knock;rot[1][0]-=.42*knock;rot[2][0]-=.60*knock;off[0][1]-=.05*knock;for(const [side,ua,fa]of SIDES){rot[ua]=mix3(rot[ua],[-2.05,0,side*.55],knock);rot[fa]=mix3(rot[fa],[-.55,0,0],knock);}}
 // winded: bent double after giving up, hands on the knees, shoulders heaving.
 const winded=pose.winded||0;
 if(winded>0){const heave=Math.sin(t*5.5)*.04;rot[0][0]+=.12*winded;rot[1][0]+=(.58+heave)*winded;rot[2][0]-=.30*winded;off[0][1]-=.11*winded;for(const [side,ua,fa,th,sh,ft]of SIDES){rot[ua]=mix3(rot[ua],[-.80,0,side*.22],winded);rot[fa]=mix3(rot[fa],[-.12,0,0],winded);rot[th]=mix3(rot[th],[-.32,0,side*.06],winded);rot[sh]=mix3(rot[sh],[.58,0,0],winded);rot[ft]=mix3(rot[ft],[0,0,0],winded);}}
 // guard: waiting at the kerb with both hands on the belt; whistle: right hand up to the mouth.
 const guard=pose.guard||0;
 if(guard>0){for(const [side,ua,fa]of SIDES){rot[ua]=mix3(rot[ua],[.35,0,side*.35],guard);rot[fa]=mix3(rot[fa],[-1.75,0,-side*.55],guard);}rot[1][0]-=.06*guard;rot[2][0]+=.04*guard;}
 const whistle=pose.whistle||0;
 if(whistle>0){rot[6]=mix3(rot[6],[-1.75,0,.40],whistle);rot[7]=mix3(rot[7],[-1.55,0,.10],whistle);rot[2][0]-=.15*whistle;}
 if(pose.arrest&&kind==='police'){
  // Chest out, left hand on the belt, right forefinger wagging.
  rot[1][0]=-.12;rot[2]=[-.12,Math.sin(t*.9)*.06,.08];
  rot[3]=[.35,0,-.35];rot[4]=[-1.75,0,.55];
  rot[6]=[-1.75,0,.40];rot[7]=[-1.35+Math.sin(t*7)*.22,0,0];
 }
 const world=[],out=new Float32Array(nb*16);
 for(let i=0;i<nb;i++){const parent=data.parents[i],p=rest[i],rel=p.map((v,k)=>v-(parent<0?0:rest[parent][k])+off[i][k]);world[i]=matrix(rel,...rot[i]);if(parent>=0)world[i]=mul(world[parent],world[i]);const inverse=matrix(p.map(v=>-v));out.set(mul(world[i],inverse),i*16);}
 return out;
}
class CharacterRenderer{
 constructor(gl){this.gl=gl;this.queue=[];this.meshes={};const NB=window.PIVNOY_MODELS.parents.length;const vs=`precision highp float;
attribute vec3 position;attribute vec3 normal;attribute vec3 albedo;attribute vec4 joints;attribute vec4 weights;attribute float material;attribute float ao;
uniform mat4 bones[${NB}];uniform vec3 origin;uniform float aspect;uniform float yaw;uniform vec3 tint;uniform float stretch;uniform mediump float shadowPass;
varying vec3 vNormal;varying vec3 vColor;varying vec3 vWorld;varying vec3 vRest;varying float vMaterial;varying float vFog;varying float vAO;
void main(){vec3 pos=position*(1./8192.);mat4 skin=bones[int(joints.x)]*weights.x+bones[int(joints.y)]*weights.y+bones[int(joints.z)]*weights.z+bones[int(joints.w)]*weights.w;vec3 p=(skin*vec4(pos,1.)).xyz;vec3 n=mat3(skin)*normal;
p.y*=1.+stretch;p.xz*=1.-stretch*.5;
mat2 turn=mat2(cos(yaw),-sin(yaw),sin(yaw),cos(yaw));p.xz=turn*p.xz;n.xz=turn*n.xz;p+=origin;if(shadowPass>.5){vec3 L=normalize(vec3(-.42,.78,-.50));p=p-L*((p.y-.045)/L.y);}
vWorld=p;vRest=pos;vNormal=normalize(n);vAO=ao;vMaterial=material;
vColor=albedo*((material>3.5&&material<5.5)?tint:vec3(1.));
vec3 v=p-vec3(0.,5.2,-8.);float y=v.y*.951+v.z*.309;float z=-v.y*.309+v.z*.951;gl_Position=vec4(v.x*1.64/aspect,y*1.64,z*1.002-.2002,z);vFog=clamp((z-30.)/70.,0.,1.);}`;
 const fs=`precision mediump float;varying vec3 vNormal;varying vec3 vColor;varying vec3 vWorld;varying vec3 vRest;varying float vMaterial;varying float vFog;varying float vAO;uniform float hue;uniform float day;uniform float dusk;uniform vec3 fog;uniform float scroll;
vec3 lamps(vec3 P,vec3 W,vec3 n){float k=floor((P.z-3.)/32.+.5);vec3 acc=vec3(0.);for(int i=-1;i<=1;i++){float lz=(k+float(i))*32.+3.-scroll;for(int s=-1;s<=1;s+=2){vec3 d=vec3(float(s)*3.95,5.1,lz)-W;float d2=dot(d,d);acc+=max(0.,dot(n,d*inversesqrt(d2)))*(4.4/(1.+d2*.06));}}return acc*vec3(1.,.80,.50);}
float hash(vec3 q){return fract(sin(dot(q,vec3(12.9898,78.233,37.719)))*43758.5453);}
float band(float lo,float hi){return step(lo,vMaterial)*(1.-step(hi,vMaterial));}uniform mediump float shadowPass;
void main(){ if(shadowPass>.5){gl_FragColor=vec4(.02,.04,.07,.46*(.25+.75*day));return;}
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
 // Per-instance garment recolour (scooter couriers): rotate cloth hues about the grey axis, leave skin, plastic and metal alone.
 if(hue!=0.){vec3 k=vec3(.57735);float cs=cos(hue),sn=sin(hue);vec3 rot=base*cs+cross(k,base)*sn+k*dot(k,base)*(1.-cs);base=mix(base,rot,clamp(isCloth+isQuilt,0.,1.));}
 // Lighting: key with soft wrap, hemisphere ambient, baked AO, fill, per-material specular, skin warmth, rim, fog.
 float ndl=dot(n,L),diff=max(0.,ndl),wrapd=clamp(ndl*.55+.45,0.,1.);
 float lit=mix(diff,wrapd,.30+.40*isSkin+.20*isHair);
 vec3 amb=mix(mix(vec3(.07,.09,.15),vec3(.12,.15,.24),n.y*.5+.5),mix(mix(vec3(.34,.30,.26),vec3(.62,.74,.84),n.y*.5+.5)*.88,vec3(.44,.31,.25),dusk*.45),day);
 float ao=mix(1.,vAO,.70-isEye*.55-isMetal*.25);
 float fill=max(0.,dot(n,normalize(vec3(.7,.3,.6))))*.26;
 vec3 sun=mix(vec3(1.04,.98,.90),vec3(1.22,.60,.32),dusk)*day+vec3(.12,.15,.24)*(1.-day);
 vec3 color=base*(amb*ao+sun*lit*1.02*ao+fill*vec3(.62,.72,.82)*(.3+.7*day));color+=base*lamps(vec3(vWorld.x,vWorld.y,vWorld.z+scroll),vWorld,n)*(1.-day)*ao;
 color+=base*vec3(.24,.07,.02)*(1.-lit)*isSkin*ao;
 vec3 h=normalize(L+eye);float ndh=max(0.,dot(n,h));
 float specPow=18.+isEye*70.+isMetal*60.+isPlastic*34.+isLeather*10.+isHair*8.;
 float specAmt=.03+isSkin*.08+isEye*.45+isLeather*.24+isMetal*.75+isPlastic*.36+isHair*.10+isRubber*.04;
 color+=vec3(1.,.95,.86)*pow(ndh,specPow)*specAmt*ao;
 color+=vec3(.9,.92,.9)*pow(ndh,5.)*.05*(isCloth+isQuilt+isKnit);
 float rim=pow(1.-max(0.,dot(n,eye)),3.);
 color+=vec3(.30,.44,.50)*rim*.30*ao;
 color=color*1.08/(1.+color*.09);
 color=mix(color,fog,vFog);
 gl_FragColor=vec4(color,1.);
}`;
 const compile=(type,source)=>{const sh=gl.createShader(type);gl.shaderSource(sh,source);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));return sh;};
 this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,vs));gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,fs));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
 this.attrs=[['position',3,0,gl.SHORT,false],['normal',3,6,gl.BYTE,true],['albedo',3,9,gl.UNSIGNED_BYTE,true],['joints',4,12,gl.UNSIGNED_BYTE,false],['weights',4,16,gl.UNSIGNED_BYTE,true],['material',1,20,gl.UNSIGNED_BYTE,false],['ao',1,21,gl.UNSIGNED_BYTE,true]].map(([n,size,offset,type,normalized])=>({loc:gl.getAttribLocation(this.program,n),size,offset,type,normalized}));
 this.uniforms=Object.fromEntries(['bones[0]','origin','aspect','yaw','tint','stretch','shadowPass','hue','day','dusk','fog','scroll'].map(n=>[n,gl.getUniformLocation(this.program,n)]));
 // Each character is a list of chunks (uint16 index limit); all chunks share the same bone matrices.
 this.skins={};for(const [kind,chunks]of Object.entries(window.PIVNOY_MODELS.meshes))this.meshes[kind]=this.decode(chunks);
 }
 decode(chunks){const gl=this.gl;return (Array.isArray(chunks)?chunks:[chunks]).map(encoded=>{const str=atob(encoded),raw=new Uint8Array(str.length);for(let i=0;i<str.length;i++)raw[i]=str.charCodeAt(i);const header=new DataView(raw.buffer),nv=header.getUint32(0,true),count=header.getUint32(4,true);if(raw.byteLength!==8+nv*STRIDE+count*2)throw Error('Invalid character asset');const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(raw.buffer,8,nv*STRIDE),gl.STATIC_DRAW);const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(raw.buffer.slice(8+nv*STRIDE,8+nv*STRIDE+count*2)),gl.STATIC_DRAW);return {vb,ib,count};});}
 // Boris = shared head/beard/hands + the outfit chunks from skins/boris-N.js, uploaded the first time they are needed; falls back to outfit 0.
 outfit(i){if(!this.skins[i]){const src=window.PIVNOY_SKINS&&window.PIVNOY_SKINS[i];if(src)this.skins[i]=this.decode(src);}return this.skins[i]||this.skins[0]||[];}
 parts(a){if(a.kind==='boris')return this.meshes.boris.concat(this.outfit(a.pose.skin|0));return this.meshes[a.kind==='rider'&&a.z>28?'rider_far':a.kind];}
 add(kind,x,y,z,t,c,slide,pose){this.queue.push({kind,x,y,z,t,c,slide,pose});}
 setLight(day,dusk,fog,scroll){this.light={day,dusk,fog,scroll};}
 draw(aspect,shadows=false){const gl=this.gl,u=this.uniforms,L=this.light||{day:1,dusk:0,fog:[.68,.79,.87],scroll:0};gl.useProgram(this.program);gl.uniform1f(u.aspect,aspect);gl.uniform1f(u.shadowPass,0);gl.uniform1f(u.day,L.day);gl.uniform1f(u.dusk,L.dusk);gl.uniform3f(u.fog,L.fog[0],L.fog[1],L.fog[2]);gl.uniform1f(u.scroll,L.scroll);
  const emit=a=>{const parts=this.parts(a);gl.uniformMatrix4fv(u['bones[0]'],false,a.bones);gl.uniform3f(u.origin,a.x,a.y,a.z);gl.uniform1f(u.yaw,a.pose.yaw||0);gl.uniform1f(u.hue,a.pose.hue||0);gl.uniform1f(u.stretch,a.pose.roll>0?0:(a.pose.stretch||0));gl.uniform3f(u.tint,1,1,1);
   for(const m of parts){gl.bindBuffer(gl.ARRAY_BUFFER,m.vb);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.ib);for(const at of this.attrs){gl.enableVertexAttribArray(at.loc);gl.vertexAttribPointer(at.loc,at.size,at.type,at.normalized,STRIDE,at.offset);}gl.drawElements(gl.TRIANGLES,m.count,gl.UNSIGNED_SHORT,0);}};
  for(const a of this.queue){a.bones=rig(a.t,a.slide,a.pose,a.kind);emit(a);}
  // Shadow pass: the same skinned meshes flattened onto the road along the sun, blended once per pixel via the stencil.
  if(shadows&&this.queue.length){gl.uniform1f(u.shadowPass,1);gl.enable(gl.STENCIL_TEST);gl.stencilFunc(gl.EQUAL,0,0xff);gl.stencilOp(gl.KEEP,gl.KEEP,gl.INCR);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);for(const a of this.queue)emit(a);gl.depthMask(true);gl.disable(gl.BLEND);gl.disable(gl.STENCIL_TEST);gl.uniform1f(u.shadowPass,0);}
  for(const a of this.attrs)gl.disableVertexAttribArray(a.loc);this.queue.length=0;}
}
window.CharacterRenderer=CharacterRenderer;window.PivnoyRig=rig;
})();
