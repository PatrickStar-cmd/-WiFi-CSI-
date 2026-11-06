import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {Reflector} from './vendor/Reflector.js';
import {poseAt,actionSegments,pointTimes,activeLinks,names,clamp} from './motion.mjs';

const v=a=>new THREE.Vector3(...a),Z=new THREE.Vector3(0,0,1),Y=new THREE.Vector3(0,1,0);
function material(color,extra={}){return new THREE.MeshStandardMaterial({color,roughness:.68,metalness:0,...extra});}
function dispose(group){group.traverse(o=>{o.geometry?.dispose();if(o.material){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});group.clear();}
function cylinder(a,b,r,color,opacity=1){
 const av=v(a),bv=v(b),d=bv.clone().sub(av);
 const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,d.length(),8),material(color,{transparent:opacity<1,opacity,depthWrite:opacity===1}));
 m.position.copy(av.add(bv).multiplyScalar(.5));m.quaternion.setFromUnitVectors(Y,d.normalize());m.castShadow=opacity===1;return m;
}
function box(size,position,color){const m=new THREE.Mesh(new THREE.BoxGeometry(...size),material(color));m.position.copy(v(position));m.castShadow=true;m.receiveShadow=true;return m;}
function sphere(radius,position,color){const m=new THREE.Mesh(new THREE.SphereGeometry(radius,16,12),material(color));m.position.copy(v(position));m.castShadow=true;return m;}

export class RoomView{
 constructor(container,labels,callbacks){
  this.container=container;this.labelsRoot=labels;this.callbacks=callbacks;this.layerState={fresnel:true,links:true,path:true,furniture:true};this.time=12.4;this.labels=[];this.selectedLink='AC';this.drag=null;this.addPoint=false;
  this.scene=new THREE.Scene();this.scene.background=null;
  this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});
  this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFShadowMap;
  this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.2;this.renderer.outputColorSpace=THREE.SRGBColorSpace;
  container.appendChild(this.renderer.domElement);
  this.camera=new THREE.OrthographicCamera(-6,6,3,-3,.05,100);this.camera.up.copy(Z);
  this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.dampingFactor=.1;this.controls.maxPolarAngle=Math.PI*.47;this.controls.minDistance=4;this.controls.maxDistance=30;
  this.scene.add(new THREE.HemisphereLight(0xffffff,0xd4dff0,2.1));
  const sun=new THREE.DirectionalLight(0xffffff,2.7);sun.position.set(1,-3,10);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-9;sun.shadow.camera.right=9;sun.shadow.camera.top=9;sun.shadow.camera.bottom=-9;sun.shadow.normalBias=.025;sun.shadow.bias=-.0001;sun.shadow.radius=6;this.scene.add(sun);
  this.staticGroup=new THREE.Group();this.linkGroup=new THREE.Group();this.pathGroup=new THREE.Group();this.furnitureGroup=new THREE.Group();this.actor=new THREE.Group();
  this.scene.add(this.staticGroup,this.linkGroup,this.pathGroup,this.furnitureGroup,this.actor);
  this.ray=new THREE.Raycaster();this.mouse=new THREE.Vector2();this.nodeGroups={};this.pointGroups={};this.furnitureGroups={};
  this.makeActor();this.setupEnvironment();this.bind();new ResizeObserver(()=>this.resize()).observe(container);
 }
 setupEnvironment(){
  const env=document.createElement('canvas');env.width=512;env.height=256;const c=env.getContext('2d'),g=c.createLinearGradient(0,0,0,256);g.addColorStop(0,'#f8fbff');g.addColorStop(.45,'#e2e9f5');g.addColorStop(.7,'#bbc9df');g.addColorStop(1,'#dce3ef');c.fillStyle=g;c.fillRect(0,0,512,256);c.fillStyle='#fff';c.fillRect(45,20,70,75);c.fillStyle='#f4f9ff';c.fillRect(310,30,100,60);const texture=new THREE.CanvasTexture(env);texture.mapping=THREE.EquirectangularReflectionMapping;texture.colorSpace=THREE.SRGBColorSpace;this.scene.environment=texture;this.scene.environmentIntensity=.28;
 }
 setupFloor(w,d){
  const floor=this.staticGroup.children[0];floor.material.dispose();floor.material=new THREE.MeshPhysicalMaterial({color:0xe9eef6,roughness:.28,metalness:.08,clearcoat:.5,clearcoatRoughness:.35});
  this.reflection=new Reflector(new THREE.PlaneGeometry(w,d),{color:0xe6ecf7,textureWidth:1024,textureHeight:512,clipBias:.001});this.reflection.position.set(w/2,d/2,.009);this.reflection.material.transparent=true;this.reflection.material.depthWrite=false;this.reflection.material.fragmentShader=this.reflection.material.fragmentShader.replace('blendOverlay( base.rgb, color ), 1.0','blendOverlay( base.rgb, color ), 0.12');this.scene.add(this.reflection);
 }
 setPrediction(prediction){
  this.personLabel.el.innerHTML=`${names[this.currentKind]} · ${this.time.toFixed(1)} s`+(prediction?` <span class="tag-arrow">→</span> ${prediction.label} <b class="${prediction.agreement==='✓'?'good':'bad'}">${prediction.agreement}</b>`:'');
 }
 layoutLabels(){
  const root=this.container.getBoundingClientRect(),w=root.width,h=root.height,svg=document.getElementById('leaders');let markup='';
  const placed=[...document.querySelectorAll('.view-controls>.segmented,.view-controls>#layers-button')].map(e=>{const r=e.getBoundingClientRect();return{x:r.x-root.x,y:r.y-root.y,w:r.width,h:r.height};});
  const overlap=(a,b)=>a.x<b.x+b.w+12&&a.x+a.w+12>b.x&&a.y<b.y+b.h+12&&a.y+a.h+12>b.y;
  const keyLabels=this.labels.filter(l=>l.el.matches('.person,.current-point,.fresnel')&&l.el.style.display!=='none'&&l.el.style.visibility!=='hidden');
  keyLabels.sort((a,b)=>Number(b.el.classList.contains('person'))-Number(a.el.classList.contains('person')));
  for(const l of keyLabels){const p=l.position.clone().project(this.camera),ax=(p.x*.5+.5)*w,ay=(-p.y*.5+.5)*h,r=l.el.getBoundingClientRect(),offset=l.offset||[0,0];let choice;
   for(const [dx,dy]of [[0,0],[0,44],[0,-44],[80,0],[-80,0],[0,88],[0,-88],[160,44],[-160,44]]){const candidate={x:clamp(ax+offset[0]+dx-r.width/2,12,Math.max(12,w-r.width-12)),y:clamp(ay+offset[1]+dy-r.height/2,12,Math.max(12,h-r.height-12)),w:r.width,h:r.height};choice=candidate;if(!placed.some(b=>overlap(candidate,b)))break;}
   l.el.style.left=choice.x+choice.w/2+'px';l.el.style.top=choice.y+choice.h/2+'px';placed.push(choice);
   const ex=clamp(ax,choice.x+9,choice.x+choice.w-9),ey=ay>choice.y+choice.h/2?choice.y+choice.h:choice.y,color=l.el.classList.contains('fresnel')?'#a5b3e0':'#a9b8cc';markup+=`<path d="M ${ax} ${ay} L ${ex} ${ey}" stroke="${color}" stroke-width="1" fill="none" opacity=".8"/><circle cx="${ax}" cy="${ay}" r="2" fill="${color}" opacity=".8"/>`;
  }
  svg.innerHTML=markup;
 }
 hover(e){
  this.rayFromEvent(e);let entity=null;for(const hit of this.ray.intersectObjects([this.staticGroup,this.pathGroup,this.furnitureGroup],true)){let o=hit.object;while(o&&!o.userData.type)o=o.parent;if(o?.userData.type){entity=o;break;}}
  for(const l of this.labels)l.el.classList.toggle('hovered',!!entity&&(l.node===entity.userData.id||l.point===entity.userData.id));this.renderer.domElement.style.cursor=entity?'grab':this.addPoint?'crosshair':'default';
 }
 makeActor(){
  const shirt=0x67849f,pants=0x94a4b9,skin=0xcbb5a7;
  this.torso=new THREE.Mesh(new THREE.CapsuleGeometry(.18,.30,6,18),material(shirt));this.torso.rotation.x=Math.PI/2;this.torso.scale.set(.85,1,1.25);this.actor.add(this.torso);
  this.head=sphere(.13,[0,0,1.62],skin);this.actor.add(this.head);
  this.neck=cylinder([0,0,1.45],[0,0,1.59],.065,skin);this.actor.add(this.neck);
  this.hair=new THREE.Mesh(new THREE.SphereGeometry(.133,16,10,0,Math.PI*2,0,Math.PI*.55),material(0x393832));this.hair.rotation.x=Math.PI/2;this.actor.add(this.hair);
  this.nose=sphere(.028,[0,0,0],skin);this.actor.add(this.nose);
  this.limbs=[];
  for(let i=0;i<8;i++){
   const m=new THREE.Mesh(new THREE.CylinderGeometry(i<4?.045:.063,i<4?.047:.063,1,18),material(i<2?shirt:i<4?skin:pants));m.castShadow=true;this.actor.add(m);this.limbs.push(m);
  }
  this.hands=[sphere(.055,[0,0,0],skin),sphere(.055,[0,0,0],skin)];this.actor.add(...this.hands);
  this.shoulderJoints=[sphere(.065,[0,0,0],shirt),sphere(.065,[0,0,0],shirt)];
  this.elbowJoints=[sphere(.047,[0,0,0],skin),sphere(.047,[0,0,0],skin)];this.actor.add(...this.shoulderJoints,...this.elbowJoints);
  this.feet=[box([.23,.10,.09],[0,0,0],0xe0ddd4),box([.23,.10,.09],[0,0,0],0xe0ddd4)];this.actor.add(...this.feet);
  this.knees=[sphere(.065,[0,0,0],pants),sphere(.065,[0,0,0],pants)];this.actor.add(...this.knees);
 }
 label(text,position,cls='',layer=null){
  const el=document.createElement('div');el.className='world-label '+cls;el.textContent=text;this.labelsRoot.appendChild(el);
  const entry={el,position:v(position),layer};this.labels.push(entry);return entry;
 }
 setProject(p){
  this.project=p;if(this.reflection){this.scene.remove(this.reflection);this.reflection.dispose();this.reflection.geometry.dispose();this.reflection=null;}dispose(this.staticGroup);dispose(this.linkGroup);dispose(this.pathGroup);dispose(this.furnitureGroup);
  this.labels.forEach(x=>x.el.remove());this.labels=[];this.nodeGroups={};this.pointGroups={};this.furnitureGroups={};this.pathLines=[];
  const [w,d,h]=p.room;
  this.staticGroup.add(box([w,d,.12],[w/2,d/2,-.06],0xe8edf5));
  this.staticGroup.add(box([.09,d,Math.min(h,1.05)],[0,d/2,Math.min(h,1.05)/2],0xe1e7f0));
  this.staticGroup.add(box([w,.09,Math.min(h,1.05)],[w/2,d,Math.min(h,1.05)/2],0xe8edf5));
  this.setupFloor(w,d);
  const gridMaterial=new THREE.LineBasicMaterial({color:0xc9d4e4,transparent:true,opacity:.17});
  const grid=[];for(let x=0;x<=w+.001;x+=.5)grid.push(x,0,.008,x,d,.008);for(let y=0;y<=d+.001;y+=.5)grid.push(0,y,.008,w,y,.008);
  const gg=new THREE.BufferGeometry();gg.setAttribute('position',new THREE.Float32BufferAttribute(grid,3));this.staticGroup.add(new THREE.LineSegments(gg,gridMaterial));
  for(const n of p.nodes){
   const g=new THREE.Group();g.position.set(n.position[0],n.position[1],0);g.userData={type:'node',id:n.id};
   g.add(box([.19,.19,.035],[0,0,.025],0x8f9cac),cylinder([0,0,.04],[0,0,n.position[2]],.025,0xa7b3c3));
   const pick=new THREE.Mesh(new THREE.SphereGeometry(.18,8,6),new THREE.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));pick.position.z=n.position[2];g.add(pick);this.staticGroup.add(g);this.nodeGroups[n.id]=g;
   const lab=this.label('',n.position,'node-marker');lab.node=n.id;
   const marker=document.createElement('span');marker.className='marker '+(n.role==='tx'?'tx':'');const name=document.createElement('span');name.className='node-name';name.textContent=n.id;lab.el.append(marker,name);
   if(n.antennas>1){for(let i=0;i<n.antennas;i++)g.add(cylinder([-.1+i*.2/(n.antennas-1),0,n.position[2]-.05],[-.1+i*.2/(n.antennas-1),0,n.position[2]+.11],.012,n.role==='tx'?0x28549b:0x31968d));}
  }
  for(const f of p.furniture){
   const g=new THREE.Group();g.position.copy(v(f.position));g.userData={type:'furniture',id:f.id};const [fw,fd,fh]=f.size;
   if(f.kind==='table'){
    g.add(box([fw,fd,.065],[0,0,fh-.033],0xc1ccda));
    for(const x of [-fw/2+.06,fw/2-.06])for(const y of [-fd/2+.06,fd/2-.06])g.add(box([.055,.055,fh-.065],[x,y,(fh-.065)/2],0xc1ccda));
   }else{
    const seat=fh*.53;g.add(box([fw,fd,.06],[0,0,seat],0xc1ccda),box([fw,.055,fh-seat],[0,fd/2-.025,(fh+seat)/2],0xc1ccda));
    for(const x of [-fw/2+.05,fw/2-.05])for(const y of [-fd/2+.05,fd/2-.05])g.add(box([.04,.04,seat],[x,y,seat/2],0xc1ccda));
   }
   g.traverse(o=>{if(o.isMesh){o.material.transparent=true;o.material.opacity=.72;o.castShadow=false;}});this.furnitureGroup.add(g);this.furnitureGroups[f.id]=g;
  }
  const arrivals=pointTimes(p);
  for(const point of p.points){
   const g=new THREE.Group();g.position.copy(v(point.position));g.userData={type:'point',id:point.id};
   const disk=new THREE.Mesh(new THREE.CylinderGeometry(.075,.075,.025,20),material(0xfaf9f5));disk.rotation.x=Math.PI/2;disk.position.z=.027;g.add(disk);
   const ring=new THREE.Mesh(new THREE.TorusGeometry(.076,.015,6,20),material(0x8e9fbc));ring.position.z=.04;g.add(ring);
   this.pathGroup.add(g);this.pointGroups[point.id]=g;
   const l=this.label(`${point.id} · ${arrivals[point.id]===undefined?'—':arrivals[point.id].toFixed(1)+' s'}`,point.position,'','path');l.offset=point.id==='P1'?[64,28]:[0,20];l.point=point.id;l.el.classList.add(point.id==='P1'?'current-point':'hover-only');
  }
  for(const s of actionSegments(p).filter(x=>x.kind==='walk')){
   const makeLine=dashed=>{const geometry=new THREE.BufferGeometry().setFromPoints([v(s.p0),v(s.p1)]);const mat=dashed?new THREE.LineDashedMaterial({color:0xc3cfdf,dashSize:.12,gapSize:.09}):new THREE.LineBasicMaterial({color:0x8296b3});const line=new THREE.Line(geometry,mat);line.computeLineDistances();this.pathGroup.add(line);return line;};
   this.pathLines.push({s,done:makeLine(false),future:makeLine(true)});
  }
  const origin=[.15,.15,.01];for(const [dir,color] of [[[1,0,0],0x956c5c],[[0,1,0],0x7d8f64],[[0,0,1],0x6c83a0]])this.staticGroup.add(new THREE.ArrowHelper(v(dir),v(origin),.35,color,.075,.04));
  this.label('X',[.56,.15,.01],'axis-label');this.label('Y',[.15,.56,.01],'axis-label');this.label('Z',[.15,.15,.45],'axis-label');
  this.personLabel=this.label('',[0,0,2],'person');this.personLabel.offset=[86,-16];this.buildLinks();this.setTime(this.time);this.applyLayers();
  if(!this.initialized){this.resetCamera();this.initialized=true;}
 }
 buildLinks(){
  this.flow=[];dispose(this.linkGroup);this.labels.filter(l=>l.layer==='fresnel').forEach(l=>l.el.remove());this.labels=this.labels.filter(l=>l.layer!=='fresnel');
  const nodes=Object.fromEntries(this.project.nodes.map(n=>[n.id,n.position]));this.fresnelGroup=new THREE.Group();this.otherLinks=new THREE.Group();this.linkGroup.add(this.otherLinks,this.fresnelGroup);
  for(const link of activeLinks(this.project)){
   const a=nodes[link[0]],b=nodes[link[1]],selected=link===this.selectedLink;
   this.otherLinks.add(cylinder(a,b,selected?.006:.003,selected?0x5489df:0x9bb0cb,selected?.9:.14));
   if(selected){
    this.otherLinks.add(cylinder(a,b,.023,0x80a8ff,.08),cylinder(a,b,.012,0x7aa4ff,.14));
    this.flow=[];
    for(let i=0;i<3;i++){const dot=sphere(.024,[0,0,0],0xaacaff);dot.material=new THREE.MeshBasicMaterial({color:0xd7e5ff});const halo=sphere(.057,[0,0,0],0x8eb6ff);halo.material=new THREE.MeshBasicMaterial({color:0x8eb6ff,transparent:true,opacity:.22,depthWrite:false});this.otherLinks.add(dot,halo);this.flow.push({dot,halo,a:v(a),b:v(b),phase:i/3});}
   }
   if(selected){
    const av=v(a),bv=v(b),distance=av.distanceTo(bv),lambda=299792458/(this.project.radio.frequencyGHz*1e9),major=(distance+lambda/2)/2,minor=Math.sqrt(major*major-distance*distance/4);
    const group=new THREE.Group();group.position.copy(av.add(bv).multiplyScalar(.5));group.quaternion.setFromUnitVectors(new THREE.Vector3(1,0,0),v(b).sub(v(a)).normalize());
    const ellipsoid=new THREE.Mesh(new THREE.SphereGeometry(1,40,18),new THREE.ShaderMaterial({transparent:true,depthWrite:false,side:THREE.FrontSide,
     vertexShader:'varying vec3 n;varying vec3 p;void main(){n=normalize(normalMatrix*normal);vec4 mv=modelViewMatrix*vec4(position,1.);p=mv.xyz;gl_Position=projectionMatrix*mv;}',
     fragmentShader:'varying vec3 n;varying vec3 p;void main(){float rim=pow(1.-abs(dot(normalize(n),normalize(-p))),2.);vec3 c=mix(vec3(.61,.60,1.),vec3(.70,.72,1.),rim);gl_FragColor=vec4(c,.025+.22*rim);}'}));ellipsoid.scale.set(major,minor,minor);group.add(ellipsoid);
    for(const plane of [0,1]){const pts=Array.from({length:81},(_,i)=>{let theta=i/80*Math.PI*2;return new THREE.Vector3(major*Math.cos(theta),plane===0?minor*Math.sin(theta):0,plane===1?minor*Math.sin(theta):0);});group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0xa6a3ec,transparent:true,opacity:.58})));}
    this.fresnelGroup.add(group);
    this.label(`${link} 第一菲涅尔区`,a.map((v,i)=>v+.3*(b[i]-v)+(i===2?.05:0)),'fresnel','fresnel').offset=[-162,-63];
   }
  }
  this.applyLayers();
 }
 setLink(link){this.selectedLink=link;if(this.project)this.buildLinks();}
 setLayer(name,value){this.layerState[name]=value;this.applyLayers();}
 applyLayers(){this.otherLinks&&(this.otherLinks.visible=this.layerState.links);this.fresnelGroup&&(this.fresnelGroup.visible=this.layerState.fresnel);this.pathGroup.visible=this.layerState.path;this.furnitureGroup.visible=this.layerState.furniture;}
 resetCamera(top=false){
  if(!this.project)return;const [w,d]=this.project.room;
  this.top=top;this.camera.zoom=1;this.controls.target.set(w/2,d/2,.5);
  this.camera.position.set(top?w/2:w/2+7,top?d/2-.001:d/2-15,top?16:4);
  this.camera.up.set(0,0,1);this.controls.update();this.fitCamera();
 }
 fitCamera(){
  if(!this.project)return;
  const [w,d]=this.project.room,W=this.container.clientWidth,H=this.container.clientHeight;
  this.camera.updateMatrixWorld();
  const pts=[[0,0,0],[w,0,0],[0,d,0],[w,d,0],[0,d,1.05],[w,d,1.05],[0,0,1.05],...this.project.nodes.map(n=>n.position),...this.project.points.map(p=>[p.position[0],p.position[1],1.96])];
  const q=pts.map(a=>v(a).applyMatrix4(this.camera.matrixWorldInverse));
  const xmin=Math.min(...q.map(p=>p.x)),xmax=Math.max(...q.map(p=>p.x)),ymin=Math.min(...q.map(p=>p.y)),ymax=Math.max(...q.map(p=>p.y));
  const aspect=W/H,half=Math.max((ymax-ymin)/.79,(xmax-xmin)/aspect/.91)/2,cx=(xmin+xmax)/2,cy=(ymin+ymax)/2+half*.035;
  this.camera.left=cx-half*aspect;this.camera.right=cx+half*aspect;this.camera.top=cy+half;this.camera.bottom=cy-half;this.camera.updateProjectionMatrix();
 }
 resize(){const w=this.container.clientWidth,h=this.container.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h,false);this.fitCamera();}
 setTime(time,provided=null){
  this.time=time;if(!this.project)return;
  const p=provided||poseAt(this.project,time),root=v(p.root),angle=p.angle,forward=new THREE.Vector3(Math.cos(angle),Math.sin(angle),0),side=new THREE.Vector3(-Math.sin(angle),Math.cos(angle),0);
  this.actor.visible=this.project.present;this.personLabel.position.copy(root).add(new THREE.Vector3(0,0,1.96));this.personLabel.el.textContent=`${names[p.kind]} · ${time.toFixed(1)} s`;this.currentKind=p.kind;this.personLabel.el.style.visibility=this.project.present?'visible':'hidden';
  const current=this.project.points.reduce((a,b)=>v(a.position).distanceTo(root)<=v(b.position).distanceTo(root)?a:b);for(const l of this.labels.filter(l=>l.point)){l.el.classList.toggle('current-point',l.point===current.id);l.el.classList.toggle('hover-only',l.point!==current.id);l.offset=l.point===current.id?[64,28]:[0,20];}
  this.torso.position.copy(root).add(new THREE.Vector3(0,0,1.17));this.torso.quaternion.setFromAxisAngle(Z,angle).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),Math.PI/2));
  this.neck.position.copy(root).add(new THREE.Vector3(0,0,1.52));
  this.head.position.copy(root).add(new THREE.Vector3(0,0,1.64));this.hair.position.copy(this.head.position);this.hair.quaternion.copy(this.torso.quaternion);this.nose.position.copy(this.head.position).addScaledVector(forward,.125).add(new THREE.Vector3(0,0,-.035));
  const shoulder=p.shoulders.map(v),hands=[v(p.points[1]),v(p.points[2])],elbows=p.elbows.map(v);
  const setLimb=(i,a,b)=>{const delta=b.clone().sub(a);this.limbs[i].position.copy(a).add(b).multiplyScalar(.5);this.limbs[i].scale.y=delta.length();this.limbs[i].quaternion.setFromUnitVectors(Y,delta.normalize());};
  for(let i=0;i<2;i++){setLimb(i,shoulder[i],elbows[i]);setLimb(i+2,elbows[i],hands[i]);this.hands[i].position.copy(hands[i]);this.shoulderJoints[i].position.copy(shoulder[i]);this.elbowJoints[i].position.copy(elbows[i]);}
  const gait=p.kind==='walk'?.12*Math.sin(2*Math.PI*1.5*(p.elapsed??time)):0;
  for(let i=0;i<2;i++){const sign=i===0?1:-1,hip=root.clone().addScaledVector(side,.105*sign).add(new THREE.Vector3(0,0,.85)),foot=root.clone().addScaledVector(side,.11*sign).addScaledVector(forward,gait*sign).add(new THREE.Vector3(0,0,.075));const knee=hip.clone().lerp(foot,.52).addScaledVector(forward,.045);setLimb(4+i,hip,knee);setLimb(6+i,knee,foot);this.knees[i].position.copy(knee);this.feet[i].position.copy(foot);this.feet[i].rotation.z=angle;}
  for(const item of this.pathLines){const {s,done,future}=item;let middle=time<=s.start?s.p0:time>=s.end?s.p1:poseAt(this.project,time).root;for(const [line,a,b,visible] of [[done,s.p0,middle,time>s.start],[future,middle,s.p1,time<s.end]]){line.visible=visible;const ar=line.geometry.attributes.position;ar.setXYZ(0,a[0],a[1],.035);ar.setXYZ(1,b[0],b[1],.035);ar.needsUpdate=true;line.geometry.computeBoundingSphere();line.computeLineDistances();}}
 }
 render(){
  this.controls.update();
  for(const f of this.flow||[]){const u=(performance.now()*.00007+f.phase)%1;f.dot.position.copy(f.a).lerp(f.b,u);f.halo.position.copy(f.dot.position);}
  this.renderer.render(this.scene,this.camera);
  const w=this.container.clientWidth,h=this.container.clientHeight;
  for(const label of this.labels){
   if(label.layer&&!this.layerState[label.layer]){label.el.style.display='none';continue;}
   const p=label.position.clone().project(this.camera);label.el.style.display=p.z>1||p.z< -1?'none':'';
   label.el.style.left=((p.x*.5+.5)*w+(label.offset?.[0]||0))+'px';label.el.style.top=((-p.y*.5+.5)*h+(label.offset?.[1]||0))+'px';
  }
  this.layoutLabels();
  const pop=document.getElementById('furniture-popover');
  if(this.selectedFurniture&&this.layerState.furniture){const f=this.project.furniture.find(x=>x.id===this.selectedFurniture);if(f){const p=v(f.position).add(new THREE.Vector3(0,0,f.size[2]+.25)).project(this.camera);pop.textContent=`${f.name} · 反射系数 ${f.reflection} · 等效散射点 ×4`;pop.style.left=((p.x*.5+.5)*w)+'px';pop.style.top=((-p.y*.5+.5)*h)+'px';pop.classList.remove('hidden');}}else pop.classList.add('hidden');
 }
 rayFromEvent(e){const rect=this.container.getBoundingClientRect();this.mouse.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);this.ray.setFromCamera(this.mouse,this.camera);}
 bind(){
  const el=this.renderer.domElement;
  el.addEventListener('pointerdown',e=>{
   if(e.button!==0||!this.project||document.body.classList.contains('computing'))return;
   this.rayFromEvent(e);let entity=null;
   const hits=this.ray.intersectObjects([this.staticGroup,this.pathGroup,this.furnitureGroup],true);
   for(const h of hits){let obj=h.object;while(obj&&!obj.userData.type)obj=obj.parent;if(obj?.userData.type){entity=obj;break;}}
   if(entity){const {type,id}=entity.userData;
    const record=type==='node'?this.project.nodes.find(n=>n.id===id):type==='point'?this.project.points.find(n=>n.id===id):this.project.furniture.find(n=>n.id===id);
    if(type==='furniture'){this.selectedFurniture=id;this.callbacks.select(type,id);return;}
    const plane=new THREE.Plane(Z,-(type==='node'?record.position[2]:0)),point=new THREE.Vector3();this.ray.ray.intersectPlane(plane,point);
    this.drag={type,id,group:entity,plane,offset:point.sub(v(record.position)),initial:[...record.position],position:[...record.position],start:[e.clientX,e.clientY],moved:false};
    this.controls.enabled=false;el.setPointerCapture(e.pointerId);this.callbacks.select(type,id);
   }else if(this.addPoint){const point=new THREE.Vector3();if(this.ray.ray.intersectPlane(new THREE.Plane(Z,0),point))this.callbacks.addPoint([point.x,point.y,0]);}
  });
  el.addEventListener('pointermove',e=>{
   if(!this.drag){this.hover(e);return;}const d=this.drag;
   if(Math.hypot(e.clientX-d.start[0],e.clientY-d.start[1])<3&&!d.moved)return;
   d.moved=true;this.rayFromEvent(e);const p=new THREE.Vector3();if(!this.ray.ray.intersectPlane(d.plane,p))return;p.sub(d.offset);
   d.position=[Math.round(clamp(p.x,.2,this.project.room[0]-.2)*100)/100,Math.round(clamp(p.y,.2,this.project.room[1]-.2)*100)/100,d.initial[2]];
   d.group.position.x=d.position[0];d.group.position.y=d.position[1];
   const label=this.labels.find(l=>d.type==='node'?l.node===d.id:l.point===d.id);if(label){label.position.x=d.position[0];label.position.y=d.position[1];}
  });
  const release=e=>{if(!this.drag)return;const d=this.drag;this.drag=null;this.controls.enabled=true;if(el.hasPointerCapture(e.pointerId))el.releasePointerCapture(e.pointerId);if(d.moved)this.callbacks.move(d.type,d.id,d.position);};
  el.addEventListener('pointerup',release);el.addEventListener('pointercancel',release);
 }
}
