import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {poseAt,actionSegments,pointTimes,activeLinks,names,clamp} from './motion.mjs';

const v=a=>new THREE.Vector3(...a),Z=new THREE.Vector3(0,0,1),Y=new THREE.Vector3(0,1,0);
function material(color,extra={}){return new THREE.MeshStandardMaterial({color,roughness:.85,metalness:0,...extra});}
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
  this.camera=new THREE.PerspectiveCamera(36,1,.05,100);this.camera.up.copy(Z);
  this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.dampingFactor=.1;this.controls.maxPolarAngle=Math.PI*.47;this.controls.minDistance=4;this.controls.maxDistance=30;
  this.scene.add(new THREE.HemisphereLight(0xffffff,0xb6a58b,2.4));
  const sun=new THREE.DirectionalLight(0xffffff,3.5);sun.position.set(1,-3,10);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-9;sun.shadow.camera.right=9;sun.shadow.camera.top=9;sun.shadow.camera.bottom=-9;sun.shadow.normalBias=.025;sun.shadow.bias=-.0001;sun.shadow.radius=4;this.scene.add(sun);
  this.staticGroup=new THREE.Group();this.linkGroup=new THREE.Group();this.pathGroup=new THREE.Group();this.furnitureGroup=new THREE.Group();this.actor=new THREE.Group();
  this.scene.add(this.staticGroup,this.linkGroup,this.pathGroup,this.furnitureGroup,this.actor);
  this.ray=new THREE.Raycaster();this.mouse=new THREE.Vector2();this.nodeGroups={};this.pointGroups={};this.furnitureGroups={};
  this.makeActor();this.bind();new ResizeObserver(()=>this.resize()).observe(container);
 }
 makeActor(){
  const shirt=0xa2a49c,pants=0xb4ab98,skin=0xd5b297;
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
  this.project=p;dispose(this.staticGroup);dispose(this.linkGroup);dispose(this.pathGroup);dispose(this.furnitureGroup);
  this.labels.forEach(x=>x.el.remove());this.labels=[];this.nodeGroups={};this.pointGroups={};this.furnitureGroups={};this.pathLines=[];
  const [w,d,h]=p.room;
  this.staticGroup.add(box([w,d,.12],[w/2,d/2,-.06],0xdad2c4));
  this.staticGroup.add(box([.09,d,Math.min(h,2.5)],[0,d/2,Math.min(h,2.5)/2],0xe7dfd2));
  this.staticGroup.add(box([w,.09,Math.min(h,2.5)],[w/2,d,Math.min(h,2.5)/2],0xe9e1d4));
  const gridMaterial=new THREE.LineBasicMaterial({color:0xf4efe7,transparent:true,opacity:.65});
  const grid=[];for(let x=0;x<=w+.001;x+=.5)grid.push(x,0,.008,x,d,.008);for(let y=0;y<=d+.001;y+=.5)grid.push(0,y,.008,w,y,.008);
  const gg=new THREE.BufferGeometry();gg.setAttribute('position',new THREE.Float32BufferAttribute(grid,3));this.staticGroup.add(new THREE.LineSegments(gg,gridMaterial));
  for(const n of p.nodes){
   const g=new THREE.Group();g.position.set(n.position[0],n.position[1],0);g.userData={type:'node',id:n.id};
   g.add(box([.19,.19,.035],[0,0,.025],0x777a75),cylinder([0,0,.04],[0,0,n.position[2]],.025,0x85867f));
   const pick=new THREE.Mesh(new THREE.SphereGeometry(.18,8,6),new THREE.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));pick.position.z=n.position[2];g.add(pick);this.staticGroup.add(g);this.nodeGroups[n.id]=g;
   const lab=this.label('',n.position,'node-marker');lab.node=n.id;
   const marker=document.createElement('span');marker.className='marker '+(n.role==='tx'?'tx':'');const name=document.createElement('span');name.className='node-name';name.textContent=n.id;lab.el.append(marker,name);
   if(n.antennas>1){for(let i=0;i<n.antennas;i++)g.add(cylinder([-.1+i*.2/(n.antennas-1),0,n.position[2]-.05],[-.1+i*.2/(n.antennas-1),0,n.position[2]+.11],.012,n.role==='tx'?0x28549b:0x31968d));}
  }
  for(const f of p.furniture){
   const g=new THREE.Group();g.position.copy(v(f.position));g.userData={type:'furniture',id:f.id};const [fw,fd,fh]=f.size;
   if(f.kind==='table'){
    g.add(box([fw,fd,.065],[0,0,fh-.033],0xbda07d));
    for(const x of [-fw/2+.06,fw/2-.06])for(const y of [-fd/2+.06,fd/2-.06])g.add(box([.055,.055,fh-.065],[x,y,(fh-.065)/2],0xb99a74));
   }else{
    const seat=fh*.53;g.add(box([fw,fd,.06],[0,0,seat],0xbca180),box([fw,.055,fh-seat],[0,fd/2-.025,(fh+seat)/2],0xbca180));
    for(const x of [-fw/2+.05,fw/2-.05])for(const y of [-fd/2+.05,fd/2-.05])g.add(box([.04,.04,seat],[x,y,seat/2],0xaa8b69));
   }
   this.furnitureGroup.add(g);this.furnitureGroups[f.id]=g;
  }
  const arrivals=pointTimes(p);
  for(const point of p.points){
   const g=new THREE.Group();g.position.copy(v(point.position));g.userData={type:'point',id:point.id};
   const disk=new THREE.Mesh(new THREE.CylinderGeometry(.075,.075,.025,20),material(0xfaf9f5));disk.rotation.x=Math.PI/2;disk.position.z=.027;g.add(disk);
   const ring=new THREE.Mesh(new THREE.TorusGeometry(.076,.015,6,20),material(0x8f8b82));ring.position.z=.04;g.add(ring);
   this.pathGroup.add(g);this.pointGroups[point.id]=g;
   const l=this.label(`${point.id} · ${arrivals[point.id]===undefined?'—':arrivals[point.id].toFixed(1)+' s'}`,point.position,'','path');l.offset=[0,24];l.point=point.id;
  }
  for(const s of actionSegments(p).filter(x=>x.kind==='walk')){
   const makeLine=dashed=>{const geometry=new THREE.BufferGeometry().setFromPoints([v(s.p0),v(s.p1)]);const mat=dashed?new THREE.LineDashedMaterial({color:0x8c8981,dashSize:.12,gapSize:.09}):new THREE.LineBasicMaterial({color:0x807c72});const line=new THREE.Line(geometry,mat);line.computeLineDistances();this.pathGroup.add(line);return line;};
   this.pathLines.push({s,done:makeLine(false),future:makeLine(true)});
  }
  const origin=[.15,.15,.01];for(const [dir,color] of [[[1,0,0],0x956c5c],[[0,1,0],0x7d8f64],[[0,0,1],0x6c83a0]])this.staticGroup.add(new THREE.ArrowHelper(v(dir),v(origin),.35,color,.075,.04));
  this.label('原点 (0,0,0)',origin,'origin').offset=[0,29];
  this.personLabel=this.label('',[0,0,2],'person');this.buildLinks();this.setTime(this.time);this.applyLayers();
  if(!this.initialized){this.resetCamera();this.initialized=true;}
 }
 buildLinks(){
  dispose(this.linkGroup);this.labels.filter(l=>l.layer==='fresnel').forEach(l=>l.el.remove());this.labels=this.labels.filter(l=>l.layer!=='fresnel');
  const nodes=Object.fromEntries(this.project.nodes.map(n=>[n.id,n.position]));this.fresnelGroup=new THREE.Group();this.otherLinks=new THREE.Group();this.linkGroup.add(this.otherLinks,this.fresnelGroup);
  for(const link of activeLinks(this.project)){
   const a=nodes[link[0]],b=nodes[link[1]],selected=link===this.selectedLink;
   this.otherLinks.add(cylinder(a,b,selected?.014:.006,selected?0x28549b:0x7b90aa,selected?.9:.22));
   if(selected){
    const av=v(a),bv=v(b),distance=av.distanceTo(bv),lambda=299792458/(this.project.radio.frequencyGHz*1e9),major=(distance+lambda/2)/2,minor=Math.sqrt(major*major-distance*distance/4);
    const group=new THREE.Group();group.position.copy(av.add(bv).multiplyScalar(.5));group.quaternion.setFromUnitVectors(new THREE.Vector3(1,0,0),v(b).sub(v(a)).normalize());
    const ellipsoid=new THREE.Mesh(new THREE.SphereGeometry(1,40,18),new THREE.MeshBasicMaterial({color:0x7196d5,transparent:true,opacity:.12,depthWrite:false,side:THREE.DoubleSide}));ellipsoid.scale.set(major,minor,minor);group.add(ellipsoid);
    for(const plane of [0,1]){const pts=Array.from({length:81},(_,i)=>{let theta=i/80*Math.PI*2;return new THREE.Vector3(major*Math.cos(theta),plane===0?minor*Math.sin(theta):0,plane===1?minor*Math.sin(theta):0);});group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x7b9dcd,transparent:true,opacity:.4})));}
    this.fresnelGroup.add(group);
    this.label(`${link} 第一菲涅尔区`,[(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2+.55],'fresnel','fresnel').offset=[-95,0];
   }
  }
  this.applyLayers();
 }
 setLink(link){this.selectedLink=link;if(this.project)this.buildLinks();}
 setLayer(name,value){this.layerState[name]=value;this.applyLayers();}
 applyLayers(){this.otherLinks&&(this.otherLinks.visible=this.layerState.links);this.fresnelGroup&&(this.fresnelGroup.visible=this.layerState.fresnel);this.pathGroup.visible=this.layerState.path;this.furnitureGroup.visible=this.layerState.furniture;}
 resetCamera(top=false){if(!this.project)return;const [w,d]=this.project.room;this.controls.target.set(w/2,d/2,.7);const extent=Math.max(w,d);this.camera.position.set(top?w/2:w/2+extent*.60,top?d/2-.001:d/2-extent*1.45,top?extent*1.65:extent*1.02);this.camera.up.set(0,0,1);this.controls.update();}
 resize(){const w=this.container.clientWidth,h=this.container.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
 setTime(time,provided=null){
  this.time=time;if(!this.project)return;
  const p=provided||poseAt(this.project,time),root=v(p.root),angle=p.angle,forward=new THREE.Vector3(Math.cos(angle),Math.sin(angle),0),side=new THREE.Vector3(-Math.sin(angle),Math.cos(angle),0);
  this.actor.visible=this.project.present;this.personLabel.position.copy(root).add(new THREE.Vector3(0,0,1.96));this.personLabel.el.textContent=`${names[p.kind]} · ${time.toFixed(1)} s`;this.personLabel.el.style.visibility=this.project.present?'visible':'hidden';
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
  this.controls.update();this.renderer.render(this.scene,this.camera);
  const w=this.container.clientWidth,h=this.container.clientHeight;
  for(const label of this.labels){
   if(label.layer&&!this.layerState[label.layer]){label.el.style.display='none';continue;}
   const p=label.position.clone().project(this.camera);label.el.style.display=p.z>1||p.z< -1?'none':'';
   label.el.style.left=((p.x*.5+.5)*w+(label.offset?.[0]||0))+'px';label.el.style.top=((-p.y*.5+.5)*h+(label.offset?.[1]||0))+'px';
  }
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
   if(!this.drag)return;const d=this.drag;
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
