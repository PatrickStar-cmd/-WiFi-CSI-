import {RoomView} from './scene.js';
import {poseAt,actionSegments,activeLinks,total,names,clamp} from './motion.mjs';
import {drawStacked,drawComparison,indexAt} from './plots.js';
import {materials,furnitureDefaults,wallDefaults,footprint,objectFits,translated,withWallDimensions,routeConflicts} from './obstacles.mjs';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clone=x=>JSON.parse(JSON.stringify(x)),fmt=(x,d=3)=>x===null||x===undefined?'—':Number(x).toFixed(d);
const clock=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(1).padStart(4,'0')}`;
const colors={walk:'var(--walk)',still:'var(--still)',wave:'var(--wave)'};
const app={project:null,presets:[],data:null,bridge:null,links:[],link:'AC',threshold:.5,time:12.4,tab:'nodes',selectedNode:'A',nodeExpanded:false,selectedAction:3,selectedPoint:'P1',object:null,placement:null,sceneSettingsOpen:false,sceneChanged:false,csiMode:'spectrum',playing:false,speed:1,zoom:1,busy:false,dirty:true,comparison:false,presetIndex:0};
let view,lastFrame=performance.now(),lastDraw=0,toastTimer,thresholdTimer;
window.studioErrors=[];window.addEventListener('error',e=>{window.studioErrors.push(e.message);if(app.bridge)error(e.message);});
const icons=await fetch('./icons.json').then(r=>r.json());
$$('[data-icon]').forEach(el=>el.innerHTML=icons[el.dataset.icon]);await document.fonts.ready;
function toast(message){$('#toast').textContent=message;$('#toast').classList.remove('hidden');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.add('hidden'),3300);}
function playIcon(){$('#play').innerHTML=icons[app.playing?'pause':'play'];}
function error(message){app.busy=false;document.body.classList.remove('computing');$('#busy').classList.add('hidden');$('#compute').disabled=false;$('#error-text').textContent=message;$('#error-dialog').showModal();}
function field(id,label,value,unit='',attrs=''){return `<div class="field"><label for="${id}">${label}</label><input id="${id}" type="number" value="${value}" ${attrs}><span class="unit">${unit}</span></div>`;}
function selectField(id,label,values,value){return `<div class="field"><label for="${id}">${label}</label><select id="${id}">${values.map(([v,t])=>`<option value="${esc(v)}" ${String(v)===String(value)?'selected':''}>${esc(t)}</option>`).join('')}</select><span></span></div>`;}
function nodeProperties(){
 const n=app.project.nodes.find(n=>n.id===app.selectedNode)||app.project.nodes[0];app.selectedNode=n.id;
 return app.project.nodes.map(node=>`<div class="node-row ${node.id===n.id?'selected':''}" data-node="${node.id}" role="button" tabindex="0"><i class="node-icon ${node.role==='tx'?'tx':''}"></i><div><b>${node.id}　${node.role==='tx'?'发射':'接收'}</b><small>(${node.position.map(v=>Number(v.toFixed(2))).join(', ')})</small></div></div>`).join('')+
 `<section class="property-block ${app.nodeExpanded?'':'hidden'}"><div class="property-title">节点 ${n.id}<button id="delete-node" title="删除节点" ${app.project.nodes.length<=2?'disabled':''}>×</button></div><div class="segmented role-switch"><button data-role="tx" class="${n.role==='tx'?'active':''}">发射</button><button data-role="rx" class="${n.role==='rx'?'active':''}">接收</button></div>`+
 n.position.map((v,i)=>field('node-'+i,'XYZ'[i],v,'m',`min="0" max="${app.project.room[i]}" step="0.1"`)).join('')+selectField('antenna','天线数',[[1,1],[2,2],[4,4]],n.antennas)+field('node-heading','朝向',n.heading,'°','min="0" max="359" step="5"')+'</section>';
}
function sceneProperties(){
 const p=app.project,r=p.radio;
 return objectProperties()+`<details class="scene-settings property-block" ${app.sceneSettingsOpen?'open':''}><summary>房间与信号</summary><div class="segmented presence"><button data-presence="false" class="${!p.present?'active':''}">无人</button><button data-presence="true" class="${p.present?'active':''}">有人</button></div><section class="property-block"><div class="property-title">房间</div>`+
 p.room.map((v,i)=>field('room-'+i,['长','宽','高'][i],v,'m','min="2" max="20" step="0.5"')).join('')+field('wall-reflection','墙面反射',p.wallReflection,'','min="0" max="1" step="0.05"')+
 '</section><section class="property-block"><div class="property-title">信号</div>'+field('frequency','频率',r.frequencyGHz,'GHz','min="1" max="7" step="0.001"')+
 selectField('bandwidth','带宽',[[20,'20 MHz'],[40,'40 MHz']],r.bandwidthMHz)+selectField('sample-rate','采样率',[[50,'50 Hz'],[100,'100 Hz']],r.rate)+
 field('snr','SNR',r.snr,'dB','min="0" max="60" step="1"')+field('body-db','人体回波',r.bodyDb,'dB','min="-40" max="0" step="1"')+field('loss','丢包率',r.loss*100,'%','min="0" max="50" step="1"')+field('seed','随机种子',p.seed,'','min="0" max="4294967295" step="1"')+
 '</section></details>';
}
const objectNames={table:'桌子',chair:'椅子',box:'柜子',wall:'隔墙'};
const objectKind=object=>object.start?'wall':object.kind;
const objects=p=>[...p.furniture,...(p.walls??[])];
const rounded=(value,decimals=3)=>Number(value.toFixed(decimals));
const selectedObject=()=>objects(app.project).find(object=>object.id===app.object);
function objectProperties(){
 const ordered=objects(app.project),object=selectedObject(),conflicts=routeConflicts(app.project);
 const list=ordered.map(o=>`<button type="button" class="object-row ${o.id===app.object?'selected':''}" data-object-id="${esc(o.id)}" aria-pressed="${o.id===app.object}"><span class="object-swatch ${objectKind(o)}"></span><span>${esc(o.name)}</span><small>${o.enabled===false?'已停用':objectNames[objectKind(o)]}</small></button>`).join('');
 const tools=Object.entries(objectNames).map(([kind,name])=>`<button type="button" data-add-object="${kind}" class="${app.placement?.kind===kind?'active':''}">＋${name}</button>`).join('');
 const conflict=conflicts.find(item=>item.segment.kind==='walk')||conflicts[0];
 const warning=conflict?`<p id="route-warning" role="status">${conflict.segment.kind==='walk'?'行走路线穿过':'停留位置落在'}「${esc(conflict.object.name)}」，请调整路径。</p>`:'';
 const header=`<section class="scene-objects"><div class="property-title">场景对象 <small>${ordered.length} 个</small></div><div class="object-add">${tools}</div><div class="object-list">${list||'<div class="object-empty">场景中暂无对象</div>'}</div>${warning}</section>`;
 if(!object)return header;
 const {size,angle}=footprint(object),room=app.project.room;
 const num=(id,label,value,unit,min,max)=>field(id,label,rounded(value),unit,`required min="${min}" max="${max}" step="any"`);
 const custom=object.material??'木材',options=[...new Set([...Object.keys(materials),custom])].map(name=>[name,name]);
 const dims=object.start?
  num('wall-height','高度',object.height,'m',.001,room[2])+num('wall-thickness','厚度',object.thickness,'m',.02,1)+num('object-loss','穿透损耗',object.lossDb,'dB',0,60):
  object.size.map((value,i)=>num('object-size-'+i,['长度','宽度','高度'][i],value,'m',.001,Math.max(...room))).join('');
 const geometry=object.start?
  [...object.start,...object.end].map((value,i)=>num(`wall-${i<2?'start':'end'}-${i%2}`,['起点 X','起点 Y','终点 X','终点 Y'][i],value,'m',0,room[i%2])).join('')+
  num('wall-length','长度',size[0],'m',.1,Math.hypot(...room.slice(0,2)))+num('wall-heading','角度',(angle*180/Math.PI+360)%360,'°',0,360):
  object.position.map((value,i)=>num('object-pos-'+i,'XYZ'[i],value,'m',0,room[i])).join('')+
  num('object-heading','角度',((object.heading??0)%360+360)%360,'°',0,360);
 const loss=object.kind==='box'?num('object-loss','穿透损耗',object.lossDb,'dB',0,60):'';
 return header+`<section class="property-block object-detail"><div class="property-title">对象属性 <div class="object-edit-tools"><button id="object-copy" title="复制对象">复制</button><button id="object-delete" title="删除对象">删除</button></div></div><div class="object-core">${dims}</div><label class="object-enabled"><input id="object-enabled" type="checkbox" ${object.enabled!==false?'checked':''}>参与仿真</label><div class="field object-name"><label for="object-name">名称</label><input id="object-name" maxlength="40" value="${esc(object.name)}" required></div>`+
  geometry+selectField('object-material','材质',options,custom)+loss+num('object-reflection','反射系数',object.reflection,'',0,1)+'</section>';
}

function showObject(id){
 const object=objects(app.project).find(object=>object.id===id);
 if(!object)return;
 app.object=id;view.selectObject(id);
 switchTab('scene');
}
function newObjectId(kind){let index=1;while(objects(app.project).some(object=>object.id===`${kind}-${index}`))index++;return `${kind}-${index}`;}
function sceneChanged(){app.sceneChanged=true;changed();}
function setPlacement(placement){
 app.placement=placement;view.setPlacement(placement);
 const banner=$('#placement-hint');banner.classList.toggle('hidden',!placement);
 if(placement)$('#placement-label').textContent=placement.kind==='wall'?(placement.anchor?'点击地面设置隔墙终点':'点击地面设置隔墙起点'):`点击地面放置${objectNames[placement.kind]}`;
}
function beginPlacement(kind){
 if(app.placement?.kind===kind){setPlacement(null);renderProperties();return;}
 view.addPoint=false;$('#add-point').textContent='＋ 路径点';
 setCamera(true);
 const label=$(`[data-layer="${kind==='wall'?'walls':'furniture'}"]`);
 label.checked=true;view.setLayer(label.dataset.layer,true);
 setPlacement({kind});renderProperties();
}
function placeObject(position){
 if(app.busy||!app.placement)return;
 const p=app.project,{kind,anchor}=app.placement,point=position.slice(0,2).map(value=>rounded(value,2));
 if(point.some((value,i)=>value<0||value>p.room[i])){toast('请在房间内摆放');return;}
 if(kind==='wall'&&!anchor){setPlacement({kind,anchor:point});return;}
 const id=newObjectId(kind);
 const name=`${objectNames[kind]} ${id.split('-').at(-1)}`;
 let object;
 if(kind==='wall')object={id,name,start:anchor,end:point,...wallDefaults(p.room),enabled:true};
 else{
  object={id,kind,name,...clone(furnitureDefaults[kind]),position:[...point,0],heading:0,material:'木材',enabled:true};
 }
 if(!objectFits(object,p.room)){toast(kind==='wall'?'隔墙应长于 0.1 m，且完整位于房间内':'请让家具的完整尺寸位于房间内');return;}
 (kind==='wall'?p.walls:p.furniture).push(object);app.object=id;
 setPlacement(null);sceneChanged();
}
function updateObject(updated){
 if(!objectFits(updated,app.project.room)){toast('请检查对象尺寸与位置，完整外形应位于房间内');renderProperties();return;}
 const collection=updated.start?app.project.walls:app.project.furniture;
 collection[collection.findIndex(object=>object.id===updated.id)]=updated;
 sceneChanged();
}
function editObjectProperty(el){
 const original=selectedObject();if(!original)return;
 let object=clone(original);const id=el.id,value=Number(el.value),axis=Number(id.at(-1));
 if(id==='object-name'){
  if(!el.value.trim()){toast('请输入对象名称');renderProperties();return;}
  object.name=el.value.trim();
 }else if(id==='object-enabled')object.enabled=el.checked;
 else if(id.startsWith('object-pos-'))object.position[axis]=value;
 else if(id.startsWith('object-size-'))object.size[axis]=value;
 else if(id==='object-heading')object.heading=value%360;
 else if(id==='object-reflection')object.reflection=value;
 else if(id==='object-loss')object.lossDb=value;
 else if(id==='object-material'){
  object.material=el.value;const preset=materials[object.material];
  if(preset){object.reflection=preset.reflection;if('lossDb' in object)object.lossDb=preset.lossDb;}
 }else if(id.startsWith('wall-start-'))object.start[axis]=value;
 else if(id.startsWith('wall-end-'))object.end[axis]=value;
 else if(id==='wall-height')object.height=value;
 else if(id==='wall-thickness')object.thickness=value;
 else if(id==='wall-length')object=withWallDimensions(object,value,footprint(object).angle);
 else if(id==='wall-heading')object=withWallDimensions(object,footprint(object).size[0],value*Math.PI/180);
 updateObject(object);
}
function copyObject(){
 const original=selectedObject(),kind=objectKind(original),id=newObjectId(kind),center=footprint(original).center;
 const offsets=[[.24,.24],[-.24,.24],[.24,-.24],[-.24,-.24],[0,.24],[.24,0],[0,-.24],[-.24,0]];
 const copy=offsets.map(([dx,dy])=>translated(original,[center[0]+dx,center[1]+dy,center[2]])).find(object=>objectFits(object,app.project.room))||clone(original);
 copy.id=id;copy.name=original.name.slice(0,35)+' 副本';
 (kind==='wall'?app.project.walls:app.project.furniture).push(copy);app.object=id;sceneChanged();
}
function deleteObject(){
 const original=selectedObject(),collection=original.start?app.project.walls:app.project.furniture;
 collection.splice(collection.findIndex(object=>object.id===app.object),1);app.object=null;sceneChanged();
}
function actionProperties(){
 const p=app.project;app.selectedAction=clamp(app.selectedAction,0,p.actions.length-1);const a=p.actions[app.selectedAction],s=actionSegments(p)[app.selectedAction];
 let out='<div class="property-title">路径点<button id="edit-path" title="点击地面添加路径点">＋</button></div>'+p.points.map((point,i)=>`<div class="point-row ${point.id===app.selectedPoint?'selected':''}" data-point="${point.id}"><b>${esc(point.id)}</b><input data-point-x="${point.id}" type="number" step="0.1" value="${point.position[0]}" aria-label="${point.id} X"><input data-point-y="${point.id}" type="number" step="0.1" value="${point.position[1]}" aria-label="${point.id} Y"><button data-delete-point="${point.id}" ${i===0?'disabled':''}>×</button></div>`).join('');
 out+='<section class="property-block"><div class="property-title">动作</div>'+actionSegments(p).map((s,i)=>`<div class="action-row ${i===app.selectedAction?'selected':''}" data-action="${i}"><span style="background:${colors[s.kind]}">${names[s.kind]}</span><small>${s.start.toFixed(1)}–${s.end.toFixed(1)} s</small></div>`).join('');
 out+=selectField('action-kind','动作',Object.entries(names).filter(([k])=>k!=='empty'),a.kind)+field('action-duration','时长',a.duration,'s','min="0.2" max="90" step="0.1"');
 if(a.kind==='walk'){
  const distance=Math.hypot(...s.p1.map((v,i)=>v-s.p0[i])),speed=distance/(a.duration-Math.min(.3,a.duration/3));
  out+=selectField('action-target','目标',p.points.map(x=>[x.id,x.id]),a.target)+field('action-speed','速度',Number(speed.toFixed(3)),'m/s','min="0.05" max="3" step="0.05"');
 }else if(a.kind==='wave')out+=field('wave-amplitude','摆幅',a.amplitude,'m','min="0" max="0.6" step="0.05"')+field('wave-frequency','频率',a.frequency,'Hz','min="0.1" max="5" step="0.1"')+field('wave-heading','朝向',a.heading,'°','min="0" max="359" step="5"');
 out+='<div class="action-tools"><button id="action-up">↑ 上移</button><button id="action-down">↓ 下移</button><button id="action-delete">删除</button></div></section>';return out;
}

function renderProperties(){
 const panel=$('#properties'),scroll=panel.scrollTop;
 panel.innerHTML=app.tab==='nodes'?nodeProperties():app.tab==='scene'?sceneProperties():actionProperties();panel.scrollTop=scroll;
 const settings=panel.querySelector('.scene-settings');if(settings)settings.ontoggle=()=>app.sceneSettingsOpen=settings.open;
 $$('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===app.tab));$('#inspector-title').textContent={nodes:'节点',scene:'场景',actions:'动作'}[app.tab];
 $('#add-node').classList.toggle('hidden',app.tab!=='nodes');$('#add-node').disabled=app.project.nodes.length>=4;
 $('#add-point').classList.toggle('hidden',app.tab!=='actions'||$('#inspector-floating').classList.contains('hidden'));
}
function setEditing(enabled){
 if(document.body.classList.contains('scene-editing')===enabled)return;
 document.body.classList.toggle('scene-editing',enabled);view.fitCamera();view.updateObjectPlans();
}
function switchTab(tab,expanded=false){
 if(app.tab!==tab)$('#properties').scrollTop=0;
 app.tab=tab;if(expanded)app.nodeExpanded=true;$('#inspector-floating').classList.remove('hidden');
 if(tab!=='scene'&&app.placement)setPlacement(null);
 if(tab!=='actions'){view.addPoint=false;$('#add-point').textContent='＋ 路径点';}
 setEditing(tab==='scene');renderProperties();
}
function closeInspector(){
 $('#inspector-floating').classList.add('hidden');$('#add-point').classList.add('hidden');
 view.addPoint=false;if(app.placement)setPlacement(null);view.cancelDrag();setEditing(false);
}
function setCamera(top){
 if(!top&&app.placement){setPlacement(null);renderProperties();}
 view.resetCamera(top);$('#view-3d').classList.toggle('active',!top);$('#view-top').classList.toggle('active',top);
}
function invalidate(){clearTimeout(thresholdTimer);app.data=null;app.dirty=true;app.playing=false;playIcon();app.bridge.invalidate();renderComparison();}
function fusionName(){return app.links.length===4?'四链路融合':'多链路融合';}
function syncProject(){
 app.links=activeLinks(app.project);if(!app.links.includes(app.link))app.link=app.links[0]||'';app.project.selectedLink=app.link;
 $('#link').innerHTML=app.links.map(x=>`<option ${x===app.link?'selected':''}>${x}</option>`).join('');$('#link-count').textContent=`${app.links.length} 条链路`;
 $$('.fusion-label').forEach(e=>e.textContent=fusionName());$('#fusion-track-label').textContent=fusionName()+'检测';$('#single-label').textContent=`单链路 ${app.link||'—'}`;$('#single-track-label').textContent=`单链路 ${app.link||'—'} 检测`;
 $('.link-grid').innerHTML=app.links.map(l=>`<button data-result-link="${l}" class="${l===app.link?'selected':''}" title="${l} · 运行后生成"><b>${l}</b><span class="link-dot unknown"></span></button>`).join('');
 app.time=clamp(app.time,0,total(app.project));view.selectedLink=app.link;view.setProject(app.project);view.selectObject(app.object);renderProperties();renderComparison();draw();
}
function changed(){invalidate();syncProject();}
function setProject(project){if(app.busy)return;view.cancelDrag();setPlacement(null);app.object=null;app.sceneChanged=false;app.project=clone(project);app.project.walls??=[];app.threshold=app.project.threshold;app.link=app.project.selectedLink;app.selectedNode=app.project.nodes[0].id;app.nodeExpanded=false;app.selectedAction=Math.min(3,app.project.actions.length-1);app.selectedPoint=app.project.points[0].id;changed();}
function reset(){setProject(app.presets[app.presetIndex]);setCamera(false);closeInspector();}
function compute(){if(app.busy)return;app.playing=false;app.project.threshold=app.threshold;app.busy=true;playIcon();closeInspector();document.body.classList.add('computing');$('#busy').classList.remove('hidden');$('#compute').disabled=true;$('#progress').value=0;$('#progress-label').textContent='静态校准';app.bridge.compute(JSON.stringify(app.project));}
function setTime(t){app.time=clamp(t,0,total(app.project));draw();}
function setLink(link){if(!app.links.includes(link))return;app.link=link;app.project.selectedLink=link;$('#link').value=link;$('#single-label').textContent=`单链路 ${link}`;$('#single-track-label').textContent=`单链路 ${link} 检测`;$$('[data-result-link]').forEach(e=>e.classList.toggle('selected',e.dataset.resultLink===link));view.setLink(link);renderComparison();draw();}
function setThreshold(value,immediate=false){app.threshold=clamp(value,.02,.98);app.project.threshold=app.threshold;clearTimeout(thresholdTimer);if(app.data)thresholdTimer=setTimeout(()=>app.bridge.changeThreshold(app.threshold),immediate?0:100);draw();}
function select(type,id){if(type==='node'){app.selectedNode=id;switchTab('nodes',true);}else if(type==='point'){app.selectedPoint=id;switchTab('actions');}else showObject(id);}
function move(type,id,position){
 if(app.busy)return;
 if(type==='furniture'||type==='wall'){updateObject(translated(objects(app.project).find(object=>object.id===id),position));return;}
 const record=(type==='node'?app.project.nodes:app.project.points).find(x=>x.id===id);record.position=position;changed();
}
function togglePathPlacement(){
 if(app.placement)setPlacement(null);
 view.addPoint=!view.addPoint;$('#add-point').textContent=view.addPoint?'✓ 点击地面':'＋ 路径点';
}
function addPoint(position){if(app.busy)return;position=[clamp(position[0],.2,app.project.room[0]-.2),clamp(position[1],.2,app.project.room[1]-.2),0].map(v=>Math.round(v*100)/100);let number=0;while(app.project.points.some(p=>p.id==='P'+number))number++;app.project.points.push({id:'P'+number,position});app.selectedPoint='P'+number;changed();}
function addAction(kind){if(app.busy)return;const a={kind,duration:kind==='walk'?5:3};if(kind==='walk')a.target=app.project.points.at(-1).id;if(kind==='wave')Object.assign(a,{amplitude:.3,frequency:1.5,heading:0});app.project.actions.push(a);app.selectedAction=app.project.actions.length-1;switchTab('actions');changed();}
function renderComparison(){
 const container=$('.comparison-grid');container.innerHTML='';
 if(!app.data){$('#comparison-improvement').textContent='运行后生成';container.innerHTML='<div class="comparison-placeholder">点击「计算仿真」查看算法对比</div>';return;}
 const before=app.data.series.original.fused.metrics.f1,after=app.data.series.optimized.fused.metrics.f1,diff=before!==null&&after!==null?after-before:null;
 $('#comparison-improvement').textContent=diff===null?'':`融合 F1 ${diff>=0?'+':''}${(diff*100).toFixed(1)} 个百分点`;$('#comparison-improvement').style.color=diff>=0?'var(--green)':'var(--red)';
 for(const [algorithm,label] of [['original','原始算法'],['optimized','优化算法']])for(const [topology,which] of [['single',`单链路 ${app.link}`],['fused',fusionName()]]){
  const row=topology==='fused'?app.data.series[algorithm].fused:app.data.series[algorithm].links[app.link],m=row.metrics,el=document.createElement('section');el.className='card comparison-card';
  el.innerHTML=`<h4>${label} · ${which}</h4><canvas data-algorithm="${algorithm}" data-topology="${topology}"></canvas><div class="comparison-stats"><span>F1 <b>${fmt(m.f1)}</b></span><span>召回 <b>${fmt(m.recall)}</b></span><span>误报 <b>${fmt(m.fpr)}</b></span><span>延迟 <b>${fmt(m.delay,2)} s</b></span></div>`;container.appendChild(el);
  el.querySelector('canvas').onpointerdown=e=>{const r=e.target.plotRect;if(r)setTime((e.clientX-e.target.getBoundingClientRect().left-r.x)/r.w*total(app.project));};
 }
}
function toggleComparison(force){app.comparison=force??!app.comparison;$('#compare-drawer').classList.toggle('hidden',!app.comparison);document.body.classList.toggle('comparing',app.comparison);renderComparison();requestAnimationFrame(()=>{view.resize();draw();});}
function draw(){
 if(!app.project)return;const duration=total(app.project),planned=poseAt(app.project,app.time);let provided=null;
 if(app.data){const i=indexAt(app.data.time,app.time),p=app.data.pose;provided={root:p.root[i],points:p.points[i],shoulders:p.shoulders[i],elbows:p.elbows[i],angle:p.angle[i],kind:p.kind[i],elapsed:planned.elapsed};}
 view.setTime(app.time,provided);$('#truth-state').textContent=names[planned.kind];$('#time-display').innerHTML=`${clock(app.time)} <span>/ ${clock(duration)}</span>`;
 const r=app.project.radio;$('#status-radio').innerHTML=`时长 ${duration.toFixed(1)} s <i>｜</i> 采样 ${r.rate} Hz <i>｜</i> ${r.frequencyGHz.toFixed(3)} GHz / ${r.bandwidthMHz} MHz`;
 let predicted='—',agreement='';
 if(app.data){const index=indexAt(app.data.decisionTime,app.time),state=app.data.series.optimized.fused.state[index],truth=planned.kind==='walk'||planned.kind==='wave';predicted=state<0?'数据不足':state===1?'运动':'静止';agreement=state<0?'':Boolean(state)===truth?'✓':truth?'× 漏检':'× 误报';
  $('#agreement').innerHTML=agreement==='✓'?icons.check:agreement;$('#agreement').className=Boolean(state)===truth?'good':'bad';
  const m=app.data.series.optimized.fused.metrics;for(const key of ['f1','precision','recall','fpr','delay'])$('#metric-'+key).textContent=fmt(m[key],key==='delay'?2:3);$('#metric-delay-unit').classList.toggle('hidden',m.delay===null);
  $('#metric-links').textContent=`${app.data.validCounts[index]}/${app.links.length}`;$('#link-valid').textContent=`${app.data.validCounts[index]} / ${app.links.length}`;
  $$('[data-result-link]').forEach(e=>{const st=app.data.series.optimized.links[e.dataset.resultLink].state[index];e.querySelector('.link-dot').className='link-dot '+(st===1?'moving':st<0?'unknown':'');e.title=`${e.dataset.resultLink} · ${st<0?'数据不足':st===1?'运动':'静止'}`;});
  $('#status').textContent=`仿真完成 · 耗时 ${app.data.seconds.toFixed(1)} s`;$('#more-label').textContent='更多指标';
 }else{for(const key of ['f1','precision','recall','fpr','delay','links'])$('#metric-'+key).textContent='—';$('#metric-delay-unit').classList.add('hidden');$('#agreement').textContent='';$('#link-valid').textContent=`— / ${app.links.length}`;$('#more-label').textContent='运行后生成';if(!app.busy)$('#status').textContent=app.sceneChanged?'场景已修改 · 待计算':'就绪';}
 const obstructions=app.data?.obstructions?.[app.link]??[],label=$('#link-obstructions');
 label.classList.toggle('hidden',!obstructions.length);
 label.closest('.link-card').classList.toggle('with-obstructions',obstructions.length>0);
 if(obstructions.length){
  const hits=obstructions.every(item=>item.kind==='wall')?`穿墙 ${obstructions.length} 次`:`穿过 ${obstructions.length} 个障碍`;
  label.textContent=`直达路径 · ${hits} · ${rounded(obstructions.reduce((sum,item)=>sum+item.lossDb,0),1)} dB`;
  label.title=obstructions.map(item=>item.name).join('、');
 }
 $('#prediction-state').textContent=predicted;$('.state').classList.toggle('long-state',predicted.length>2||agreement.length>1);view.setPrediction(app.data?{label:predicted,agreement}:null);
 const modes={amplitude:'幅度',phase:'相位',heatmap:'热力图',spectrum:'时频谱'};$('#csi-title').textContent=`${app.link||'—'}的CSI${modes[app.csiMode]}`;
 $('#stacked').style.width=$('#plot-scroll').clientWidth*app.zoom+'px';drawStacked($('#stacked'),app);
 if(app.comparison&&app.data)$$('.comparison-card canvas').forEach(c=>drawComparison(c,app,c.dataset.topology==='fused'?app.data.series[c.dataset.algorithm].fused:app.data.series[c.dataset.algorithm].links[app.link]));
}
function bindProperties(){
 $('#properties').addEventListener('click',e=>{
  if(app.busy)return;
  const add=e.target.closest('[data-add-object]'),object=e.target.closest('[data-object-id]');
  if(add){beginPlacement(add.dataset.addObject);return;}
  if(object){if(app.placement)setPlacement(null);showObject(object.dataset.objectId);return;}
  const node=e.target.closest('[data-node]'),action=e.target.closest('[data-action]'),role=e.target.closest('[data-role]'),presence=e.target.closest('[data-presence]');
  if(node){app.nodeExpanded=node.dataset.node===app.selectedNode?!app.nodeExpanded:true;app.selectedNode=node.dataset.node;renderProperties();return;}
  if(action){app.selectedAction=Number(action.dataset.action);renderProperties();return;}
  if(role){app.project.nodes.find(n=>n.id===app.selectedNode).role=role.dataset.role;changed();return;}
  if(presence){app.project.present=presence.dataset.presence==='true';changed();return;}
  const id=e.target.closest('button')?.id;
  if(id==='object-copy'){copyObject();
  }else if(id==='object-delete'){deleteObject();
  }else if(id==='delete-node'){app.project.nodes=app.project.nodes.filter(n=>n.id!==app.selectedNode);app.selectedNode=app.project.nodes[0].id;changed();
  }else if(id==='action-delete'){if(app.project.actions.length===1){toast('请保留一个动作');return;}app.project.actions.splice(app.selectedAction,1);changed();
  }else if(id==='action-up'||id==='action-down'){
   const i=app.selectedAction,j=i+(id==='action-up'?-1:1);if(j<0||j>=app.project.actions.length)return;[app.project.actions[i],app.project.actions[j]]=[app.project.actions[j],app.project.actions[i]];app.selectedAction=j;changed();
  }else if(id==='edit-path'){togglePathPlacement();
  }else if(e.target.dataset.deletePoint){
   const point=e.target.dataset.deletePoint;if(app.project.actions.some(a=>a.target===point)){toast('请先调整引用此路径点的走动动作');return;}app.project.points=app.project.points.filter(p=>p.id!==point);changed();
  }
 });
 $('#properties').addEventListener('change',e=>{
  if(app.busy)return;const el=e.target,id=el.id,value=Number(el.value),p=app.project,r=p.radio,n=p.nodes.find(n=>n.id===app.selectedNode),a=p.actions[app.selectedAction];
  if(el.type==='number'&&(!Number.isFinite(value)||!el.checkValidity())){el.reportValidity();renderProperties();return;}
  if(id.startsWith('object-')||id.startsWith('wall-start-')||id.startsWith('wall-end-')||['wall-height','wall-thickness','wall-length','wall-heading'].includes(id)){editObjectProperty(el);return;}
  if(id.startsWith('node-')&&id!=='node-heading')n.position[Number(id.at(-1))]=value;
  else if(id==='node-heading')n.heading=value;
  else if(id==='antenna')n.antennas=value;
  else if(id.startsWith('room-')){
   const room=[...p.room];room[Number(id.at(-1))]=value;
   if(objects(p).some(object=>!objectFits(object,room))){toast('调整房间前请将场景对象移入新范围');renderProperties();return;}
   p.room=room;
  }
  else if(id==='wall-reflection')p.wallReflection=value;
  else if(id==='frequency')r.frequencyGHz=value;
  else if(id==='bandwidth')r.bandwidthMHz=value;
  else if(id==='sample-rate')r.rate=value;
  else if(id==='snr')r.snr=value;
  else if(id==='body-db')r.bodyDb=value;
  else if(id==='loss')r.loss=value/100;
  else if(id==='seed')p.seed=Math.round(value);
  else if(id==='action-kind'){a.kind=el.value;if(a.kind==='walk')a.target=p.points.at(-1).id;if(a.kind==='wave')Object.assign(a,{amplitude:.3,frequency:1.5,heading:0});}
  else if(id==='action-duration')a.duration=Math.max(.2,value);
  else if(id==='action-target')a.target=el.value;
  else if(id==='action-speed'){
   if(value<=0){error('走动速度应大于零');return;}
   const s=actionSegments(p)[app.selectedAction],d=Math.hypot(...s.p1.map((v,i)=>v-s.p0[i])),steady=d/value;a.duration=Math.max(.2,steady>=.6?steady+.3:1.5*steady);
  }else if(id==='wave-amplitude')a.amplitude=value;
  else if(id==='wave-frequency')a.frequency=value;
  else if(id==='wave-heading')a.heading=value;
  else if(el.dataset.pointX||el.dataset.pointY){const point=p.points.find(x=>x.id===(el.dataset.pointX||el.dataset.pointY));point.position[el.dataset.pointX?0:1]=value;}
  changed();
 });
}
function bind(){
 $$('[data-tab]').forEach(b=>b.onclick=()=>{if(app.tab===b.dataset.tab&&!$('#inspector-floating').classList.contains('hidden'))closeInspector();else switchTab(b.dataset.tab);});bindProperties();$('#inspector-close').onclick=closeInspector;
 $('#add-node').onclick=()=>{if(app.busy)return;const id=[...'ABCD'].find(id=>!app.project.nodes.some(n=>n.id===id));if(!id)return;app.project.nodes.push({id,role:app.project.nodes.some(n=>n.role==='tx')?'rx':'tx',position:[1,1,1.2],antennas:1,heading:0});app.selectedNode=id;app.nodeExpanded=true;changed();};
  $('#preset').onchange=e=>{app.presetIndex=Number(e.target.value);setProject(app.presets[app.presetIndex]);setCamera(false);closeInspector();toggleComparison(app.presetIndex===2);};
 $('#compute').onclick=compute;$('#open').onclick=()=>app.bridge.openScene();$('#save').onclick=()=>app.bridge.saveScene(JSON.stringify(app.project));
  $('#view-3d').onclick=()=>setCamera(false);$('#view-top').onclick=()=>setCamera(true);$('#camera-reset').onclick=()=>setCamera(view.top);
 $('#layers-button').onclick=e=>{e.stopPropagation();$('#layers-menu').classList.toggle('hidden');};$$('[data-layer]').forEach(el=>el.onchange=()=>view.setLayer(el.dataset.layer,el.checked));
 $('#export-menu').onclick=e=>{e.stopPropagation();$('#export-options').classList.toggle('hidden');};$$('[data-export]').forEach(el=>el.onclick=()=>{$('#export-options').classList.add('hidden');requestAnimationFrame(()=>app.bridge.exportFile(el.dataset.export,JSON.stringify({threshold:app.threshold,link:app.link})));});
 document.addEventListener('click',e=>{if(!e.target.closest('.menu-wrap'))$('#export-options').classList.add('hidden');if(!e.target.closest('#layers-button,#layers-menu'))$('#layers-menu').classList.add('hidden');});
  $('#add-point').onclick=togglePathPlacement;
  $('#placement-cancel').onclick=()=>{setPlacement(null);renderProperties();};
 $('#link').onchange=e=>setLink(e.target.value);$('.link-grid').onclick=e=>{const button=e.target.closest('[data-result-link]');if(button)setLink(button.dataset.resultLink);};
 $('#more').onclick=()=>{if(app.data)$('#more-metrics').classList.toggle('hidden');};
 $$('[data-csi]').forEach(b=>b.onclick=()=>{app.csiMode=b.dataset.csi;$$('[data-csi]').forEach(x=>x.classList.toggle('active',x===b));requestAnimationFrame(draw);});
 $('#demo').onclick=()=>{const enabled=document.body.classList.toggle('demo');$('#demo span').textContent=enabled?'退出演示':'演示模式';closeInspector();requestAnimationFrame(()=>{view.resize();draw();});};
 $('#play').onclick=()=>{if(app.time>=total(app.project))app.time=0;app.playing=!app.playing;playIcon();};
 $('#step-back').onclick=()=>{app.playing=false;playIcon();setTime(app.time-1/app.project.radio.rate);};$('#step-forward').onclick=()=>{app.playing=false;playIcon();setTime(app.time+1/app.project.radio.rate);};
 $('#speed').onchange=e=>app.speed=Number(e.target.value);$('#zoom').oninput=e=>{app.zoom=Number(e.target.value);draw();};$$('[data-add-action]').forEach(b=>b.onclick=()=>addAction(b.dataset.addAction));
 $('#compare-toggle').onclick=()=>toggleComparison();$('#compare-close').onclick=()=>toggleComparison(false);
 $('#guide').onclick=()=>{if($('#guide-overlay').classList.contains('hidden')){reset();$('#guide-overlay').classList.remove('hidden');$('#guide').classList.add('on');}else{$('#guide-overlay').classList.add('hidden');$('#guide').classList.remove('on');}};$('#guide-close').onclick=()=>{$('#guide-overlay').classList.add('hidden');$('#guide').classList.remove('on');};$('#error-close').onclick=()=>$('#error-dialog').close();
 const timeline=$('#stacked');let drag=null;
 timeline.onpointerdown=e=>{const rect=timeline.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,a=timeline.axis;if(x<a.left)return;app.playing=false;playIcon();
  if(Math.abs(y-a.score.ty(app.threshold))<8&&x<=a.end+20&&y>=a.rows.score[0])drag={type:'threshold'};
  else{const key=a.keys.find(k=>Math.abs(k.x-x)<8&&y<a.ruler);if(key){setTime(key.t);return;}const ss=actionSegments(app.project),inTruth=y>=a.rows.truth[0]&&y<a.rows.truth[0]+a.rows.truth[1],edge=inTruth?ss.findIndex(s=>Math.abs(x-a.tx(s.end))<6):-1;
   if(edge>=0&&!app.busy)drag={type:'resize',index:edge,start:ss[edge].start};else{drag={type:'seek'};const t=(x-a.left)/a.plotW*a.duration;if(inTruth){const i=ss.findIndex(s=>t>=s.start&&t<s.end);if(i>=0){app.selectedAction=i;switchTab('actions');}}setTime(t);}}
  timeline.setPointerCapture(e.pointerId);
 };
 timeline.onpointermove=e=>{const a=timeline.axis,x=e.clientX-timeline.getBoundingClientRect().left,y=e.clientY-timeline.getBoundingClientRect().top;timeline.style.cursor=Math.abs(y-a.score.ty(app.threshold))<8&&x>=a.left?'ns-resize':'crosshair';if(!drag)return;if(drag.type==='threshold')setThreshold(1-(y-a.score.y)/a.score.h);else if(drag.type==='seek')setTime((x-a.left)/a.plotW*a.duration);else drag.end=(x-a.left)/a.plotW*a.duration;};
 const release=e=>{if(drag?.type==='resize'&&drag.end!==undefined){app.project.actions[drag.index].duration=Math.max(.2,Math.round((drag.end-drag.start)*10)/10);changed();}if(drag?.type==='threshold')setThreshold(app.threshold,true);drag=null;if(timeline.hasPointerCapture(e.pointerId))timeline.releasePointerCapture(e.pointerId);};timeline.onpointerup=release;timeline.onpointercancel=release;
  document.addEventListener('keydown',e=>{
   if(e.code==='Escape'&&app.placement){e.preventDefault();setPlacement(null);renderProperties();return;}
   if(e.code==='Escape'&&view.cancelDrag()){e.preventDefault();return;}
   if(e.code==='Escape'&&view.addPoint){e.preventDefault();togglePathPlacement();return;}
   if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName))return;
   if(e.code==='Space'){e.preventDefault();$('#play').click();}else if(e.ctrlKey&&e.key==='s'){e.preventDefault();$('#save').click();}else if(e.ctrlKey&&e.key==='o'){e.preventDefault();$('#open').click();}else if(e.code==='Escape'){closeInspector();toggleComparison(false);}
  });
 new ResizeObserver(draw).observe($('#time-section'));
}
function loop(now){const delta=Math.min((now-lastFrame)/1000,.1);lastFrame=now;if(app.project){if(app.playing){app.time=Math.min(total(app.project),app.time+delta*app.speed);if(app.time>=total(app.project)){app.playing=false;playIcon();}if(now-lastDraw>33){draw();lastDraw=now;}}view.render();}requestAnimationFrame(loop);}
function layout(){const rect=e=>{const r=e.getBoundingClientRect();return{x:r.x,y:r.y,w:r.width,h:r.height};};return{numbers:$$('.metric-number').map(rect),labels:$$('.metric-label').map(rect),tracks:$$('#labels>div').map(rect),canvas:rect($('#stacked')),axis:$('#stacked').axis,sceneLabels:view.labels.filter(l=>l.el.matches('.person,.current-point,.fresnel')&&getComputedStyle(l.el).visibility!=='hidden').map(l=>({text:l.el.textContent,...rect(l.el)})),viewport:rect($('#viewport')),timeSection:rect($('#time-section')),rightHasCanvas:!!$('#results canvas')};}
function boot(bridge,config){
  app.bridge=bridge;app.presets=config.presets;app.project=clone(config.presets[0]);app.project.walls??=[];app.links=activeLinks(app.project);$('#preset').innerHTML=config.presets.map((p,i)=>`<option value="${i}">${esc(p.name)}</option>`).join('');view=new RoomView($('#scene-canvas'),$('#scene-labels'),{select,move,addPoint,placeObject});
 bridge.progress.connect((value,text)=>{$('#progress').value=value;$('#progress-label').textContent=text;$('#status').textContent=`计算中 · ${value}% · ${text}`;});
  bridge.result.connect(raw=>{app.data=JSON.parse(raw);app.dirty=false;app.sceneChanged=false;app.busy=false;document.body.classList.remove('computing');$('#busy').classList.add('hidden');$('#compute').disabled=false;renderComparison();draw();});
  bridge.error.connect(error);bridge.loaded.connect(raw=>{setProject(JSON.parse(raw));setCamera(false);});bridge.exported.connect(path=>toast('已保存 · '+path));bridge.decisionsChanged.connect(raw=>{if(app.data){app.data.series=JSON.parse(raw);renderComparison();draw();}});
 bind();syncProject();requestAnimationFrame(loop);
 window.studio={app,view,setTime,setThreshold,setLink,setProject,compute,switchTab,addAction,toggleComparison,closeInspector,reset,layout,
 snapshot:()=>({ready:true,busy:app.busy,hasData:!!app.data,time:app.time,link:app.link,threshold:app.threshold,truth:$('#truth-state').textContent,prediction:$('#prediction-state').textContent,links:app.links,stateAtCursor:app.data?app.data.series.optimized.fused.state[indexAt(app.data.decisionTime,app.time)]:null,metrics:app.data?.series.optimized.fused.metrics,errors:window.studioErrors,project:app.project,sceneObjects:view.scene.children.length,webgl:!!view.renderer.getContext(),comparison:app.comparison,csiMode:app.csiMode,demo:document.body.classList.contains('demo'),layout:layout()})};window.studioReady=true;
}
new QWebChannel(qt.webChannelTransport,channel=>channel.objects.studio.initialize(raw=>boot(channel.objects.studio,JSON.parse(raw))));
