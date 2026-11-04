export const names={still:'停留',walk:'走动',wave:'挥手',empty:'无人'};
export const total=p=>p.actions.reduce((s,a)=>s+Number(a.duration),0);
export const activeLinks=p=>p.nodes.filter(n=>n.role==='tx').flatMap(a=>p.nodes.filter(n=>n.role==='rx').map(b=>a.id+b.id));
export const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export function armJoints(shoulder,target,pole){
 const upper=.30,fore=.30,delta=target.map((v,i)=>v-shoulder[i]),length=Math.hypot(...delta),direction=delta.map(v=>v/length);
 const reach=clamp(length,1e-6,upper+fore-1e-6),hand=shoulder.map((v,i)=>v+direction[i]*reach);
 const along=(upper*upper-fore*fore+reach*reach)/(2*reach),dot=pole.reduce((s,v,i)=>s+v*direction[i],0);
 const bend=pole.map((v,i)=>v-dot*direction[i]),norm=Math.hypot(...bend),height=Math.sqrt(Math.max(0,upper*upper-along*along));
 const elbow=shoulder.map((v,i)=>v+direction[i]*along+bend[i]/norm*height);
 return {elbow,hand};
}
export function actionSegments(p){
 const points=Object.fromEntries(p.points.map(x=>[x.id,x.position]));
 let start=0,point=[...p.points[0].position],angle=0;
 return p.actions.map(a=>{
  const angle0=angle;
  const target=a.kind==='walk'?[...points[a.target]]:[...point];
  if(a.kind==='walk'&&Math.hypot(target[0]-point[0],target[1]-point[1])>1e-9)angle=Math.atan2(target[1]-point[1],target[0]-point[0]);
  if(a.kind==='wave')angle=a.heading*Math.PI/180;
  const s={...a,start,end:start+a.duration,p0:[...point],p1:target,angle,angle0};
  point=target;start=s.end;return s;
 });
}
export function poseAt(p,t){
 const ss=actionSegments(p),s=ss.find(x=>t<x.end)||ss.at(-1);
 const dt=clamp(t-s.start,0,s.duration),r=Math.min(.3,s.duration/3),edge=Math.min(dt,s.duration-dt);
 const env=edge<r?.5*(1-Math.cos(Math.PI*edge/r)):1;
 const turn=Math.atan2(Math.sin(s.angle-s.angle0),Math.cos(s.angle-s.angle0));
 const root=[...s.p0],angle=s.angle0+turn*.5*(1-Math.cos(Math.PI*Math.min(dt/r,1))),forward=[Math.cos(angle),Math.sin(angle),0],side=[-Math.sin(angle),Math.cos(angle),0];
 let left=side.map((x,i)=>.25*x+(i===2?.85:0)),right=side.map((x,i)=>-.25*x+(i===2?.85:0));
 if(s.kind==='walk'){
  const head=.5*(dt-r/Math.PI*Math.sin(Math.PI*dt/r)),remain=s.duration-dt;
  const tail=s.duration-r-.5*(remain-r/Math.PI*Math.sin(Math.PI*remain/r));
  const distance=(dt<r?head:dt>s.duration-r?tail:dt-r/2)/(s.duration-r);
  for(let i=0;i<3;i++)root[i]+=distance*(s.p1[i]-s.p0[i]);
  const swing=.12*env*Math.sin(2*Math.PI*1.5*dt);
  left=left.map((v,i)=>v+swing*forward[i]);right=right.map((v,i)=>v-swing*forward[i]);
 }else if(s.kind==='wave'){
  const displacement=s.amplitude*env*env*Math.sin(2*Math.PI*s.frequency*dt);
  right=right.map((v,i)=>v+displacement*side[i]+.40*Math.sqrt(env)*forward[i]+(i===2?.75*env:0));
 }
 const shoulders=[],elbows=[],hands=[];
 for(const [i,target] of [left,right].entries()){
  const sign=i===0?1:-1,shoulder=root.map((v,k)=>v+.205*sign*side[k]+(k===2?1.36:0));
  const pole=forward.map((v,k)=>.7*v+sign*side[k]+(k===2?-.3:0));
  const arm=armJoints(shoulder,target.map((v,k)=>v+root[k]),pole);
  shoulders.push(shoulder);elbows.push(arm.elbow);hands.push(arm.hand);
 }
 return {root,points:[root.map((v,i)=>v+(i===2?1.1:0)),...hands],shoulders,elbows,angle,kind:p.present?s.kind:'empty',elapsed:dt};
}
export function pointTimes(p){
 const result={[p.points[0].id]:p.actions[0]?.kind==='walk'?0:(actionSegments(p).find(s=>s.kind==='walk')?.start||0)};
 for(const s of actionSegments(p))if(s.kind==='walk')result[s.target]=s.end;
 return result;
}
