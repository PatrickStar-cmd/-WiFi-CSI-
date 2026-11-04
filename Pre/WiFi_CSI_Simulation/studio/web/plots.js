import {total,actionSegments,names,clamp} from './motion.mjs';
const C={ink:'#292824',muted:'#7f7c74',line:'#c5c1b7',single:'#91a0b5',fusion:'#394b63',red:'#c73c39',amber:'#b48222',walk:'#dce5cf',still:'#eae3d7',wave:'#ddd2ed',bg:'#faf9f5'};
export function canvasContext(canvas){
 const rect=canvas.getBoundingClientRect(),w=Math.max(1,rect.width),h=Math.max(1,rect.height),d=Math.min(devicePixelRatio,2);
 if(canvas.width!==Math.round(w*d)||canvas.height!==Math.round(h*d)){canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);}
 const ctx=canvas.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);ctx.font='14px "Microsoft YaHei"';ctx.textBaseline='middle';return {ctx,w,h};
}
export const indexAt=(times,t)=>{let lo=0,hi=times.length;while(lo<hi){const m=(lo+hi)>>1;if(times[m]<=t+1e-8)lo=m+1;else hi=m;}return Math.max(0,lo-1);};
function ticks(duration){const step=duration<=12?2:duration<=40?5:duration<=80?10:30;const a=[];for(let t=0;t<=duration;t+=step)a.push(t);if(a.at(-1)<duration-.5)a.push(duration);return a;}
function frame(ctx,w,h,duration,min=0,max=1){
 const r={x:42,y:22,w:Math.max(10,w-57),h:Math.max(10,h-51),min,max};r.tx=t=>r.x+t/duration*r.w;r.ty=v=>r.y+r.h-(v-min)/(max-min)*r.h;
 ctx.strokeStyle=C.line;ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(r.x,r.y);ctx.lineTo(r.x,r.y+r.h);ctx.lineTo(r.x+r.w,r.y+r.h);ctx.stroke();ctx.fillStyle=C.ink;
 for(const value of [min,(min+max)/2,max]){const y=r.ty(value);ctx.textAlign='right';ctx.fillText(Math.abs(value)<1e-7?'0':Math.abs(value)>=10?value.toFixed(0):value.toFixed(1),r.x-7,y);}
 ctx.textAlign='center';for(const t of ticks(duration)){const x=r.tx(t);ctx.beginPath();ctx.moveTo(x,r.y+r.h);ctx.lineTo(x,r.y+r.h+4);ctx.stroke();ctx.fillText(`${Number(t.toFixed(1))} s`,x,r.y+r.h+17);}
 return r;
}
function line(ctx,r,times,values,color,width=1.4){
 ctx.save();ctx.beginPath();ctx.rect(r.x,r.y,r.w,r.h);ctx.clip();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();let pen=false;
 for(let i=0;i<values.length;i++){if(values[i]===null||!Number.isFinite(values[i])){pen=false;continue;}const x=r.tx(times[i]),y=r.ty(values[i]);if(pen)ctx.lineTo(x,y);else ctx.moveTo(x,y);pen=true;}ctx.stroke();ctx.restore();
}
function cursor(ctx,r,time,label=true){const x=r.tx(time);ctx.strokeStyle=C.red;ctx.lineWidth=1.3;ctx.beginPath();ctx.moveTo(x,r.y-4);ctx.lineTo(x,r.y+r.h);ctx.stroke();if(label){const text=`${time.toFixed(1)} s`;ctx.font='14px "Segoe UI"';ctx.textAlign='center';const width=ctx.measureText(text).width+10;ctx.fillStyle=C.red;ctx.beginPath();ctx.roundRect(x-width/2,r.y-20,width,17,4);ctx.fill();ctx.fillStyle='#fff';ctx.fillText(text,x,r.y-11);}}
export function drawScores(canvas,app){
 const {ctx,w,h}=canvasContext(canvas),duration=total(app.project),r=frame(ctx,w,h,duration);
 canvas.plotRect=r;
 if(app.data){const results=app.data.series.optimized;line(ctx,r,app.data.decisionTime,results.links[app.link].score,C.single,1.3);line(ctx,r,app.data.decisionTime,results.fused.score,C.fusion,2);}
 ctx.strokeStyle='#898b84';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(r.x,r.ty(app.threshold));ctx.lineTo(r.x+r.w,r.ty(app.threshold));ctx.stroke();ctx.setLineDash([]);
 ctx.fillStyle=C.bg;ctx.beginPath();ctx.arc(r.x+r.w,r.ty(app.threshold),4,0,2*Math.PI);ctx.fill();ctx.stroke();
 ctx.font='14px "Microsoft YaHei"';ctx.fillStyle='#66665f';ctx.textAlign='right';ctx.fillText(`阈值 ${app.threshold.toFixed(2)}`,r.x+r.w-3,r.ty(app.threshold)-12);
 if(app.data)cursor(ctx,r,app.time);
}
const heatCache=new WeakMap();
function heatImage(matrix){
 if(heatCache.has(matrix))return heatCache.get(matrix);
 const rows=matrix.length,cols=matrix[0].length,can=document.createElement('canvas');can.width=cols;can.height=rows;const ctx=can.getContext('2d'),im=ctx.createImageData(cols,rows);
 const flat=matrix.flat().filter(Number.isFinite).sort((a,b)=>a-b),lo=flat[Math.floor(flat.length*.02)],hi=flat[Math.floor(flat.length*.98)];
 for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){const f=clamp((matrix[row][col]-lo)/(hi-lo||1),0,1),i=((rows-1-row)*cols+col)*4;
  const low=[45,59,83],high=[229,217,151];for(let k=0;k<3;k++)im.data[i+k]=Math.round(low[k]*(1-f)+high[k]*f);im.data[i+3]=255;}
 ctx.putImageData(im,0,0);heatCache.set(matrix,can);return can;
}
export function drawCSI(canvas,app){
 const {ctx,w,h}=canvasContext(canvas),duration=total(app.project);
 if(!app.data){frame(ctx,w,h,duration);return;}
 const trace=app.data.traces[app.link],mode=app.csiMode;
 if(mode==='amplitude'||mode==='phase'){
  const values=trace[mode],finite=values.filter(Number.isFinite),lo=Math.min(...finite),hi=Math.max(...finite),margin=Math.max((hi-lo)*.08,.02);
  const r=frame(ctx,w,h,duration,mode==='amplitude'?Math.max(0,lo-margin):lo-margin,hi+margin);
  line(ctx,r,app.data.time,values,'#555950',1);cursor(ctx,r,app.time);canvas.plotRect=r;
 }else{
  const isHeat=mode==='heatmap',matrix=isHeat?trace.heatmap:trace.spectrum;
  const r=frame(ctx,w,h,duration,isHeat?-28:0,isHeat?28:trace.frequencies.at(-1));
  const stamps=isHeat?trace.heatTime:trace.spectrumTime;
  const step=stamps.length>1?stamps[1]-stamps[0]:duration;
  const start=isHeat?stamps[0]:stamps[0]-step/2,end=stamps.at(-1)+(isHeat?step:step/2);
  ctx.save();ctx.beginPath();ctx.rect(r.x,r.y,r.w,r.h);ctx.clip();ctx.imageSmoothingEnabled=false;
  ctx.drawImage(heatImage(matrix),r.tx(start),r.y,r.tx(end)-r.tx(start),r.h);ctx.restore();cursor(ctx,r,app.time);canvas.plotRect=r;
 }
}
function roundedBar(ctx,x,y,w,h,color){if(w<.1)return;ctx.fillStyle=color;ctx.beginPath();ctx.roundRect(x,y,w,h,Math.min(3,w/2));ctx.fill();}
function hatch(ctx,x,y,w,h){ctx.save();ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();ctx.fillStyle='#fff9f5';ctx.fillRect(x,y,w,h);ctx.strokeStyle='#d46a66';ctx.lineWidth=.7;for(let p=x-h;p<x+w+h;p+=7){ctx.beginPath();ctx.moveTo(p,y+h);ctx.lineTo(p+h,y);ctx.stroke();}ctx.restore();ctx.strokeStyle=C.red;ctx.lineWidth=.6;ctx.strokeRect(x,y,w,h);if(w>37){ctx.font='14px "Microsoft YaHei"';ctx.textAlign='center';ctx.fillStyle=C.red;const tw=ctx.measureText('漏检').width;ctx.fillStyle='#fff9f5';ctx.fillRect(x+w/2-tw/2-2,y+2,tw+4,h-4);ctx.fillStyle=C.red;ctx.fillText('漏检',x+w/2,y+h/2);}}
function intervals(times,states,end){const a=[];for(let i=0;i<times.length;i++){const finish=i+1<times.length?times[i+1]:end;if(a.length&&a.at(-1).state===states[i])a.at(-1).end=finish;else a.push({start:times[i],end:finish,state:states[i]});}return a;}
function errors(project,times,states){
 const ss=actionSegments(project),end=total(project),bounds=[...new Set([0,end,...times,...ss.map(s=>s.start),...ss.map(s=>s.end)])].sort((a,b)=>a-b),result=[];
 for(let i=0;i<bounds.length-1;i++){const a=bounds[i],b=bounds[i+1],mid=(a+b)/2,s=ss.find(x=>mid>=x.start&&mid<x.end),truth=project.present&&s?.kind!=='still',prediction=states[indexAt(times,mid)],kind=prediction<0?'unknown':truth&&prediction===0?'fn':!truth&&prediction===1?'fp':null;
  if(!kind)continue;if(result.length&&result.at(-1).kind===kind&&Math.abs(result.at(-1).end-a)<1e-6)result.at(-1).end=b;else result.push({start:a,end:b,kind});}
 return result;
}
export function drawTimeline(canvas,app){
 const {ctx,w,h}=canvasContext(canvas),duration=total(app.project),left=158,right=22,top=29,rowH=(h-top-7)/3,barH=Math.min(24,rowH-7),tx=t=>left+t/duration*(w-left-right);
 canvas.axis={left,right,width:w,duration,top,rowH,tx};
 ctx.strokeStyle=C.line;ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(left,top-5);ctx.lineTo(w-right,top-5);ctx.stroke();
 ctx.font='14px "Segoe UI"';ctx.fillStyle=C.ink;ctx.textAlign='center';for(const t of ticks(duration)){const x=tx(t);ctx.fillText(`${Number(t.toFixed(1))} s`,x,10);ctx.beginPath();ctx.moveTo(x,top-13);ctx.lineTo(x,top-5);ctx.stroke();}
 for(let t=0;t<=duration;t++){const x=tx(t);ctx.beginPath();ctx.moveTo(x,top-9);ctx.lineTo(x,top-5);ctx.stroke();}
 const labels=['动作（真值）',`单链路 ${app.link}`,app.links.length===4?'四链路融合':'多链路融合'];
 for(let row=0;row<3;row++){const y=top+row*rowH+(rowH-barH)/2;roundedBar(ctx,left,y,w-left-right,barH,'#eeefec');ctx.font='600 14px "Microsoft YaHei"';ctx.fillStyle=C.ink;ctx.textAlign='left';ctx.fillText(labels[row],18,y+barH/2);}
 const segments=app.project.present?actionSegments(app.project):[{start:0,end:duration,kind:'empty'}];
 ctx.font='14px "Microsoft YaHei"';ctx.textAlign='center';
 for(let i=0;i<segments.length;i++){
  const s=segments[i],x=tx(s.start),width=tx(s.end)-x,y=top+(rowH-barH)/2;
  roundedBar(ctx,x,y,width-1,barH,C[s.kind]||C.still);ctx.save();ctx.beginPath();ctx.rect(x,y,width,barH);ctx.clip();ctx.fillStyle=C.ink;
  let label=names[s.kind];if(s.kind==='walk'){const from=app.project.points.find(p=>Math.hypot(p.position[0]-s.p0[0],p.position[1]-s.p0[1])<.01)?.id||'';if(width>90)label+=` ${from}→${s.target}`;}
  ctx.fillText(label,x+width/2,y+barH/2);ctx.restore();
 }
 if(app.data){const results=app.data.series.optimized;
  for(const [row,data] of [[1,results.links[app.link]],[2,results.fused]]){
   const y=top+row*rowH+(rowH-barH)/2;
   for(const it of intervals(app.data.decisionTime,data.state,duration))if(it.state===1)roundedBar(ctx,tx(it.start),y,tx(it.end)-tx(it.start),barH,C.fusion);
   for(const e of errors(app.project,app.data.decisionTime,data.state)){
    const x=tx(e.start),width=tx(e.end)-x;
    if(e.kind==='fn')hatch(ctx,x,y,width,barH);else if(e.kind==='fp'){ctx.strokeStyle=C.amber;ctx.lineWidth=1.4;ctx.strokeRect(x,y+.5,width,barH-1);}else {ctx.strokeStyle='#999';ctx.setLineDash([3,3]);ctx.strokeRect(x,y,width,barH);ctx.setLineDash([]);}
   }
  }
 }
 const x=tx(app.time);ctx.strokeStyle=C.red;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,top-13);ctx.lineTo(x,h-3);ctx.stroke();ctx.fillStyle=C.red;ctx.beginPath();ctx.arc(x,top-14,4,0,Math.PI*2);ctx.fill();
}
export function drawComparison(canvas,app,row){
 const {ctx,w,h}=canvasContext(canvas),r=frame(ctx,w,h-20,total(app.project));if(!app.data)return;
 line(ctx,r,app.data.decisionTime,row.score,C.fusion,1.7);ctx.setLineDash([4,3]);ctx.strokeStyle='#888';ctx.beginPath();ctx.moveTo(r.x,r.ty(app.threshold));ctx.lineTo(r.x+r.w,r.ty(app.threshold));ctx.stroke();ctx.setLineDash([]);cursor(ctx,r,app.time,false);
 for(const it of intervals(app.data.decisionTime,row.state,total(app.project)))if(it.state===1)roundedBar(ctx,r.tx(it.start),h-12,r.tx(it.end)-r.tx(it.start),7,C.fusion);
}
