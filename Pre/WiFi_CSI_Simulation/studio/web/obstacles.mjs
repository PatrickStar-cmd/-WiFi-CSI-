import {actionSegments} from './motion.mjs';

export const materials = {
 '轻质隔墙': {reflection:.35,lossDb:6},
 '混凝土': {reflection:.55,lossDb:15},
 '玻璃': {reflection:.30,lossDb:4},
 '木材': {reflection:.30,lossDb:3},
 '自定义': null,
};
export const furnitureDefaults={
 table:{size:[1.35,.75,.75],reflection:.30},
 chair:{size:[.45,.45,.85],reflection:.20},
 box:{size:[.9,.45,1.6],reflection:.32,lossDb:materials['木材'].lossDb},
};
export const wallDefaults=room=>({height:Math.min(2.5,room[2]),thickness:.15,material:'轻质隔墙',...materials['轻质隔墙']});

export function footprint(object){
 if(object.start){
  const {start:[ax,ay],end:[bx,by],thickness,height}=object;
  return {center:[(ax+bx)/2,(ay+by)/2,0],size:[Math.hypot(bx-ax,by-ay),thickness,height],angle:Math.atan2(by-ay,bx-ax)};
 }
 return {center:object.position,size:object.size,angle:(object.heading??0)*Math.PI/180};
}

export function footprintExtent(object){
 const {size:[width,depth],angle}=footprint(object),cos=Math.abs(Math.cos(angle)),sin=Math.abs(Math.sin(angle));
 return [(cos*width+sin*depth)/2,(sin*width+cos*depth)/2];
}

export function objectFits(object,room){
 const {center,size}=footprint(object),extent=footprintExtent(object),epsilon=1e-9;
 return size.every(value=>Number.isFinite(value)&&value>0)&&
  (!object.start||(size[0]>=.1-epsilon&&size[1]>=.02&&size[1]<=1))&&
  center.slice(0,2).every((value,i)=>value-extent[i]>=-epsilon&&value+extent[i]<=room[i]+epsilon)&&
  center[2]>=0&&center[2]+size[2]<=room[2]+epsilon;
}

export function translated(object,position){
 if(!object.start)return {...object,position:[...position]};
 const center=footprint(object).center;
 const shift=point=>point.map((value,i)=>value+position[i]-center[i]);
 return {...object,start:shift(object.start),end:shift(object.end)};
}

export function withWallDimensions(wall,length,angle){
 const {center}=footprint(wall),dx=length/2*Math.cos(angle),dy=length/2*Math.sin(angle);
 return {...wall,start:[center[0]-dx,center[1]-dy],end:[center[0]+dx,center[1]+dy]};
}

export function intersectsFootprint(object,start,end){
 const {center:[cx,cy],size,angle}=footprint(object),cos=Math.cos(angle),sin=Math.sin(angle);
 const local=([x,y])=>[cos*(x-cx)+sin*(y-cy),-sin*(x-cx)+cos*(y-cy)];
 const a=local(start),b=local(end);
 let entry=0,exit=1;
 for(let i=0;i<2;i++){
  const delta=b[i]-a[i],half=size[i]/2;
  if(Math.abs(delta)<1e-10){if(Math.abs(a[i])>half)return false;continue;}
  const t0=(-half-a[i])/delta,t1=(half-a[i])/delta;
  entry=Math.max(entry,Math.min(t0,t1));exit=Math.min(exit,Math.max(t0,t1));
  if(entry>exit)return false;
 }
 return true;
}

export function routeConflicts(project){
 if(!project.present)return [];
 const solids=[...(project.walls??[]),...project.furniture.filter(object=>object.kind==='box'&&object.position[2]<1.8)].filter(object=>object.enabled!==false);
 return actionSegments(project).flatMap(segment=>solids.filter(object=>intersectsFootprint(object,segment.p0,segment.p1)).map(object=>({segment,object})));
}
