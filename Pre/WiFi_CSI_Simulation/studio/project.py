"""Editable scene format. All world coordinates use metres, with Z vertical."""
from copy import deepcopy
import math

DEFAULT = {
    'format': 'csi-studio-1', 'name': '01单人走动',
    'room': [6.0, 5.0, 3.0], 'present': True, 'seed': 20261104,
    'radio': {'frequencyGHz': 2.462, 'bandwidthMHz': 20, 'rate': 100,
              'snr': 25, 'bodyDb': -15, 'loss': 0, 'phaseError': 0.02},
    'nodes': [
        {'id': 'A', 'role': 'tx', 'position': [.5,.5,1.2], 'antennas': 1, 'heading': 45},
        {'id': 'B', 'role': 'tx', 'position': [5.5,.5,1.2], 'antennas': 1, 'heading': 135},
        {'id': 'C', 'role': 'rx', 'position': [5.5,4.5,1.2], 'antennas': 1, 'heading': 225},
        {'id': 'D', 'role': 'rx', 'position': [.5,4.5,1.2], 'antennas': 1, 'heading': 315}],
    'points': [{'id':'P0','position':[1.1,1.0,0]},
               {'id':'P1','position':[3.0,2.5,0]},
               {'id':'P2','position':[4.8,1.0,0]}],
    'actions': [{'kind':'still','duration':3.5},
                {'kind':'walk','duration':6.5,'target':'P1'},
                {'kind':'still','duration':2},
                {'kind':'wave','duration':4.5,'amplitude':.30,'frequency':1.5,'heading':0},
                {'kind':'walk','duration':7.5,'target':'P2'},
                {'kind':'still','duration':6}],
    'furniture': [
        {'id':'table','kind':'table','name':'木桌','position':[4.0,3.8,0],
         'size':[1.35,.75,.75],'reflection':.30},
        {'id':'chair1','kind':'chair','name':'木椅','position':[3.65,3.05,0],
         'size':[.45,.45,.85],'reflection':.20},
        {'id':'chair2','kind':'chair','name':'木椅','position':[4.35,3.05,0],
         'size':[.45,.45,.85],'reflection':.20}],
    'walls': [],
    'wallReflection': .5, 'selectedLink':'AC', 'threshold': .5,
}

def preset(index=0):
    p=deepcopy(DEFAULT)
    if index==1:
        p['name']='02挥手检测'
        p['points']=[{'id':'P0','position':[3.0,2.5,0]}]
        p['actions']=[{'kind':'still','duration':4},
                      {'kind':'wave','duration':6,'amplitude':.20,'frequency':1.2,'heading':0},
                      {'kind':'still','duration':4},
                      {'kind':'wave','duration':8,'amplitude':.35,'frequency':1.8,'heading':45},
                      {'kind':'still','duration':8}]
    elif index==2:
        p['name']='03算法对比'
        p['radio']['snr']=20
    elif index==3:
        p['name']='04双人动静对照'
        p['people']=[
            {'id':'P1','name':'人员 1','zone':[.45,.65,2.35,4.3],
             'points':[{'id':'P0','position':[1.05,1.1,0]},
                       {'id':'P1','position':[1.45,3.2,0]},
                       {'id':'P2','position':[.9,2.2,0]}],
             'actions':[{'kind':'still','duration':4},
                        {'kind':'walk','duration':6,'target':'P1'},
                        {'kind':'walk','duration':6,'target':'P2'},
                        {'kind':'still','duration':14}]},
            {'id':'P2','name':'人员 2','zone':[3.65,.65,5.55,4.3],
             'points':[{'id':'P0','position':[4.95,1.1,0]},
                       {'id':'P1','position':[4.85,2.5,0]}],
             'actions':[{'kind':'still','duration':10},
                        {'kind':'walk','duration':6,'target':'P1'},
                        {'kind':'wave','duration':6,'amplitude':.36,'frequency':1.45,'heading':80},
                        {'kind':'still','duration':8}]},
        ]
    return p

def links(project):
    return [a['id']+b['id'] for a in project['nodes'] if a['role']=='tx'
            for b in project['nodes'] if b['role']=='rx']

def duration(project):
    if 'people' in project:
        return max(duration(person) for person in project['people'])
    return sum(a['duration'] for a in project['actions'])


def person_project(project, person):
    """Project one person's schedule to the shared playback duration."""
    scene={key:value for key,value in project.items() if key!='people'}
    scene.update(present=True,points=person['points'],actions=list(person['actions']))
    remaining=duration(project)-duration(person)
    if remaining>1e-9:
        scene['actions'].append({'kind':'still','duration':remaining})
    return scene

def within_room_xy(point, room):
    return len(point)==2 and all(math.isfinite(x) and -1e-9<=x<=room[i]+1e-9 for i,x in enumerate(point))


def validate_objects(project):
    room=project['room']
    walls=project.setdefault('walls',[])
    objects=project['furniture']+walls
    ids=[obj['id'] for obj in objects]
    if any(not isinstance(key,str) or not key for key in ids) or len(set(ids))!=len(ids):
        raise ValueError('场景对象应有互不重复的编号。')
    for obj in objects:
        if not isinstance(obj['name'],str) or not obj['name'].strip():
            raise ValueError('请输入场景对象的名称。')
        if not isinstance(obj.get('enabled',True),bool) or not isinstance(obj.get('material',''),str):
            raise ValueError('场景对象的参与计算状态或材质无效。')
        if not 0<=obj['reflection']<=1:
            raise ValueError('反射系数应在 0–1 之间。')

    for furniture in project['furniture']:
        position=furniture['position']; size=furniture['size']; heading=furniture.get('heading',0)
        if furniture['kind'] not in ('table','chair','box'):
            raise ValueError('请选择桌子、椅子或通用障碍物。')
        if len(size)!=3 or any(not math.isfinite(s) or s<=0 for s in size) or not math.isfinite(heading):
            raise ValueError('家具尺寸或朝向无效。')
        if len(position)!=3 or any(not math.isfinite(s) for s in position):
            raise ValueError('家具坐标应为有效数值。')
        x,y,z=position; length,width,height=size
        angle=math.radians(heading % 360)
        cos,sin=abs(math.cos(angle)),abs(math.sin(angle))
        half_x=(length*cos+width*sin)/2; half_y=(length*sin+width*cos)/2
        if not (within_room_xy((x-half_x,y-half_y),room) and
                within_room_xy((x+half_x,y+half_y),room) and 0<=z and z+height<=room[2]+1e-9):
            raise ValueError(f"{furniture['name']} 的完整外形应位于房间内。")
        if furniture['kind']=='box' and not 0<=furniture['lossDb']<=60:
            raise ValueError('障碍物穿透损耗应在 0–60 dB 之间。')

    for wall in walls:
        a,b=wall['start'],wall['end']
        height,thickness,loss=wall['height'],wall['thickness'],wall['lossDb']
        if not (within_room_xy(a,room) and within_room_xy(b,room)):
            raise ValueError(f"{wall['name']} 的两端应位于房间内。")
        length=math.dist(a,b)
        if length<.1-1e-9 or not (0<height<=room[2] and .02<=thickness<=1 and 0<=loss<=60):
            raise ValueError('隔墙长度至少 0.10 m，厚度 0.02–1 m，穿透损耗 0–60 dB，高度应在房间内。')
        dx=-(b[1]-a[1])*thickness/(2*length); dy=(b[0]-a[0])*thickness/(2*length)
        if any(not within_room_xy((x+s*dx,y+s*dy),room) for x,y in (a,b) for s in (-1,1)):
            raise ValueError(f"{wall['name']} 的完整外形应位于房间内。")


def validate_route(project):
    from .obstacles import box_prism, wall_prism

    blockers=[(w['name'],wall_prism(w)) for w in project['walls'] if w.get('enabled',True)]
    blockers.extend((f['name'],box_prism(f)) for f in project['furniture']
                    if f['kind']=='box' and f.get('enabled',True))
    tracks=[(person['name']+'：',person) for person in project['people']] if 'people' in project else [('',project)]
    for name,prism in blockers:
        for node in project['nodes']:
            if prism.contains(node['position']):
                raise ValueError(f"节点 {node['id']} 位于 {name} 内，请移出障碍物。")
        bottom=prism.center[2]-prism.half_size[2]
        top=min(1.8,prism.center[2]+prism.half_size[2])
        if not project['present'] or bottom>=top:
            continue
        z=(bottom+top)/2
        for prefix,track in tracks:
            route={p['id']:(*p['position'][:2],z) for p in track['points']}
            current=track['points'][0]['id']
            if prism.contains(route[current]):
                raise ValueError(f"{prefix}起点 {current} 位于 {name} 内，请移动起点。")
            for action in track['actions']:
                if action['kind']=='walk':
                    target=action['target']
                    if prism.crosses(route[current],route[target]) or prism.contains(route[target]):
                        raise ValueError(f'{prefix}路线 {current} → {target} 与 {name} 相交，请调整路径点。')
                    current=target


def validate_track(track, room):
    if not track['points'] or len({x['id'] for x in track['points']})!=len(track['points']):
        raise ValueError('请设置名称互不重复的路径点。')
    for point in track['points']:
        pos=point['position']
        if len(pos)!=3 or pos[2]!=0 or any(not math.isfinite(x) or x<.2 or x>room[i]-.2 for i,x in enumerate(pos[:2])):
            raise ValueError('路径点应位于室内地面，并与墙面间隔 0.2 m。')
    if not track['actions'] or not .2<=duration(track)<=180:
        raise ValueError('动作总时长应在 0.2–180 秒之间。')
    point_ids={x['id'] for x in track['points']}
    for a in track['actions']:
        if a['kind'] not in ('still','walk','wave') or not math.isfinite(a['duration']) or a['duration']<.2:
            raise ValueError('动作类型或持续时间无效。')
        if a['kind']=='walk' and a['target'] not in point_ids:
            raise ValueError('走动动作的目标路径点已被删除，请重新选择。')
        if a['kind']=='wave' and not (0<=a['amplitude']<=.6 and .1<=a['frequency']<=5 and math.isfinite(a['heading'])):
            raise ValueError('挥手摆幅应在 0–0.6 m，频率应在 0.1–5 Hz，朝向应为有效角度。')


def validate_people(project):
    people=project['people']
    room=project['room']
    if not isinstance(people,list) or [person['id'] for person in people]!=['P1','P2']:
        raise ValueError('双人场景应依次包含人员 P1 和 P2。')
    if project.get('present') is not True:
        raise ValueError('双人场景需启用人员。')
    for person in people:
        name=person['name']
        if not isinstance(name,str) or not name.strip():
            raise ValueError('请输入人员名称。')
        zone=person['zone']
        if (len(zone)!=4 or not within_room_xy(zone[:2],room) or not within_room_xy(zone[2:],room)
                or zone[0]>=zone[2] or zone[1]>=zone[3]):
            raise ValueError(f'{name} 的活动区应为房间内有效的矩形。')
        try:
            validate_track(person,room)
        except ValueError as exc:
            raise ValueError(f'{name}：{exc}') from exc
    a,b=(person['zone'] for person in people)
    if all(max(a[i],b[i])<min(a[i+2],b[i+2]) for i in range(2)):
        raise ValueError('两人的活动区应互不重叠。')


def validate(p):
    if p['format']!='csi-studio-1':
        raise ValueError('请选择 CSI Studio 场景文件。')
    if len(p['room'])!=3 or any(not math.isfinite(x) or x<2 or x>20 for x in p['room']):
        raise ValueError('房间长、宽、高应在 2–20 m 范围内。')
    if not 2<=len(p['nodes'])<=4 or not links(p):
        raise ValueError('请配置 2–4 个节点，并至少保留一个发射端和一个接收端。')
    ids=[n['id'] for n in p['nodes']]
    if len(set(ids))!=len(ids) or any(x not in 'ABCD' or len(x)!=1 for x in ids):
        raise ValueError('节点名称使用互不重复的 A、B、C、D。')
    for n in p['nodes']:
        if n['role'] not in ('tx','rx') or n['antennas'] not in (1,2,4):
            raise ValueError('节点角色或天线数无效。')
        if not math.isfinite(n['heading']):
            raise ValueError('节点朝向应为有效角度。')
        if len(n['position'])!=3 or any(not math.isfinite(x) or x<0 or x>p['room'][i] for i,x in enumerate(n['position'])):
            raise ValueError(f"节点 {n['id']} 应位于房间内。")
    for i,a in enumerate(p['nodes']):
        for b in p['nodes'][i+1:]:
            if math.dist(a['position'],b['position'])<.1:
                raise ValueError('节点之间至少间隔 0.1 m。')
    if 'people' in p:
        validate_people(p)
    else:
        validate_track(p,p['room'])
    r=p['radio']
    if r['rate'] not in (50,100) or r['bandwidthMHz'] not in (20,40):
        raise ValueError('采样率请选择 50 / 100 Hz，带宽请选择 20 / 40 MHz。')
    if not (1<=r['frequencyGHz']<=7 and 0<=r['snr']<=60 and 0<=r['loss']<=.5):
        raise ValueError('频率、信噪比或丢包率超出可选范围。')
    if not (-40<=r['bodyDb']<=0 and 0<=r['phaseError']<=.2):
        raise ValueError('人体回波应在 −40–0 dB，相位扰动应在 0–0.2 之间。')
    if not isinstance(p['seed'],int) or not 0<=p['seed']<2**32:
        raise ValueError('随机种子应为 0–4294967295 之间的整数。')
    if not 0<=p['wallReflection']<=1 or not 0.02<=p['threshold']<=.98:
        raise ValueError('反射系数或检测阈值无效。')
    validate_objects(p)
    return p
