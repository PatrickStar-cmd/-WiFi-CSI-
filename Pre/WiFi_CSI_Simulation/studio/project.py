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
    return p

def links(project):
    return [a['id']+b['id'] for a in project['nodes'] if a['role']=='tx'
            for b in project['nodes'] if b['role']=='rx']

def duration(project):
    return sum(a['duration'] for a in project['actions'])

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
    if not p['points'] or len({x['id'] for x in p['points']})!=len(p['points']):
        raise ValueError('请设置名称互不重复的路径点。')
    for point in p['points']:
        if len(point['position'])!=3 or point['position'][2]!=0 or any(not math.isfinite(x) or x<.2 or x>p['room'][i]-.2 for i,x in enumerate(point['position'][:2])):
            raise ValueError('路径点应位于室内地面，并与墙面间隔 0.2 m。')
    if not p['actions'] or not .2<=duration(p)<=180:
        raise ValueError('动作总时长应在 0.2–180 秒之间。')
    point_ids={x['id'] for x in p['points']}
    for a in p['actions']:
        if a['kind'] not in ('still','walk','wave') or not math.isfinite(a['duration']) or a['duration']<.2:
            raise ValueError('动作类型或持续时间无效。')
        if a['kind']=='walk' and a['target'] not in point_ids:
            raise ValueError('走动动作的目标路径点已被删除，请重新选择。')
        if a['kind']=='wave' and not (0<=a['amplitude']<=.6 and .1<=a['frequency']<=5):
            raise ValueError('挥手摆幅应在 0–0.6 m，频率应在 0.1–5 Hz。')
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
    for f in p['furniture']:
        if not 0<=f['reflection']<=1 or any(s<=0 for s in f['size']):
            raise ValueError('家具尺寸及反射系数无效。')
        if any(not math.isfinite(v) or v<0 or v>p['room'][i] for i,v in enumerate(f['position'])):
            raise ValueError('家具应位于房间内。')
        if any(f['position'][i]-f['size'][i]/2<0 or f['position'][i]+f['size'][i]/2>p['room'][i] for i in (0,1)) or f['position'][2]+f['size'][2]>p['room'][2]:
            raise ValueError('家具的完整尺寸应位于房间内。')
    return p
