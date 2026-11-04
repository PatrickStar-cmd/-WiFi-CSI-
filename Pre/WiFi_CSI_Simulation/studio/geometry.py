"""Three-dimensional paths and the single authoritative human pose sequence."""
import numpy as np
from .project import links, duration

C0=299_792_458.0
CALIBRATION=10.0
UPPER_ARM=.30
FOREARM=.30


def arm_joints(shoulder, target, pole):
    """Two-bone inverse kinematics; project the hand onto reachable space."""
    delta=target-shoulder
    length=np.linalg.norm(delta,axis=1)
    direction=delta/length[:,None]
    reach=np.clip(length,1e-6,UPPER_ARM+FOREARM-1e-6)
    hand=shoulder+direction*reach[:,None]
    along=(UPPER_ARM**2-FOREARM**2+reach**2)/(2*reach)
    bend=pole-direction*np.sum(pole*direction,axis=1)[:,None]
    bend/=np.linalg.norm(bend,axis=1)[:,None]
    height=np.sqrt(np.maximum(0,UPPER_ARM**2-along**2))
    elbow=shoulder+direction*along[:,None]+bend*height[:,None]
    return elbow,hand

def segments(project):
    points={p['id']:np.asarray(p['position'],float) for p in project['points']}
    point=points[project['points'][0]['id']].copy()
    result=[]
    start=0.
    heading=0.
    for a in project['actions']:
        end=start+a['duration']
        previous_angle=heading
        target=points[a['target']].copy() if a['kind']=='walk' else point.copy()
        if a['kind']=='walk' and np.linalg.norm(target-point)>1e-9:
            heading=float(np.arctan2(target[1]-point[1],target[0]-point[0]))
        if a['kind']=='wave':
            heading=np.deg2rad(a['heading'])
        result.append({**a,'start':start,'end':end,'p0':point.copy(),'p1':target.copy(),'angle':heading,'angle0':previous_angle})
        point=target
        start=end
    return result

def pose(project,times):
    t=np.atleast_1d(times).astype(float)
    root=np.repeat(np.asarray(project['points'][0]['position'],float)[None],len(t),axis=0)
    hands=np.zeros((len(t),2,3))
    shoulders=np.zeros_like(hands)
    elbows=np.zeros_like(hands)
    angle=np.zeros(len(t))
    kind=np.full(len(t),'still',dtype='<U8')
    for s in segments(project):
        mask=(t>=s['start'])&(t<s['end'])
        if s['start']==0:
            mask|=t<0
        if s['end']==duration(project):
            mask|=t>=s['end']
        local=np.clip(t[mask]-s['start'],0,s['duration'])
        ramp=min(.3,s['duration']/3)
        edge=np.minimum(local,s['duration']-local)
        env=np.where(edge<ramp,.5*(1-np.cos(np.pi*edge/ramp)),1)
        turn=np.arctan2(np.sin(s['angle']-s['angle0']),np.cos(s['angle']-s['angle0']))
        angle[mask]=s['angle0']+turn*.5*(1-np.cos(np.pi*np.minimum(local/ramp,1)))
        kind[mask]=s['kind']
        root[mask]=s['p0']
        forward=np.column_stack([np.cos(angle[mask]),np.sin(angle[mask]),np.zeros(mask.sum())])
        side=np.column_stack([-np.sin(angle[mask]),np.cos(angle[mask]),np.zeros(mask.sum())])
        left=.25*side+[0,0,.85]
        right=-.25*side+[0,0,.85]
        if s['kind']=='walk':
            head=.5*(local-ramp/np.pi*np.sin(np.pi*local/ramp))
            remain=s['duration']-local
            tail=s['duration']-ramp-.5*(remain-ramp/np.pi*np.sin(np.pi*remain/ramp))
            distance=np.where(local<ramp,head,np.where(local>s['duration']-ramp,tail,local-ramp/2))
            root[mask]=s['p0']+distance[:,None]/(s['duration']-ramp)*(s['p1']-s['p0'])
            swing=.12*env*np.sin(2*np.pi*1.5*local)
            left+=swing[:,None]*forward
            right-=swing[:,None]*forward
        elif s['kind']=='wave':
            displacement=s['amplitude']*env**2*np.sin(2*np.pi*s['frequency']*local)
            right+=env[:,None]*np.array([0,0,.75])+displacement[:,None]*side
            right+=.40*np.sqrt(env)[:,None]*forward
        for i,target in enumerate((left,right)):
            sign=1 if i==0 else -1
            shoulder=root[mask]+.205*sign*side+[0,0,1.36]
            pole=.7*forward+sign*side+[0,0,-.3]
            elbow,hand=arm_joints(shoulder,root[mask]+target,pole)
            shoulders[mask,i]=shoulder
            elbows[mask,i]=elbow
            hands[mask,i]=hand
    if not project['present']:
        kind[:]='empty'
    body=root+np.array([0,0,1.10])
    points=np.concatenate([body[:,None],hands],axis=1)
    return {'root':root,'points':points,'shoulders':shoulders,'elbows':elbows,'angle':angle,'kind':kind,
            'truth':np.isin(kind,['walk','wave']).astype(np.int8)}

def furniture_points(f):
    x,y,z=f['position']; w,d,h=f['size']
    return np.array([[x-w/2,y-d/2,z+h],[x+w/2,y-d/2,z+h],
                     [x-w/2,y+d/2,z+h],[x+w/2,y+d/2,z+h]])

def fresnel(a,b,frequency_hz):
    length=float(np.linalg.norm(np.asarray(b)-a))
    wavelength=C0/frequency_hz
    major=(length+wavelength/2)/2
    minor=np.sqrt(major**2-(length/2)**2)
    return {'major':major,'minor':float(minor),'length':length}

class Channel3D:
    def __init__(self,project):
        self.project=project
        self.nodes={n['id']:n for n in project['nodes']}
        self.links=links(project)
        self.k=np.r_[np.arange(-28,0),np.arange(1,29)]
        self.frequency=project['radio']['frequencyGHz']*1e9
        self.frequencies=self.frequency+self.k*project['radio']['bandwidthMHz']*1e6/64
        rng=np.random.default_rng(project['seed'])
        room=np.asarray(project['room'])
        self.scatterers=rng.uniform([.4,.4,.2],room-[.4,.4,.3],(8,3))
        self.coefficients=rng.uniform(.02,.06,8)*np.exp(1j*rng.uniform(-np.pi,np.pi,8))
        for f in project['furniture']:
            self.scatterers=np.r_[self.scatterers,furniture_points(f)]
            self.coefficients=np.r_[self.coefficients,np.full(4,-f['reflection']/4,dtype=complex)]
        self.static={link:self._static(link) for link in self.links}
        self.background={link:self._sum(*paths) for link,paths in self.static.items()}
        # Fixed physical coefficient across layouts: moving nodes never rescales the person.
        self.body_scale=.22*10**((project['radio']['bodyDb']+15)/20)

    def _sum(self,d,a):
        return np.sum(a[...,None]*np.exp(-2j*np.pi*d[...,None]*self.frequencies/C0),axis=-2)

    def gain(self,node,direction):
        n=node['antennas']
        if n==1:
            return np.ones(direction.shape[:-1],complex)
        angle=np.deg2rad(node['heading'])
        axis=np.array([-np.sin(angle),np.cos(angle),0])
        unit=direction/np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),.05)
        return np.mean(np.exp(1j*np.pi*(unit@axis)[...,None]*np.arange(n)),axis=-1)

    def point_paths(self,link,points,coefficients):
        tx,rx=(self.nodes[x] for x in link)
        a=points-np.asarray(tx['position']);b=points-np.asarray(rx['position'])
        d1=np.linalg.norm(a,axis=-1);d2=np.linalg.norm(b,axis=-1)
        gains=self.gain(tx,a)*self.gain(rx,b)
        return d1+d2,coefficients*gains/np.maximum(d1*d2,.05)

    def _static(self,link):
        tx,rx=(self.nodes[x] for x in link)
        a,b=(np.asarray(x['position']) for x in (tx,rx))
        direct=np.linalg.norm(a-b)
        images=[];receiver_images=[]
        for axis,size in enumerate(self.project['room']):
            for plane in (0,size):
                im=a.copy();im[axis]=2*plane-a[axis];images.append(im)
                im=b.copy();im[axis]=2*plane-b[axis];receiver_images.append(im)
        images=np.asarray(images)
        wall=np.linalg.norm(images-b,axis=1)
        wamp=-self.project['wallReflection']*np.array([1,1,1,1,.6,.4])/wall
        wamp=wamp*self.gain(tx,np.asarray(receiver_images)-a)*self.gain(rx,images-b)
        sd,sa=self.point_paths(link,self.scatterers,self.coefficients)
        da=self.gain(tx,(b-a)[None])[0]*self.gain(rx,(a-b)[None])[0]/direct
        return np.r_[direct,wall,sd],np.r_[da,wamp,sa]

    def response(self,link,times):
        t=np.atleast_1d(times)
        result=np.repeat(self.background[link][None],len(t),axis=0)
        if self.project['present']:
            d,a=self.point_paths(link,pose(self.project,t)['points'],np.array([1,.35,.35]))
            result+=self._sum(d,self.body_scale*a)
        return result

    def observe(self,link):
        rate=self.project['radio']['rate']
        txs=[n['id'] for n in self.project['nodes'] if n['role']=='tx']
        offset=txs.index(link[0])/(len(txs)*rate)
        times=np.arange(-CALIBRATION,duration(self.project),1/rate)+offset
        times=times[times<duration(self.project)]
        h=self.response(link,times)
        rng=np.random.default_rng([self.project['seed'],ord(link[0]),ord(link[1])])
        r=self.project['radio']
        noise_power=np.mean(abs(self.background[link])**2)*10**(-r['snr']/10)
        noise=np.sqrt(noise_power/2)*(rng.normal(size=h.shape)+1j*rng.normal(size=h.shape))
        phase=rng.uniform(-np.pi,np.pi,len(times))
        slope=rng.normal(0,r['phaseError'],len(times))
        observed=np.exp(1j*(phase[:,None]+slope[:,None]*self.k))*h+noise
        valid=rng.random(len(times))>=r['loss']
        observed[~valid]=np.nan+1j*np.nan
        return {'time':times,'csi':observed.astype(np.complex64),'valid':valid,
                'packet_id':np.arange(len(times))}
