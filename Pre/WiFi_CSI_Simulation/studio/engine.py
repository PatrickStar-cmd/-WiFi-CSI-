"""Generate one shared CSI capture, then evaluate both frozen detectors."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import time
import joblib
import numpy as np
from scipy.signal import stft
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from csi_sim.features import compute_features
from csi_sim.optimized import multiscale_features, hysteresis_states
from .project import validate, validate_route, duration, links, person_project
from .geometry import Channel3D, CALIBRATION, pose, segments

ASSETS=Path(__file__).parent/'assets'

def plain(obj):
    if isinstance(obj,np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj,dict):
        return {k:plain(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj,(float,np.floating)):
        return round(float(obj),7) if np.isfinite(obj) else None
    if isinstance(obj,np.integer):
        return int(obj)
    return obj

def extract(observations,project):
    rate=project['radio']['rate']
    times=np.arange(0,duration(project)-1e-9,.2)
    names=links(project);n=len(times);nt=int(rate)
    features=np.zeros((n,len(names),16),np.float32)
    samples=np.zeros((n,len(names),56,nt),np.float32)
    valid=np.zeros((n,len(names)),bool)
    quality=np.zeros(valid.shape)
    scales=[];noises=[]
    for li,name in enumerate(names):
        o=observations[name];a=np.abs(o['csi']).astype(float)
        cal=(o['time']<0)&o['valid']
        if cal.sum()<rate:
            raise ValueError(f'{name} 静态校准数据不足。')
        scale=np.sqrt(np.mean(a[cal]**2));a/=scale
        noise=max(float(np.median(a[cal].var(axis=0))),1e-8)
        scales.append(scale);noises.append(noise)
        for wi,end in enumerate(times):
            ids=np.flatnonzero((o['time']>end-1)&(o['time']<=end+1e-9)&o['valid'])
            if len(ids)<.8*rate or len(ids)<3:
                continue
            stamp=o['time'][ids]
            if np.max(np.diff(np.r_[end-1,stamp,end]))>.05+1e-9:
                continue
            if (np.diff(o['packet_id'][ids])==1).sum()<2:
                continue
            grid=end-np.arange(nt-1,-1,-1)/rate
            filled=np.stack([np.interp(grid,stamp,a[ids,k]) for k in range(56)],axis=1)
            f,_=compute_features(a[ids],stamp,o['packet_id'][ids],filled,rate,1e-8,20*np.log10(scale))
            features[wi,li]=f;samples[wi,li]=filled.T
            valid[wi,li]=True;quality[wi,li]=min(len(ids)/rate,1)/noise
    enhanced=np.concatenate([features,multiscale_features(samples,noises,rate,[.2,.4,1])],axis=-1)
    return {'time':times,'features':features,'enhanced':enhanced,'valid':valid,
            'quality':quality,'scales':np.asarray(scales),'noise':np.asarray(noises)}

def fuse_scores(scores,valid,quality,rule):
    good=valid&np.isfinite(scores)
    if rule=='max':
        result=np.max(np.where(good,scores,-np.inf),axis=1)
    else:
        weights=np.where(good,quality,0.)
        minimum=np.min(np.where(weights>0,weights,np.inf),axis=1)
        weights=np.minimum(weights,minimum[:,None]*10)
        total=weights.sum(axis=1)
        logw=np.full(weights.shape,-np.inf)
        np.log(weights,out=logw,where=weights>0)
        result=np.log(np.maximum(total,1e-300))-logsumexp(logw-np.where(good,scores,0),axis=1)
    result[good.sum(axis=1)==0]=np.nan
    return result

def predict_observations(data,names,rate):
    folder=ASSETS/f'rate{rate}'
    calibration=json.loads((folder/'calibration.json').read_text(encoding='utf-8'))
    out={}
    with threadpool_limits(limits=2):
        for name,file,features,key in [('original','original_B2.joblib','features','O0_original'),
                                       ('optimized','selected.joblib','enhanced','A2_optimized')]:
            model=joblib.load(folder/file)
            scores=np.full(data['valid'].shape,np.nan)
            good=data['valid']
            if good.any():
                scores[good]=np.logaddexp(0,model.decision_function(data[features][good]))
            settings=calibration[key]
            rule=settings['four_quality']['fusion']
            pooled={k:float(np.median([settings[l][k] for l in ('AC','AD','BC','BD')]))
                    for k in ('threshold','off_threshold','on_count','off_count')}
            out[name]={'perLink':scores,'fused':fuse_scores(scores,good,data['quality'],rule),
                       'settings':{l:deepcopy(settings[l] if l in settings else pooled) for l in names},
                       'fusionSettings':deepcopy(settings['four_quality']),
                       'modelHash':hashlib.sha256((folder/file).read_bytes()).hexdigest()}
    return out

def motion_events(project):
    intervals=[]
    if not project['present']:
        return intervals
    for s in segments(project):
        if s['kind']=='still':
            continue
        if intervals and abs(intervals[-1][1]-s['start'])<1e-8:
            intervals[-1][1]=s['end']
        else:
            intervals.append([s['start'],s['end']])
    return intervals

def evaluate(truth,states,times,events):
    tp=int(((truth==1)&(states==1)).sum());fp=int(((truth==0)&(states==1)).sum())
    fn=int(((truth==1)&(states!=1)).sum());tn=int(((truth==0)&(states==0)).sum())
    ratio=lambda a,b:a/b if b else None
    rises=times[(states==1)&(np.r_[0,states[:-1]]!=1)]
    delays=[];hits=0
    for start,end in events:
        detected=rises[(rises>=start-1e-9)&(rises<end)]
        if len(detected):
            hits+=1;delays.append(float(detected[0]-start))
    return {'f1':ratio(2*tp,2*tp+fp+fn),'precision':ratio(tp,tp+fp),
            'recall':ratio(tp,tp+fn),'fpr':ratio(fp,fp+tn),'delay':float(np.mean(delays)) if delays else None,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'coverage':float(np.mean(states>=0)),
            'eventRecall':ratio(hits,len(events)), 'eventCount':len(events)}

def classify(scores,settings,threshold):
    base=settings['threshold']
    relative=threshold/(1-threshold)
    state=hysteresis_states(scores,base*relative,settings['off_threshold']*relative,
                             int(settings['on_count']),int(settings['off_count']))
    # Monotonic display scale preserves stable scores; 0.5 is the frozen operating point.
    return {'score':scores/(scores+max(base,1e-12)),'state':state,
            'on':threshold,'off':(settings['off_threshold']*relative)/(settings['off_threshold']*relative+base)}

COMBINATIONS=('00','10','01','11')


def decode_people(prediction, confidence_floor):
    """Decode four joint posteriors without access to the simulated people."""
    posterior=np.asarray(prediction['posterior'])
    valid=np.asarray(prediction['valid'],dtype=bool)
    finite=np.isfinite(posterior).all(axis=1)
    safe=np.where(finite[:,None],posterior,0.)
    klass=safe.argmax(axis=1)
    accepted=valid&finite&(safe.max(axis=1)>=confidence_floor)
    scores=np.column_stack((posterior[:,1]+posterior[:,3],posterior[:,2]+posterior[:,3]))
    scores[~finite|~valid]=np.nan
    states=np.column_stack((klass&1,klass>>1))
    states[~accepted]=-1
    return scores,states


def evaluate_people(truth, states, times, events):
    people=[]
    for index in range(2):
        target=truth[:,index]; state=states[:,index]
        report=evaluate(target,state,times,events[index])
        other_motion=(target==0)&(truth[:,1-index]==1)
        report.update(unknown=int((state==-1).sum()),
                      fpr=report['fp']/int((target==0).sum()) if (target==0).any() else None,
                      crossTalk=float(np.mean(state[other_motion]==1)) if other_motion.any() else None)
        people.append(report)
    known=(states>=0).all(axis=1)
    truth_index=truth[:,0]+2*truth[:,1]
    pred_index=np.where(known,states[:,0]+2*states[:,1],4)
    confusion=np.zeros((4,5),int)
    np.add.at(confusion,(truth_index,pred_index),1)
    combos=[]
    for index,key in enumerate(COMBINATIONS):
        row=confusion[index]; count=int(row.sum()); correct=int(row[index])
        combos.append({'id':key,'count':count,'correct':correct,'unknown':int(row[-1]),
                       'accuracy':correct/count if count else None})
    return people,{'jointAccuracy':float(np.mean(truth_index==pred_index)),
                   'coverage':float(known.mean()),'confusion':confusion,
                   'combinations':combos}


class Run:
    def __init__(self,project,progress=lambda p,s:None):
        start=time.perf_counter()
        project=self.project=validate(deepcopy(project));self.names=links(project)
        validate_route(project)
        progress(3,'静态校准')
        self.channel=Channel3D(project)
        self.observations={}
        for i,name in enumerate(self.names):
            self.observations[name]=self.channel.observe(name)
            progress(5+int(40*(i+1)/len(self.names)),f'生成 CSI · {name}')
        progress(48,'提取多尺度特征')
        self.data=extract(self.observations,project)
        progress(75,'原始 / 优化算法')
        self.raw=predict_observations(self.data,self.names,project['radio']['rate'])
        self.people_results={}
        self.dual_metrics=None
        if project.get('people'):
            from .multi_detector_v3 import calibrate, geometry_config, predict

            progress(84,'校准双人活动区')
            calibration=calibrate(geometry_config(project),lambda p,s:progress(84+int(p*.08),s))
            self.people_calibration=calibration.metadata
            progress(94,'逐人识别动静')
            # The detector receives the mixed capture and pre-calibrated observation model only.
            times=self.data['time']
            observation_prediction=predict(self.observations,project['radio']['rate'],self.names,times,calibration)
            scores,states=decode_people(observation_prediction,calibration.confidence_threshold)
            individuals=[person_project(project,person) for person in project['people']]
            per_truth=np.column_stack([pose(individual,times)['truth'] for individual in individuals])
            per_events=[motion_events(individual) for individual in individuals]
            metrics,self.dual_metrics=evaluate_people(per_truth,states,times,per_events)
            grid=np.arange(0,duration(project),1/project['radio']['rate'])
            self.people_results={person['id']:{'truth':per_truth[:,i],'state':states[:,i],
                                  'score':scores[:,i],'pose':pose(individuals[i],grid),'metrics':metrics[i]}
                                 for i,person in enumerate(project['people'])}
            self.truth=per_truth.max(axis=1)
            self.events=[]
            for event_start,event_end in sorted(event for events in per_events for event in events):
                if self.events and event_start<=self.events[-1][1]+1e-9:
                    self.events[-1][1]=max(event_end,self.events[-1][1])
                else:
                    self.events.append([event_start,event_end])
        else:
            self.truth=pose(project,self.data['time'])['truth']
            self.events=motion_events(project)
        self.visual=self._visuals()
        self.seconds=time.perf_counter()-start
        progress(100,'仿真完成')

    def decisions(self,threshold=None):
        threshold=self.project['threshold'] if threshold is None else float(threshold)
        if not .02<=threshold<=.98:
            raise ValueError('阈值应在 0.02–0.98 之间。')
        output={}
        for name,raw in self.raw.items():
            def entry(score,setting):
                row=classify(score,setting,threshold)
                row['metrics']=evaluate(self.truth,row['state'],self.data['time'],self.events)
                return row
            output[name]={'links':{l:entry(raw['perLink'][:,i],raw['settings'][l]) for i,l in enumerate(self.names)},
                          'fused':entry(raw['fused'],raw['fusionSettings'])}
        return plain(output)

    def _visuals(self):
        rate=self.project['radio']['rate']
        grid=np.arange(0,duration(self.project),1/rate)
        trace={}
        for li,name in enumerate(self.names):
            o=self.observations[name]
            good=o['valid'];stamp=o['time'][good]
            raw=o['csi'][good]
            amp=np.abs(raw)/self.data['scales'][li]
            amplitude=np.interp(grid,stamp,amp[:,35])
            phase=np.unwrap(np.angle(raw),axis=1)
            basis=np.column_stack([self.channel.k,np.ones(56)])
            phase-=phase@np.linalg.pinv(basis).T@basis.T
            phase_trace=np.interp(grid,stamp,phase[:,35])
            step=max(1,int(rate/20))
            heat_t=grid[::step]
            heat=np.stack([np.interp(heat_t,stamp,amp[:,k]) for k in range(56)])
            nseg=min(128,len(grid));overlap=min(nseg-1,int(nseg*.8))
            f,st,z=stft(amplitude-amplitude.mean(),rate,nperseg=nseg,noverlap=overlap)
            mask=(f<=40)&(f>0)
            trace[name]={'amplitude':amplitude,'phase':phase_trace,
                         'heatmap':heat,'heatTime':heat_t,
                         'spectrum':20*np.log10(np.maximum(abs(z[mask]),1e-6)),
                         'spectrumTime':st,'frequencies':f[mask]}
        base_pose=next(iter(self.people_results.values()))['pose'] if self.people_results else pose(self.project,grid)
        return {'time':grid,'pose':base_pose,'traces':trace}

    def payload(self,threshold=None):
        return plain({'project':self.project,'links':self.names,'duration':duration(self.project),
                      'seconds':self.seconds,'decisionTime':self.data['time'],'truth':self.truth,
                      'validCounts':self.data['valid'].sum(axis=1),'series':self.decisions(threshold),
                      'obstructions':{l:self.channel.direct_obstructions(l) for l in self.names},
                      **({'peopleResults':self.people_results,'dualMetrics':self.dual_metrics}
                         if self.people_results else {}),
                      **self.visual,
                      'metadata':{'physics':'3d-geometric-multipath-v2','calibrationSeconds':CALIBRATION,
                                  'noiseReference':'unobstructed static geometry at the same scene layout',
                                  'featureCount':{'original':16,'optimized':40},
                                  **({'personClassifier':self.people_calibration} if self.people_results else {}),
                                  'models':{k:v['modelHash'] for k,v in self.raw.items()},
                                  'weightsOrigin':'core_optimized_20261104_v1; frozen 2D-trained models',
                                  'scoreDomain':'e/(e+frozen_threshold), e=-log(1-p)',
                                  'metricSamplingHz':5,'displaySubcarrier':8,
                                  'phaseView':'linear subcarrier phase removed'}})
