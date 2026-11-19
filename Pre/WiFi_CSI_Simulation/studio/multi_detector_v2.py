"""Zone-wise detection from causal features of mixed multi-link CSI."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .geometry import Channel3D, pose
from .multi_detector import CLASSES, MIN_LINKS, WINDOW, _echo, _link_features, geometry_config

TRAIN_SEED = 819056
SAMPLES_PER_CLASS = 384
CONFIDENCE_FLOOR = .44
START_THRESHOLDS = np.asarray([.25,.20])
CONTINUE_THRESHOLDS = np.asarray([.09,.04])


@dataclass(frozen=True)
class Calibration:
    heads: tuple
    names: tuple
    rate: int
    confidence_threshold: float
    metadata: dict


def _sample_body(rng, zone, active, times, obstacles):
    """Draw an unrelated motion and its phase within this observation window."""
    low, high = np.asarray(zone).reshape(2, 2)
    heading = rng.uniform(-np.pi, np.pi)
    speed = np.exp(rng.uniform(np.log(.10), np.log(1.2)))
    length = rng.uniform(1.6, 3.4)
    shift = speed * length * np.array([np.cos(heading), np.sin(heading)])
    shift /= max(1., np.max(abs(shift) / (.80 * (high-low))))
    clearance = abs(shift)/2
    for _ in range(16):
        mean = rng.uniform(low+clearance, high-clearance)
        start = np.r_[mean-shift/2, 0.]
        end = np.r_[mean+shift/2, 0.]
        if not any(obj.crosses(start+[0,0,1.1], end+[0,0,1.1]) or
                   obj.contains(start+[0,0,1.1]) or obj.contains(end+[0,0,1.1]) for obj in obstacles):
            break
    kind = 'walk' if rng.random() < .73 else 'wave'
    wave = {'kind':'wave','duration':length,'amplitude':rng.uniform(.11,.45),
            'frequency':rng.uniform(.8,2.5),'heading':np.rad2deg(heading)}
    move = {'kind':'walk','duration':length,'target':'p1'} if kind == 'walk' else wave
    ending = times[-1]
    if active:
        # Also model decisions near a true onset, where only a suffix has motion evidence.
        age = rng.uniform(.04, .9) if rng.random() < .18 else rng.uniform(1.25, length)
        onset = ending-age
    else:
        # Motion can have ended only recently, but the most recent sample is stationary.
        pause = rng.uniform(.05,.95) if rng.random() < .22 else 1.3
        onset = ending-length-pause
    scene = {'present':True, 'points':[{'id':'p0','position':start},{'id':'p1','position':end}],
             'actions':[{'kind':'still','duration':6+onset},move,{'kind':'still','duration':2}]}
    return pose(scene, times+6)['points']


def _local_features(amplitude, rate, floor):
    original = _link_features(amplitude, rate, floor)
    groups = amplitude.reshape(*amplitude.shape[:-1], 4, 14).mean(axis=-1)
    smooth = uniform_filter1d(groups, size=max(3, round(.04*rate)), axis=1, mode='nearest')
    extra, contrast = [], []
    variance_floor = max(float(floor), 1e-12)/14
    for seconds in (.24,.5,WINDOW):
        segment = smooth[:, -round(seconds*rate):]
        centered = segment-segment.mean(axis=1,keepdims=True)
        power = np.mean(centered**2,axis=1)
        contrast.append(np.log1p(power.mean(axis=1)/variance_floor))
        # Relative subcarrier contrast and spatially pooled derivative propagation.
        extra.append(np.log1p(np.mean(np.diff(centered,axis=2)**2,axis=1)/variance_floor))
        diff = np.diff(segment,axis=1).mean(axis=2)
        norm = (np.mean(diff**2,axis=1)+variance_floor)
        for period in (.08,.16,.32):
            lag=round(rate*period)
            if lag<diff.shape[1]/2:
                extra.append((np.mean(diff[:,:-lag]*diff[:,lag:],axis=1)/norm)[:,None])
    return np.concatenate([original,*extra],axis=1), np.column_stack(contrast)


def _mixture_features(per_link, contrast):
    # Pairwise differences expose quieter links that become active alongside a stronger mover.
    return np.concatenate([*per_link,*(a-b for i,a in enumerate(contrast) for b in contrast[i+1:])],axis=1)


def _training(config):
    rng=np.random.default_rng(TRAIN_SEED)
    rate=config['radio']['rate']
    labels=np.repeat(np.arange(4),SAMPLES_PER_CLASS)
    rng.shuffle(labels)
    offsets=np.arange(-round(WINDOW*rate)+1,1)/rate
    variants=[Channel3D({**config,'seed':TRAIN_SEED+i,'present':False}) for i in range(5)]
    features=[]
    for first in range(0,len(labels),40):
        y=labels[first:first+40]
        bits=np.column_stack((y & 1,y >> 1))
        channel=variants[(first//40)%len(variants)]
        points=[np.array([_sample_body(rng,zone,bool(flag),offsets,channel.obstacles) for flag in bits[:,i]])
                for i,zone in enumerate(config['zones'])]
        strength=rng.uniform(.8,1.2,bits.shape)
        per_link, contrasts=[], []
        for link in channel.links:
            mixed=channel.background[link]+sum(_echo(channel,link,p,strength[:,i]) for i,p in enumerate(points))
            floor=channel.noise_power[link]/2
            noise=np.sqrt(floor)*(rng.standard_normal(mixed.shape)+1j*rng.standard_normal(mixed.shape))
            feature, contrast=_local_features(np.abs(mixed+noise),rate,floor)
            missing=rng.random(len(y))<.04
            feature[missing]=np.nan
            contrast[missing]=np.nan
            per_link.append(feature)
            contrasts.append(contrast)
        features.append(_mixture_features(per_link,contrasts))
    return np.concatenate(features),labels,tuple(variants[0].links)


@lru_cache(maxsize=4)
def _fit(key):
    config=json.loads(key)
    x,y,names=_training(config)
    heads=[]
    with threadpool_limits(limits=2):
        for bit in range(2):
            model=HistGradientBoostingClassifier(max_iter=125,max_leaf_nodes=19,min_samples_leaf=15,
                                                learning_rate=.10,l2_regularization=.7,random_state=TRAIN_SEED)
            model.fit(x,(y >> bit)&1)
            heads.append(model)
    return Calibration(tuple(heads),names,config['radio']['rate'],CONFIDENCE_FLOOR,{
        'method':'mixed-amplitude spatial contrast and separate motion heads',
        'geometryHash':hashlib.sha256(key.encode()).hexdigest(),
        'calibrationSeed':TRAIN_SEED,'windowsPerCombination':SAMPLES_PER_CLASS,
        'windowSeconds':WINDOW,'classOrder':list(CLASSES),'confidenceThreshold':CONFIDENCE_FLOOR,
        'minimumValidLinks':MIN_LINKS,'startThresholds':START_THRESHOLDS.tolist(),
        'continueThresholds':CONTINUE_THRESHOLDS.tolist(),
    })


def calibrate(config,progress=None):
    if set(config)!={'room','radio','nodes','walls','furniture','wallReflection','zones'}:
        raise ValueError('双人校准只接受公开几何和活动区。')
    if progress:
        progress(0,'双人活动区校准')
    calibration=_fit(json.dumps(config,sort_keys=True,separators=(',',':')))
    if progress:
        progress(100,'双人检测器就绪')
    return calibration


def _observable_features(observations,rate,names,times):
    samples=round(WINDOW*rate)
    offsets=np.arange(samples-1,-1,-1)/rate
    valid_links=np.zeros((len(times),len(names)),dtype=bool)
    features, contrasts=[], []
    for i,link in enumerate(names):
        record=observations[link]
        usable=record['valid']&np.isfinite(record['csi']).all(axis=1)
        stamp=record['time'][usable]
        amp=np.abs(record['csi'][usable])
        windows=np.zeros((len(times),samples,56),dtype=np.float32)
        quiet=amp[stamp<0]
        if len(quiet)>=max(3,round(rate/2)):
            floor=np.median(np.mean(np.diff(quiet,axis=0)**2,axis=0))/2
            for j,now in enumerate(times):
                first=np.searchsorted(stamp,now-WINDOW,side='right')
                last=np.searchsorted(stamp,now+1e-9,side='right')
                if last-first < .76*samples:
                    continue
                t=stamp[first:last]
                if np.max(np.diff(np.r_[now-WINDOW,t,now]))>.16:
                    continue
                windows[j]=np.stack([np.interp(now-offsets,t,col) for col in amp[first:last].T],axis=1)
                valid_links[j,i]=True
        else:
            floor=1.
        f,c=_local_features(windows,rate,floor)
        f[~valid_links[:,i]]=np.nan
        c[~valid_links[:,i]]=np.nan
        features.append(f)
        contrasts.append(c)
    return _mixture_features(features,contrasts), valid_links


def _causal_joint_scores(scores, valid):
    """Translate independent motion scores with a lower continuation threshold."""
    posterior=np.full((len(scores),4),np.nan)
    active=np.zeros(2,dtype=bool)
    for index,raw in enumerate(scores):
        if not valid[index]:
            active[:]=False
            continue
        boundary=np.where(active,CONTINUE_THRESHOLDS,START_THRESHOLDS)
        scaled=raw*(1-boundary)
        p,q=scaled/(scaled+(1-raw)*boundary)
        active[:]=(p>=.5,q>=.5)
        posterior[index]=((1-p)*(1-q),p*(1-q),(1-p)*q,p*q)
    return posterior


def predict(observations,rate,names,times,calibration):
    """Infer each zone from the same mixed observations; no person schedule is supplied."""
    if tuple(names)!=calibration.names or rate!=calibration.rate:
        raise ValueError('链路或采样率与双人校准不一致。')
    features, valid_links=_observable_features(observations,rate,names,times)
    valid=valid_links.sum(axis=1)>=MIN_LINKS
    scores=np.full((len(times),2),np.nan)
    if valid.any():
        with threadpool_limits(limits=2):
            scores[valid]=np.column_stack([head.predict_proba(features[valid])[:,1] for head in calibration.heads])
    posterior=_causal_joint_scores(scores,valid)
    return {'posterior':posterior,'valid':valid,'confidence':posterior.max(axis=1),'valid_links':valid_links}
