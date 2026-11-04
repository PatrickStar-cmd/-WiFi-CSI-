"""Core contracts of the desktop simulator, independent of presentation mock data."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import zipfile
import numpy as np
import pytest
from studio.project import preset, validate, links
from studio.geometry import Channel3D, pose, segments, fresnel, C0
from studio.engine import Run, extract, predict_observations
from studio.exporting import export_data

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def run():
    return Run(preset())

def test_pose_continuity_and_animation_agreement():
    project=preset()
    boundaries=[s['end'] for s in segments(project)[:-1]]
    for t in boundaries:
        p=pose(project,[t-1e-7,t+1e-7])
        assert np.max(abs(np.diff(p['points'],axis=0)))<1e-5
    times=[-1,0,3.5,3.65,7.2,10,12,12.15,12.4,14.3,16.5,16.65,21,24,30]
    script="import {poseAt} from './studio/web/motion.mjs'; const i=JSON.parse(process.argv[1]); console.log(JSON.stringify(i.times.map(t=>poseAt(i.project,t))));"
    js=subprocess.run(['node','--input-type=module','-e',script,json.dumps({'project':project,'times':times})],cwd=ROOT,check=True,capture_output=True,text=True)
    values=json.loads(js.stdout);py=pose(project,times)
    np.testing.assert_allclose([v['root'] for v in values],py['root'],atol=1e-10)
    np.testing.assert_allclose([v['points'] for v in values],py['points'],atol=1e-10)
    np.testing.assert_allclose([v['angle'] for v in values],py['angle'],atol=1e-10)
    np.testing.assert_allclose([v['shoulders'] for v in values],py['shoulders'],atol=1e-10)
    np.testing.assert_allclose([v['elbows'] for v in values],py['elbows'],atol=1e-10)


def test_wave_arm_lengths_and_shoulder_attachment():
    for amplitude in (0,.3,.6):
        p=preset();p['actions'][3]['amplitude']=amplitude
        joints=pose(p,np.arange(0,30,.01))
        np.testing.assert_allclose(np.linalg.norm(joints['elbows']-joints['shoulders'],axis=-1),.30,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(joints['points'][:,1:]-joints['elbows'],axis=-1),.30,atol=1e-10)
        # Shoulder centres are embedded in the upper torso's capsule surface.
        torso_side_radius=1.25*np.sqrt(.18**2-(1.36-1.17-.15)**2)
        assert .205<torso_side_radius
        local=joints['shoulders']-joints['root'][:,None,:]
        np.testing.assert_allclose(np.linalg.norm(local[:,:,:2],axis=-1),.205,atol=1e-10)
        np.testing.assert_allclose(local[:,:,2],1.36,atol=1e-10)
        # A conservative bound includes the forearm's radius outside the torso.
        elbows=joints['elbows'][:,1,None,:];hands=joints['points'][:,2,None,:]
        along=elbows+np.linspace(0,1,31)[None,:,None]*(hands-elbows)-joints['root'][:,None,:]
        angle=joints['angle'][:,None]
        x=(along[:,:,0]*np.cos(angle)+along[:,:,1]*np.sin(angle))/.85
        y=(-along[:,:,0]*np.sin(angle)+along[:,:,1]*np.cos(angle))/1.25
        z=np.maximum(abs(along[:,:,2]-1.17)-.15,0)
        assert np.min(np.sqrt(x*x+y*y+z*z)-.18)>.047/.85

def test_geometry_height_furniture_and_fresnel():
    p=preset();a=Channel3D(p);p2=deepcopy(p);p2['nodes'][0]['position'][2]=1.8;b=Channel3D(p2)
    assert a.body_scale==b.body_scale
    assert not np.allclose(a.response('AC',[12.4]),b.response('AC',[12.4]))
    expected=np.linalg.norm(np.array([-.5,.5,1.8])-np.array([5.5,4.5,1.2]))
    assert b.static['AC'][0][1]==pytest.approx(expected)
    p3=deepcopy(p);p3['furniture'][0]['reflection']=0
    assert not np.allclose(a.background['AC'],Channel3D(p3).background['AC'])
    f=fresnel(np.array([0.,0,0]),np.array([6.,0,0]),2.462e9)
    assert 2*np.hypot(3,f['minor'])==pytest.approx(6+C0/2.462e9/2)

def test_observation_reproducibility_and_causal_features(run):
    channel=Channel3D(preset())
    np.testing.assert_array_equal(channel.observe('AC')['csi'],run.observations['AC']['csi'])
    altered=deepcopy(run.observations)
    for o in altered.values():o['csi'][o['time']>10.05]*=3
    data=extract(altered,run.project)
    past=run.data['time']<=10
    np.testing.assert_array_equal(data['features'][past],run.data['features'][past])
    np.testing.assert_array_equal(data['enhanced'][past],run.data['enhanced'][past])
    assert not np.allclose(data['enhanced'][~past],run.data['enhanced'][~past])

def test_truth_is_excluded_from_detector_inputs(run):
    data={**run.data,'truth':np.full(run.truth.shape,999,dtype=np.int32),'actions':'unused'}
    predictions=predict_observations(data,run.names,100)
    for algorithm in ('original','optimized'):
        np.testing.assert_array_equal(predictions[algorithm]['perLink'],run.raw[algorithm]['perLink'])
    before=run.observations['AC']['csi'].copy()
    changed=run.decisions(.8)
    np.testing.assert_array_equal(before,run.observations['AC']['csi'])
    assert changed['optimized']['fused']['state']!=run.decisions(.5)['optimized']['fused']['state']

def test_empty_room_and_reconfigured_topology():
    p=preset(1);p['present']=False;p['radio']['rate']=50
    p['nodes'][1]['role']='rx'
    assert links(validate(p))==['AB','AC','AD']
    r=Run(p)
    assert not r.truth.any()
    for name in r.names:
        np.testing.assert_allclose(r.channel.response(name,[0,12,20]),np.repeat(r.channel.background[name][None],3,axis=0))
    assert r.data['features'].shape[1]==3
    assert r.payload()['validCounts']==[3]*len(r.truth)

def test_saved_export_reproduces_current_controls(run,tmp_path):
    target=tmp_path/'data.zip';export_data(run,target,.8,'BD')
    with zipfile.ZipFile(target) as z:
        p=json.loads(z.read('project.json'))
        assert p['threshold']==.8 and p['selectedLink']=='BD'
        assert {'project.json','metadata.json','metrics.json','observations.npz','decisions.csv'}==set(z.namelist())
        metrics=json.loads(z.read('metrics.json'))
        assert metrics['optimized']['fusion']==run.decisions(.8)['optimized']['fused']['metrics']
        assert json.loads(z.read('metadata.json'))['models']['optimized']==run.raw['optimized']['modelHash']
