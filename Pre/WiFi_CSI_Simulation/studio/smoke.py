"""Exercise the real Qt/WebGL window, then save screenshots and a test receipt."""
from pathlib import Path
import json
import time
import hashlib
from PySide6.QtCore import QObject, QTimer, Qt, QPoint, QRect
from PySide6.QtTest import QTest
from unittest.mock import patch
from .exporting import export_data, export_slides

class Smoke(QObject):
    def __init__(self,window,folder,app):
        super().__init__();self.window=window;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.app=app;self.started=time.monotonic();self.stage=0;self.results=[];self.next_at=0
    def js(self,text,callback=None):
        if callback:self.window.page.runJavaScript(text,callback)
        else:self.window.page.runJavaScript(text)
    def start(self):
        self.window.activateWindow()
        self.timer=QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(350)
    def poll(self):
        if time.monotonic()-self.started>110:
            self.finish('timeout');return
        if time.monotonic()<self.next_at:return
        self.js('JSON.stringify(window.studio?.snapshot() ?? null)',self.receive)
    def receive(self,raw):
        try:self.advance(raw)
        except Exception as e:self.finish(f'{type(e).__name__} at stage {self.stage}: {e}')
    def drag(self,points):
        """Send real mouse events through Qt to the embedded browser."""
        x,y,dx,dy=json.loads(points)
        target=self.window.view.focusProxy()
        self.window.activateWindow();target.setFocus()
        QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(x),round(y)))
        for i in range(1,9):
            QTest.mouseMove(target,QPoint(round(x+dx*i/8),round(y+dy*i/8)),25)
        QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(x+dx),round(y+dy)))
    def advance(self,raw):
        if not raw:return
        state=json.loads(raw)
        if not state:return
        if state['errors'] or self.window.errors:self.finish('javascript errors');return
        if self.stage==0:
            assert state['webgl'] and not state['hasData']
            self.results.append({'check':'initial','state':state})
            self.window.view.grab().save(str(self.folder/'01-ready.png'))
            self.js("document.querySelector('#compute').click()");self.stage=1
        elif self.stage==1 and state['hasData'] and not state['busy']:
            self.stage=1.5;self.next_at=time.monotonic()+.7
        elif self.stage==1.5:
            assert state['truth']=='挥手' and state['prediction']=='运动'
            assert state['csiMode']=='spectrum'
            self.assert_layout(state)
            self.results.append({'check':'computed','state':state})
            self.window.view.grab().save(str(self.folder/'02-main.png'))
            self.initial_hash=hashlib.sha256(self.window.bridge.run_data.observations['AC']['csi'].tobytes()).hexdigest()
            export_data(self.window.bridge.run_data,self.folder/'results.zip',state['threshold'],state['link'])
            export_slides(self.window.bridge.run_data,self.folder/'report.pptx',self.folder/'02-main.png',state['threshold'],state['link'])
            self.pose_frames=[(12.0,'front'),(12.15,'front'),(12.4,'front'),(12.5,'front'),(12.833,'front'),(12.4,'side'),(16.35,'side'),(16.5,'front')]
            self.pose_index=0;self.stage=1.6;self.prepare_pose_frame()
        elif self.stage==1.6:
            t,angle=self.pose_frames[self.pose_index]
            image_path=str(self.folder/f'pose-{self.pose_index:02}-{angle}-{t:.3f}.png')
            self.js("JSON.stringify((()=>{const r=window.studio.view.container.getBoundingClientRect();return [r.x,r.y,r.width,r.height]})())",lambda raw,path=image_path:self.window.view.grab(QRect(*[round(v) for v in json.loads(raw)])).save(path))
            self.pose_index+=1
            if self.pose_index<len(self.pose_frames):self.prepare_pose_frame()
            else:
                self.results.append({'check':'wave_keyframes','count':len(self.pose_frames),'views':['front','side']})
                self.js("{const v=window.studio.view;v.camera.zoom=1;v.camera.updateProjectionMatrix();v.resetCamera();for(const layer of ['fresnel','links','path','furniture'])v.setLayer(layer,true);}window.studio.setTime(12.4)")
                self.stage=1.65;self.next_at=time.monotonic()+.6
        elif self.stage==1.65:
            self.js("document.querySelector('#demo').click()");self.stage=1.66;self.next_at=time.monotonic()+.5
        elif self.stage==1.66:
            assert state['demo']
            self.assert_layout(state)
            self.window.view.grab().save(str(self.folder/'02-demo.png'))
            self.results.append({'check':'demo_mode','passed':True})
            self.js("{const z=document.querySelector('#zoom');z.value='2';z.dispatchEvent(new Event('input'));}window.studio.setTime(12.4)")
            self.stage=1.67;self.next_at=time.monotonic()+.4
        elif self.stage==1.67:
            self.assert_layout(state)
            self.js("JSON.stringify((()=>{const c=document.querySelector('#stacked'),r=c.getBoundingClientRect(),k=c.axis.keys.find(k=>k.id==='P0');return [r.x+k.x,r.y+c.axis.ruler-5,0,0]})())",self.drag)
            self.stage=1.68;self.next_at=time.monotonic()+.4
        elif self.stage==1.68:
            assert abs(state['time']-3.5)<1e-6
            self.results.append({'check':'shared_zoom_and_keyframe','passed':True})
            self.js("{const z=document.querySelector('#zoom');z.value='1';z.dispatchEvent(new Event('input'));}document.querySelector('#demo').click();window.studio.setTime(12.4)")
            self.stage=1.7;self.next_at=time.monotonic()+.4
        elif self.stage==1.7:
            self.js("document.querySelector('[data-csi=heatmap]').click(); document.querySelector('#compare-toggle').click()")
            self.stage=2;self.next_at=time.monotonic()+.7
        elif self.stage==2:
            assert state['comparison']
            self.assert_layout(state)
            self.results.append({'check':'comparison','state':state});self.window.view.grab().save(str(self.folder/'03-comparison.png'))
            self.js("document.querySelector('#compare-close').click();document.querySelector('[data-csi=heatmap]').click();window.studio.setLink('BD');window.studio.setTime(19.4)")
            self.stage=3;self.next_at=time.monotonic()+.6
        elif self.stage==3:
            assert state['link']=='BD' and state['time']==19.4
            self.results.append({'check':'link_and_heatmap','state':state});self.window.view.grab().save(str(self.folder/'04-heatmap.png'))
            self.js("document.querySelector('[data-csi=amplitude]').click()")
            self.stage=3.1;self.next_at=time.monotonic()+.4
        elif self.stage==3.1:
            assert state['csiMode']=='amplitude'
            self.assert_layout(state)
            self.window.view.grab().save(str(self.folder/'04-amplitude.png'))
            self.js("document.querySelector('[data-csi=phase]').click()")
            self.stage=3.2;self.next_at=time.monotonic()+.4
        elif self.stage==3.2:
            assert state['csiMode']=='phase'
            self.assert_layout(state)
            self.window.view.grab().save(str(self.folder/'04-phase.png'))
            self.results.append({'check':'four_CSI_views','passed':True})
            self.js("document.querySelector('[data-csi=spectrum]').click()")
            self.stage=3.3;self.next_at=time.monotonic()+.4
        elif self.stage==3.3:
            self.js("JSON.stringify((()=>{const c=document.querySelector('#stacked'),r=c.getBoundingClientRect(),a=c.axis;return [r.x+a.end+2,r.y+a.score.ty(.5),0,-.3*a.score.h]})())",self.drag)
            self.stage=4;self.next_at=time.monotonic()+.6
        elif self.stage==4:
            assert .78<state['threshold']<.82
            self.results.append({'check':'native_threshold_drag_and_spectrum','state':state});self.window.view.grab().save(str(self.folder/'05-spectrum.png'))
            assert hashlib.sha256(self.window.bridge.run_data.observations['AC']['csi'].tobytes()).hexdigest()==self.initial_hash
            self.js("window.studio.switchTab('nodes');let el=document.querySelector('#node-2');el.value='1.8';el.dispatchEvent(new Event('change',{bubbles:true}))")
            self.stage=5;self.next_at=time.monotonic()+.4
        elif self.stage==5:
            self.results.append({'check':'height_invalidates','state':state})
            if state['hasData']:self.finish('stale data');return
            self.js("document.querySelector('#compute').click()");self.stage=6
        elif self.stage==6 and state['hasData'] and not state['busy']:
            different=hashlib.sha256(self.window.bridge.run_data.observations['AC']['csi'].tobytes()).hexdigest()!=self.initial_hash
            self.results.append({'check':'height_changes_CSI','different':different,'state':state})
            assert different
            self.window.activateWindow()
            self.js("document.querySelector('#view-top').click()")
            self.stage=7;self.next_at=time.monotonic()+1.0
        elif self.stage==7:
            self.window.view.grab().save(str(self.folder/'drag-before.png'))
            self.node_before=state['project']['nodes'][0]['position'][:]
            self.stage=8;self.next_at=time.monotonic()+.8
            self.js("""(()=>{const v=window.studio.view,r=v.container.getBoundingClientRect(),p=v.nodeGroups.A.position.clone();p.z=window.studio.app.project.nodes[0].position[2];p.project(v.camera);return JSON.stringify([r.left+(p.x*.5+.5)*r.width,r.top+(-p.y*.5+.5)*r.height,45,-24]);})()""",self.drag)
        elif self.stage==8:
            after=state['project']['nodes'][0]['position']
            self.window.view.grab().save(str(self.folder/'drag-after.png'))
            assert after[:2]!=self.node_before[:2], 'node drag did not update XY'
            assert after[2]==self.node_before[2] and not state['hasData']
            self.results.append({'check':'native_node_drag','before':self.node_before,'after':after})
            self.js("window.studio.switchTab('actions');document.querySelector('#add-point').click()")
            self.stage=9;self.next_at=time.monotonic()+.5
        elif self.stage==9:
            self.stage=10;self.next_at=time.monotonic()+.7
            self.js("""(()=>{const v=window.studio.view,r=v.container.getBoundingClientRect(),p=v.nodeGroups.A.position.clone().set(2,4,0).project(v.camera);return JSON.stringify([r.left+(p.x*.5+.5)*r.width,r.top+(-p.y*.5+.5)*r.height,0,0]);})()""",self.drag)
        elif self.stage==10:
            assert len(state['project']['points'])==4, 'floor click did not add waypoint'
            self.results.append({'check':'native_floor_waypoint','point':state['project']['points'][-1]})
            self.js("document.querySelector('[data-add-action=wave]').click();{const input=document.querySelector('#action-duration');input.value='2.5';input.dispatchEvent(new Event('change',{bubbles:true}));}document.querySelector('#step-forward').click()")
            self.stage=11;self.next_at=time.monotonic()+.6
        elif self.stage==11:
            assert state['project']['actions'][-1]['kind']=='wave' and state['project']['actions'][-1]['duration']==2.5
            assert abs(state['time']-19.41)<1e-8
            self.results.append({'check':'action_edit_and_frame_step','state':state})
            self.window.view.grab().save(str(self.folder/'06-guide-and-editor.png'))
            self.saved_project=state['project'];path=str((self.folder/'edited-scene.json').resolve())
            with patch('studio.desktop.QFileDialog.getSaveFileName',return_value=(path,'JSON')):
                self.window.bridge.saveScene(json.dumps(self.saved_project))
            assert json.loads(Path(path).read_text(encoding='utf-8'))==self.saved_project
            self.js("window.studio.reset()")
            self.stage=12;self.next_at=time.monotonic()+.4
        elif self.stage==12:
            assert len(state['project']['points'])==3
            with patch('studio.desktop.QFileDialog.getOpenFileName',return_value=(str((self.folder/'edited-scene.json').resolve()),'JSON')):
                self.window.bridge.openScene()
            self.stage=13;self.next_at=time.monotonic()+.4
        elif self.stage==13:
            assert state['project']==self.saved_project, 'scene open/save round trip'
            self.results.append({'check':'scene_file_roundtrip','same':True})
            self.window.resize(1280,720)
            self.js("window.studio.reset();document.querySelector('#view-3d').click();window.studio.setTime(12.4)")
            self.stage=14;self.next_at=time.monotonic()+.7
        elif self.stage==14:
            self.assert_layout(state)
            self.window.view.grab().save(str(self.folder/'07-1280x720.png'))
            self.results.append({'check':'projection_window','width':self.window.view.width(),'height':self.window.view.height(),'webgl':state['webgl']})
            self.js("document.querySelector('#guide').click()");self.stage=15;self.next_at=time.monotonic()+.4
        elif self.stage==15:
            assert not state['hasData'] and len(state['project']['points'])==3
            self.results.append({'check':'guide_resets_scene','passed':True})
            self.window.view.grab().save(str(self.folder/'08-guide.png'))
            self.finish('passed')
    def assert_layout(self,state):
        layout=state['layout']
        assert not layout['rightHasCanvas']
        assert max(r['y'] for r in layout['numbers'])-min(r['y'] for r in layout['numbers'])<.1
        assert max(r['y'] for r in layout['labels'])-min(r['y'] for r in layout['labels'])<.1
        assert layout['viewport']['y']+layout['viewport']['h']<=layout['timeSection']['y']+.1
        for label,row in zip(layout['tracks'][1:],layout['axis']['rows'].values()):
            assert abs(label['y']-layout['canvas']['y']-row[0])<.1
            assert abs(label['h']-row[1])<.1
    def prepare_pose_frame(self):
        t,angle=self.pose_frames[self.pose_index]
        location=[7,2.5,2.1] if angle=='front' else [3,-1.5,2.1]
        self.js(f"window.studio.setTime({t});{{const v=window.studio.view;v.controls.target.set(3,2.5,1.18);v.camera.position.set({','.join(map(str,location))});v.camera.zoom=1.65;v.camera.updateProjectionMatrix();v.controls.update();for(const layer of ['fresnel','links','path','furniture'])v.setLayer(layer,false);}}")
        self.next_at=time.monotonic()+.65
    def finish(self,status):
        self.timer.stop()
        report={'status':status,'seconds':time.monotonic()-self.started,'checks':self.results,'consoleErrors':self.window.errors}
        (self.folder/'smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('SMOKE '+status,flush=True);self.window.close();self.app.exit(0 if status=='passed' else 1)
