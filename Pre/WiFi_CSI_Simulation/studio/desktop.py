"""Qt desktop shell with a local WebChannel and a worker for scientific work."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
import traceback
from PySide6.QtCore import QObject, Signal, Slot, QThread, QUrl, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from .project import preset, validate
from .engine import Run
from .exporting import export_data, export_slides

ROOT=Path(__file__).parent

class Worker(QThread):
    progress=Signal(int,str)
    completed=Signal(object)
    failed=Signal(str)
    def __init__(self,project):
        super().__init__();self.project=project
    def run(self):
        try:self.completed.emit(Run(self.project,self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())

class Bridge(QObject):
    progress=Signal(int,str)
    result=Signal(str)
    error=Signal(str)
    loaded=Signal(str)
    exported=Signal(str)
    decisionsChanged=Signal(str)
    def __init__(self,window):
        super().__init__();self.window=window;self.worker=None;self.run_data=None
    @Slot(result=str)
    def initialize(self):
        return json.dumps({'presets':[preset(i) for i in range(3)],'version':'0.1.0'},ensure_ascii=False)
    @Slot(str)
    def compute(self,raw):
        if self.worker is not None and self.worker.isRunning():return
        try:project=validate(json.loads(raw))
        except Exception as e:self.error.emit(str(e));return
        self.run_data=None
        self.worker=Worker(project)
        self.worker.progress.connect(self.progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self.error)
        self.worker.start()
    @Slot(object)
    def _completed(self,run):
        self.run_data=run
        self.result.emit(json.dumps(run.payload(),ensure_ascii=False,allow_nan=False))
    @Slot(float)
    def changeThreshold(self,value):
        try:
            if self.run_data:
                self.decisionsChanged.emit(json.dumps(self.run_data.decisions(value),allow_nan=False))
        except Exception as e:self.error.emit(str(e))
    @Slot()
    def invalidate(self):
        self.run_data=None
    @Slot()
    def openScene(self):
        path,_=QFileDialog.getOpenFileName(self.window,'打开场景','','CSI Studio (*.json)')
        if not path:return
        try:
            p=validate(json.loads(Path(path).read_text(encoding='utf-8')))
            self.loaded.emit(json.dumps(p,ensure_ascii=False))
        except Exception as e:self.error.emit(str(e))
    @Slot(str)
    def saveScene(self,raw):
        try:p=validate(json.loads(raw))
        except Exception as e:self.error.emit(str(e));return
        path,_=QFileDialog.getSaveFileName(self.window,'保存场景','CSIStudio场景.v1.json','CSI Studio (*.json)')
        if path:
            Path(path).write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
            self.exported.emit(path)
    @Slot(str,str)
    def exportFile(self,kind,raw):
        settings=json.loads(raw)
        suffix,description={'image':('png','PNG 图片 (*.png)'),
                            'data':('zip','仿真数据 (*.zip)'),
                            'slides':('pptx','PowerPoint (*.pptx)')}[kind]
        if kind!='image' and self.run_data is None:
            self.error.emit('请先计算仿真。');return
        path,_=QFileDialog.getSaveFileName(self.window,'导出','CSIStudio汇报.v1.'+suffix,description)
        if not path:return
        try:
            if kind=='image':self.window.view.grab().save(path)
            elif kind=='data':export_data(self.run_data,path,settings['threshold'],settings['link'])
            else:
                with tempfile.TemporaryDirectory(prefix='csi-studio-') as tmp:
                    picture=Path(tmp)/'interface.png';self.window.view.grab().save(str(picture))
                    export_slides(self.run_data,path,picture,settings['threshold'],settings['link'])
            self.exported.emit(path)
        except Exception:self.error.emit(traceback.format_exc())

class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        text=f'JS {level.name} {source}:{line}: {message}'
        print(text,flush=True)
        if level==QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            self.parent().errors.append(text)

class Window(QMainWindow):
    def __init__(self):
        super().__init__();self.errors=[]
        self.setWindowTitle('CSI Studio');self.resize(1600,900);self.setMinimumSize(1200,675)
        self.setWindowIcon(QIcon(str(ROOT/'web'/'icon.svg')))
        self.view=QWebEngineView(self)
        self.page=Page(self);self.view.setPage(self.page)
        self.setCentralWidget(self.view)
        settings=self.view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard,False)
        self.bridge=Bridge(self);self.channel=QWebChannel(self.page)
        self.channel.registerObject('studio',self.bridge);self.page.setWebChannel(self.channel)
        self.view.load(QUrl.fromLocalFile(str((ROOT/'web'/'index.html').resolve())))
    def closeEvent(self,event):
        if self.bridge.worker and self.bridge.worker.isRunning():
            self.bridge.worker.wait()
        super().closeEvent(event)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',type=Path)
    parser.add_argument('--debug-port',type=int)
    args=parser.parse_args()
    if args.debug_port:os.environ['QTWEBENGINE_REMOTE_DEBUGGING']=str(args.debug_port)
    app=QApplication(sys.argv[:1]);app.setApplicationName('CSI Studio')
    window=Window();window.show()
    if args.smoke:
        from .smoke import Smoke
        window.smoke=Smoke(window,args.smoke,app)
        QTimer.singleShot(1200,window.smoke.start)
    return app.exec()
