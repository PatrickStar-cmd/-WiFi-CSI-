"""Build a self-contained Windows directory; no project venv is needed at runtime."""
from pathlib import Path
import subprocess
import sys
import os
import shutil
import json
from importlib.metadata import distribution
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import Qt
from PIL import Image

ROOT=Path(__file__).resolve().parent

def main():
    app=QApplication.instance() or QApplication([])
    work=ROOT/'build'/'studio';work.mkdir(parents=True,exist_ok=True)
    image=QImage(256,256,QImage.Format.Format_ARGB32);image.fill(Qt.GlobalColor.transparent)
    painter=QPainter(image);QSvgRenderer(str(ROOT/'studio/web/icon.svg')).render(painter);painter.end()
    png=work/'icon.png';ico=work/'icon.ico';image.save(str(png))
    with Image.open(png) as icon:icon.save(ico,format='ICO',sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
    cmd=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onedir','--windowed',
         '--name','CSI Studio','--icon',str(ico),'--distpath',str(ROOT/'dist'),
         '--workpath',str(work/'pyinstaller'),'--specpath',str(work),
         '--add-data',str(ROOT/'studio/web')+';studio/web',
         '--add-data',str(ROOT/'studio/assets/rate50')+';studio/assets/rate50',
         '--add-data',str(ROOT/'studio/assets/rate100')+';studio/assets/rate100',
         '--hidden-import','sklearn.ensemble._hist_gradient_boosting.gradient_boosting',
         '--hidden-import','sklearn._loss._loss',
         '--exclude-module','torch','--exclude-module','matplotlib',
         '--exclude-module','tkinter','--exclude-module','pytest',
         str(ROOT/'run_studio.py')]
    target=ROOT/'dist'/'CSI Studio'
    if not target.resolve().is_relative_to((ROOT/'dist').resolve()):
        raise ValueError('Build output must stay inside the project dist directory.')
    # Resolve Windows ICU/API DLLs from Windows, not unrelated tools on PATH.
    environment=os.environ.copy()
    system=Path(environment['SYSTEMROOT'])
    packages=ROOT/'.venv/Lib/site-packages'
    environment['PATH']=os.pathsep.join(map(str,[Path(sys.executable).parent,
        packages/'PySide6',packages/'shiboken6',system/'System32',system]))
    subprocess.run(cmd,cwd=ROOT,env=environment,check=True)
    shutil.copy2(ROOT/'studio/README.md',target/'CSI Studio使用说明.v6.md')
    licenses=target/'licenses';licenses.mkdir(exist_ok=True)
    for file in ('vendor/THREE-LICENSE.txt','vendor/LUCIDE-LICENSE.txt','SourceHanSans-OFL.txt'):
        shutil.copy2(ROOT/'studio/web'/file,licenses/Path(file).name)
    packages={}
    for name in ('numpy','scipy','scikit-learn','joblib','threadpoolctl','Pillow','h5py',
                 'PySide6','PySide6-Essentials','PySide6-WebEngine','PySide6-Pdf','shiboken6',
                 'python-pptx','lxml','XlsxWriter'):
        dist=distribution(name);packages[name]=dist.version
        for file in dist.files or []:
            if any(part.lower() in ('licenses','license') for part in file.parts) or file.name.lower().startswith(('license','copying','notice')):
                if '.dist-info' not in str(file):continue
                destination=licenses/name/str(file).split('.dist-info/',1)[-1]
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(dist.locate_file(file),destination)
    (target/'runtime-versions.json').write_text(json.dumps(packages,indent=2),encoding='utf-8')
    (target/'release.json').write_text(json.dumps({'app':'CSI Studio','version':'0.2.4'},indent=2),encoding='utf-8')
    print(target/'CSI Studio.exe')

if __name__=='__main__':main()
