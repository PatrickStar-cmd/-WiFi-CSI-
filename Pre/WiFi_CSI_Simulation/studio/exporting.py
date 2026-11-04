"""Local exports contain the same scene, CSI, decisions, and model provenance."""
from pathlib import Path
from copy import deepcopy
from io import BytesIO, StringIO
import csv
import json
import zipfile
import numpy as np
from .engine import plain
from .geometry import segments

def export_data(run,path,threshold,selected_link):
    payload=run.payload(threshold)
    output=StringIO(newline='')
    writer=csv.writer(output)
    writer.writerow(['time_s','truth','algorithm','topology','score','state','threshold'])
    for algorithm,results in payload['series'].items():
        rows={**results['links'],'fusion':results['fused']}
        for topology,row in rows.items():
            writer.writerows(zip(payload['decisionTime'],payload['truth'],
                           [algorithm]*len(payload['truth']),[topology]*len(payload['truth']),
                           row['score'],row['state'],[threshold]*len(payload['truth'])))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        project=deepcopy(run.project)
        project.update(threshold=threshold,selectedLink=selected_link)
        z.writestr('project.json',json.dumps(project,ensure_ascii=False,indent=2))
        z.writestr('decisions.csv','\ufeff'+output.getvalue())
        z.writestr('metadata.json',json.dumps({**payload['metadata'],'threshold':threshold,
                      'selectedLink':selected_link,'seconds':run.seconds},ensure_ascii=False,indent=2))
        z.writestr('metrics.json',json.dumps({a:{'links':{l:r['metrics'] for l,r in v['links'].items()},
                       'fusion':v['fused']['metrics']} for a,v in payload['series'].items()},ensure_ascii=False,indent=2))
        buf=BytesIO()
        arrays={f'{l}_{k}':v for l,o in run.observations.items() for k,v in o.items()}
        arrays.update({'subcarriers':run.channel.k,'pose_time':run.visual['time'],
                       'body_points':run.visual['pose']['points'],'truth':run.truth,
                       'decision_time':run.data['time']})
        np.savez_compressed(buf,**arrays)
        z.writestr('observations.npz',buf.getvalue())

def export_slides(run,path,screenshot,threshold,selected_link):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from PIL import Image
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    def slide(title):
        s=prs.slides.add_slide(prs.slide_layouts[6])
        s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string('FAF9F5')
        t=s.shapes.add_textbox(Inches(.4),Inches(.22),Inches(12.5),Inches(.5)).text_frame
        p=t.paragraphs[0];p.text=title;p.font.name='Microsoft YaHei';p.font.size=Pt(26)
        p.font.color.rgb=RGBColor.from_string('292824')
        return s
    s=slide('Wi-Fi CSI 三维人体运动感知')
    with Image.open(screenshot) as image:
        ratio=image.width/image.height
    width=min(12.5,6.15*ratio);height=width/ratio
    s.shapes.add_picture(str(screenshot),Inches((13.333-width)/2),Inches(.95+(6.15-height)/2),width=Inches(width),height=Inches(height))
    s=slide('同一份 CSI · 原始与优化算法对比')
    results=run.decisions(threshold)
    metrics=[]
    for algorithm,label in [('original','原始'),('optimized','优化')]:
        for topology,title in [(selected_link,f'单链路 {selected_link}'),('fused','融合')]:
            m=results[algorithm]['fused' if topology=='fused' else 'links']
            if topology!='fused':m=m[topology]
            metrics.append((f'{label} · {title}',m['metrics']))
    data=CategoryChartData();data.categories=['F1','精确率','召回率','误报率']
    for name,m in metrics:
        data.add_series(name,[m[k] for k in ['f1','precision','recall','fpr']])
    chart=s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,Inches(.45),Inches(1),Inches(12.3),Inches(5.5),data).chart
    chart.has_legend=True;chart.legend.position=XL_LEGEND_POSITION.BOTTOM
    chart.value_axis.minimum_scale=0;chart.value_axis.maximum_scale=1
    # OOXML axis identifiers are unsigned integers.
    axis_elements=chart._chartSpace.xpath('.//c:axId | .//c:crossAx')
    axis_ids={value:str(i+1001) for i,value in enumerate(dict.fromkeys(x.get('val') for x in axis_elements))}
    for element in axis_elements:element.set('val',axis_ids[element.get('val')])
    for font in (chart.legend.font,chart.category_axis.tick_labels.font,chart.value_axis.tick_labels.font):
        font.name='Microsoft YaHei';font.size=Pt(17)
    colors=['B4B9C2','8896AA','63768C','344960']
    for series,color in zip(chart.series,colors):
        series.format.fill.solid();series.format.fill.fore_color.rgb=RGBColor.from_string(color)
    rows=[['房间 / m',' × '.join(str(x) for x in run.project['room'])],
          ['人员','有人' if run.project['present'] else '无人'],
          ['链路','、'.join(run.names)],['采样率',f"{run.project['radio']['rate']} Hz"],
          ['信噪比',f"{run.project['radio']['snr']} dB"],['随机种子',str(run.project['seed'])],
          ['检测阈值',f'{threshold:.2f}']]
    action_names={'still':'停留','walk':'走动','wave':'挥手'}
    if run.project['present']:
        rows.extend([[f"{x['start']:.1f}–{x['end']:.1f} s",action_names[x['kind']]] for x in segments(run.project)])
    for start in range(0,len(rows),13):
        part=rows[start:start+13]
        s=slide('场景与动作配置'+(' · 续' if start else ''))
        table=s.shapes.add_table(len(part)+1,2,Inches(.55),Inches(1),Inches(12.2),Inches(5.9)).table
        table.cell(0,0).text='项目';table.cell(0,1).text='配置'
        for i,row in enumerate(part,1):
            for j,text in enumerate(row):table.cell(i,j).text=text
        for i,row in enumerate(table.rows):
            for cell in row.cells:
                cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string('F0EEE6' if i%2==0 else 'FAF9F5')
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.name='Microsoft YaHei';paragraph.font.size=Pt(15)
                    paragraph.font.color.rgb=RGBColor.from_string('292824')
    prs.save(path)

