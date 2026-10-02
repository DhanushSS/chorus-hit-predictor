"""Build new one/two-page PDFs from a validated run without replacing V1."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json,sys
from xml.sax.saxutils import escape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chorus_hit.config import ROOT
from chorus_hit.reporting import report_context
from chorus_hit.artifacts import load_run
from chorus_hit.audit import atomic_json,sha256_file
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from pypdf import PdfReader

STYLES=getSampleStyleSheet()
STYLES.add(ParagraphStyle(name='ProjectTitle',fontName='Helvetica-Bold',fontSize=20,leading=24,textColor=colors.HexColor('#17323C'),spaceAfter=10))
STYLES.add(ParagraphStyle(name='Section',fontName='Helvetica-Bold',fontSize=11.5,leading=15,spaceBefore=9,spaceAfter=5,textColor=colors.HexColor('#127C80')))
STYLES.add(ParagraphStyle(name='Copy',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=7))
STYLES.add(ParagraphStyle(name='Note',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=5,textColor=colors.HexColor('#52666E')))


def p(text,style='Copy'):return Paragraph(text,STYLES[style])

def table(rows,widths):
    t=Table([[p(escape(str(x)),'Note') for x in row] for row in rows],colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EFF5F5')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#127C80')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    return t


def render(path,story,date):
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#52666E'))
        canvas.drawString(42,24,f'Chorus Hit Predictor | Audit and research update | {date}')
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=40,
        title='Predicting Hit Songs Using Repeated Chorus',author='Dhanush Sai Suprapadha; Deepthi V').build(story,onFirstPage=footer,onLaterPages=footer)


def build(run_id,out,date,historical=None):
    out=Path(out)
    if out.exists():raise FileExistsError('Report directory exists; preserve exports and choose a new directory')
    c=report_context(run_id,historical);s=c['summary'];t=c['text'];d=c['data'];h=c['historical'];baseline=load_run('v1_baseline').summary
    header=[p('Predicting Hit Songs Using Repeated Chorus','ProjectTitle'),p('Dhanush Sai Suprapadha - PES2UG24CS154<br/>Deepthi V - PES2UG24CS150<br/>UE24CS352A - Machine Learning','Note')]
    counts=s['counts'];features=s['feature_count'];groups=d.artist.nunique();labels=d.label.value_counts()
    intro=f'Can a 15-second chorus distinguish a source year-end Billboard hit (class 1) from another chart song (class 0)? Both classes can have charted. The public feature-table task differs from the supplied reference paper. This update preserves its labels and original historical-test membership.'
    dataset=f'The audited table contains {len(d)} songs, {groups} provided artist names and {features} raw audio features. Class counts are {int(labels.get(0,0))} other-chart songs and {int(labels.get(1,0))} year-end hits. Original recordings are unavailable. Names are grouping metadata, never model inputs.'
    config_path=ROOT/'results/v2'/run_id/'config.json'
    config=json.loads(config_path.read_text()) if config_path.is_file() else None
    if config:
        protocol=f"The original development partition has {counts['train_songs']} songs from {counts['train_artists']} artist-name groups. We compare {len(config['candidates'])} predeclared settings with fixed seeds. {config['outer_folds']} outer folds each use {config['inner_folds']} inner folds for nested selection. The full development partition then selects the final candidate using inner-fold balanced accuracy. Historical labels never tune it."
    else:protocol='The preserved V1 model uses its original grouped development cross-validation and original historical-test evaluation.'
    if config and config['mode']=='quick':
        protocol=f"The original development partition has {counts['train_songs']} songs from {counts['train_artists']} artist-name groups. {config['inner_folds']} grouped folds compare {len(config['candidates'])} declared configurations. These are tuning scores, with no outer evaluation. Historical labels never select the candidate."
    transforms='All learned imputation, filtering, scaling, PCA and supervised selection fit inside training folds. The bounded search retains the original classifier controls, adds Extra Trees, and compares unreduced features, PCA, ANOVA selection, seeded mutual information and a declared feature-family subset. Native estimator thresholds remain fixed.' if config else t['method']
    changes='The audit reproduced all original recorded metrics. Repairs bind audits to actual input bytes, require honest alternate-source provenance, preserve stable identities and isolate experiment outputs. One validated loader checks run IDs, data/model hashes, ordered schemas, labels and score contracts. Completed runs preserve every fold, warning and failed trial.'
    limitations='Artist strings do not fully resolve aliases or guest performers. Three sampled positive labels have secondary corroboration, while primary chart/recording verification remains incomplete. No labels changed. Waveform-feature parity remains unverified. The new audio and embedding interfaces have software tests, but no real-song embedding experiment has run.'
    target_rows=[['Metric',s['evaluation_status'].replace('_',' ').title(),'Strict >75%']]+[[k.replace('_',' ').title(),f"{v['value']:.1%}",'Pass' if v['passed'] else 'Fail'] for k,v in s['target_status']['metrics'].items()]
    matched=[['Historical benchmark','V1 original','V2 candidate']]
    if h:
        for k in ['accuracy','balanced_accuracy','precision','recall','f1']:
            matched.append([k.replace('_',' ').title(),f"{baseline['metrics'][k]:.1%}",f"{h['metrics'][k]:.1%}"])
    result=f"Run <b>{escape(run_id)}</b>. Final development candidate: <b>{escape(s['model_name'])}</b>. {escape(t['result'])} Full-development mean tuning balanced accuracy: {s['tuning_balanced_accuracy']:.1%}. Nested predictions assess the selection procedure, rather than the single final refit."
    if s['evaluation_status']!='nested_development':result=f"Run <b>{escape(run_id)}</b>. {escape(t['result'])} {escape(t['method'])}"
    historical_text='No historical comparison requested for this export.'
    if h:
        delta=h['paired_vs_v1'];delta_interpretation='This interval includes zero.' if delta['ci95'][0]<=0<=delta['ci95'][1] else 'This interval excludes zero in this historical comparison.'
        historical_text=f"The frozen candidate scored {h['metrics']['balanced_accuracy']:.1%} balanced accuracy on the same {h['metrics']['n']} historical songs from {h['groups']} groups. The paired change from V1 is {100*delta['balanced_accuracy_difference']:+.1f} percentage points, with a group-bootstrap interval of {100*delta['ci95'][0]:+.1f} to {100*delta['ci95'][1]:+.1f} points. {delta_interpretation} These already known records provide a historical comparison, not fresh confirmation."
    references='[1] Eric Liu, CS229, 2021: <link href="https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf" color="#127C80">reference report</link>.<br/>[2] <link href="https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus" color="#127C80">Antonios Malak feature dataset</link>, pinned commit 838e76f, Apache-2.0.<br/>[3] <link href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html" color="#127C80">Scikit-learn nested cross-validation</link>. Local run manifests and predictions supply all reported metrics.'
    story=header+[p('Question and dataset','Section'),p(intro),p(dataset),p('Controlled evaluation','Section'),p(protocol),p(transforms),
        p('Audit and implementation','Section'),p(changes),p('Data and audio limits','Section'),p(limitations),
        p('Reproducibility','Section'),p(f'Run identity: {escape(run_id)}. Configuration, folds, trials, predictions, uncertainty, package versions and model hashes are preserved in its immutable directory. The baseline archive and exact executed CSV-run source snapshot are retained. Implementation and materials use Codex assistance.'),
        PageBreak(),p('Measured results and interpretation','ProjectTitle'),p(result),table(target_rows,[195,155,161]),p(t['uncertainty']),p(t['conclusion']),
        p('Matched historical comparison','Section'),table(matched,[195,155,161]) if h else Spacer(1,1),p(historical_text),
        p('Demonstration and next inputs','Section'),p('The demo keeps the original model active and labels the V2 option as a research candidate. It shows run identity, evaluation status and strict target checks. Uploaded audio stays exploratory. Next work needs permitted recordings, verified chart/recording identities and a genuinely fresh, locked evaluation collection. A new feature or population study requires its own baseline.'),
        p('References','Section'),p(references,'Note')]
    short=header+[p('Question and evidence','Section'),p(intro),p(dataset),p('Method','Section'),p(protocol),p(t['method']),p('Measured outcome','Section'),
        p(result),table(target_rows,[195,155,161]),p(t['uncertainty']),p(t['conclusion']),p('Historical comparison and limits','Section'),p(historical_text),
        p('The baseline demo remains active. New audio predictions are exploratory. No fresh test or real-song embedding results exist. Full data provenance, trial records and the exact software checks accompany the project.','Note'),p(references,'Note')]
    out.mkdir(parents=True);render(out/'Project_Report_2_Pages.pdf',story,date);render(out/'Project_Summary_1_Page.pdf',short,date)
    for file,n in [('Project_Report_2_Pages.pdf',2),('Project_Summary_1_Page.pdf',1)]:
        actual=len(PdfReader(out/file).pages)
        if actual!=n:raise ValueError(f'Layout requires repair: {file} has {actual} pages, expected {n}')
    atomic_json(out/'report_source.json',{'run_id':run_id,'run_manifest_sha256':sha256_file(ROOT/'results/v2'/run_id/'manifest.json'),
        'date':date,'summary':s,'historical':h,'files':{p.name:sha256_file(p) for p in out.glob('*.pdf')}})
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--historical',type=Path);p.add_argument('--date',default=datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat())
    a=p.parse_args();build(a.run_id,a.out,a.date,a.historical)

if __name__=='__main__':main()
