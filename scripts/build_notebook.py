"""Create and execute a new walkthrough from a validated immutable run."""
import argparse,os,json,sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chorus_hit.config import ROOT
from chorus_hit.reporting import report_context


def build(run_id,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Preserve the existing notebook; choose a new output')
    context=report_context(run_id);t=context['text'];s=context['summary'];md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    nb=nbf.v4.new_notebook();nb.cells=[
        md('# Predicting Hit Songs Using Repeated Chorus\n\nDhanush Sai Suprapadha - PES2UG24CS154\n\nDeepthi V - PES2UG24CS150\n\nThis walkthrough reads validated experiment artifacts. It never retrains or tunes on historical labels.'),
        code(f'''from pathlib import Path
import sys,json
import pandas as pd
from IPython.display import display,Markdown
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
from chorus_hit.artifacts import load_run
from chorus_hit.evaluation import classification_metrics
from chorus_hit.reporting import summary_text
RUN_ID={run_id!r}
assets=load_run(RUN_ID)
summary=assets.summary
display(pd.Series({{'run_id':RUN_ID,'model':summary['model_name'],'raw_features':summary['feature_count'],'evaluation':summary['evaluation_status'],'source_hash':assets.manifest['dataset']['source_sha256']}}))'''),
        md('## 1. Dataset and task\n\nClass 1 means a source year-end Billboard hit. Class 0 contains other chart songs. Both classes can have charted. Source recordings are unavailable. No artist/title/path or chart-position metadata enters the model.'),
        code("display(assets.data[['track_id','artist','title','label']].head())\ndisplay(assets.data.label.value_counts().rename('songs'))\ndisplay(pd.Series(summary['counts']))"),
        md('## 2. Grouped model selection\n\nAll learned preprocessing fits inside the training folds. V2 uses a fixed candidate list and separate outer folds to assess the selection procedure. The historical partition never tunes settings. Artist strings are the declared identity level; aliases and collaborations remain unresolved.'),
        code("display(assets.bundle['pipeline'])\nconfig_path=assets.path/'config.json'\nif config_path.exists():\n    config=json.loads(config_path.read_text())\n    display(pd.DataFrame(config['candidates']))\n    display(pd.read_csv(assets.path/'final_development_ranking.csv').sort_values('mean_balanced_accuracy',ascending=False))"),
        md('## 3. Recompute the reported evaluation\n\nNested rows come from models fitted without their outer-fold groups. The final saved candidate is a separate fit on all development rows. Comparing its training predictions to these nested predictions would be the wrong check. Different outer candidates can have incomparable score scales, so pooled nested ROC-AUC is omitted.'),
        code("predictions=assets.predictions\nscores=None if summary['evaluation_status']=='nested_development' else predictions.score\nrecomputed=classification_metrics(predictions.label,predictions.prediction,scores)\nfor key in ['accuracy','balanced_accuracy','precision','recall','f1']:\n    assert abs(recomputed[key]-summary['metrics'][key])<1e-12\ndisplay(pd.Series(recomputed))\ndisplay(pd.DataFrame(summary['target_status']['metrics']).T)\ndisplay(pd.DataFrame(summary['ci95'],index=['lower','upper']).T)"),
        md('## 4. Diagnostics\n\nTraining/validation gaps and learning curves describe this selected configuration. They are retrospective diagnostics, not a fresh evaluation or proof of a single performance bottleneck.'),
        code("curve=assets.path/'learning_curves.csv'\nif curve.exists():\n    display(pd.read_csv(curve))\ndisplay(predictions.head(10))\nprint('All errors remain in the record:',int((predictions.label!=predictions.prediction).sum()))"),
        md('## 5. Interpretation\n\n'+t['result']+'\n\n'+t['uncertainty']+'\n\n'+t['conclusion']),
        code("text=summary_text(summary)\ndisplay(Markdown('**'+text['title']+'**\\n\\n'+text['method']+'\\n\\n'+text['result']+'\\n\\n'+text['uncertainty']+'\\n\\n'+text['conclusion']))"),
        md('## 6. Demonstration and missing inputs\n\nStart the Streamlit app and inspect its run identity. The original demo remains active; the V2 candidate is a research option. Audio uploads use the versioned extractor, but waveform parity remains unverified.\n\nNew experiments require permitted recordings, verified label/recording identities and a new locked evaluation set. Embedding interfaces have only software-fixture tests, with no measured real-song embedding results.\n\nSources: [reference paper](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf), [public feature dataset](https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus), and local immutable run manifests. Both students should run and understand the project before presenting it.')]
    nb.metadata={'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    out.parent.mkdir(parents=True,exist_ok=True);nbf.write(nb,out);print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.run_id,a.out)
