"""Validated inference for a stored song or exploratory local audio."""
import argparse,json
from .artifacts import load_run


def main():
    p=argparse.ArgumentParser(description=__doc__);source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--track-id');source.add_argument('--audio');p.add_argument('--start',type=float);p.add_argument('--run-id')
    args=p.parse_args();a=load_run(args.run_id)
    if args.audio:
        from .audio import extract_record
        X,segment,meta=extract_record(args.audio,args.start,a.bundle['extractor_version'])
        prediction,score,semantics=a.predict(X,a.bundle['extractor_version']);info={'experimental_audio':True,**meta}
    else:
        row=a.data.loc[a.data.track_id==args.track_id]
        if row.empty:p.error('Unknown track ID')
        X=row[a.bundle['features']];prediction,score,semantics=a.predict(X)
        held_out=args.track_id not in a.bundle['train_track_ids']
        info={'track_id':args.track_id,'title':row.title.iloc[0],'artist':row.artist.iloc[0],
              'actual_label':int(row.label.iloc[0]),'held_out':held_out,
              'evaluation_status':'historical_test_example' if held_out else 'training_example'}
    print(json.dumps({**info,'run_id':a.summary['run_id'],'model':a.bundle['model_name'],'prediction':int(prediction[0]),
       'label':a.bundle['labels'][str(int(prediction[0]))],'score':float(score[0]),'score_semantics':semantics,
       'evidence_status':a.summary['evidence_status']},indent=2))

if __name__=='__main__':main()
