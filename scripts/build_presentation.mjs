// Run with the documented Codex artifact runtime; see docs/REPRODUCE_V2.md.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const root=process.cwd();
const workspaceDir=process.env.CHORUS_WORKSPACE||root;
const build=process.env.CHORUS_BUILD_DIR||path.join(root,'work/slides-v2');
const skill=process.env.SKILL_DIR;
if(!skill)throw new Error('Set SKILL_DIR to the installed presentation skill');
const input=JSON.parse(await fs.readFile(path.join(root,'docs/v2/presentation_input.json'),'utf8'));
const s=input.summary,h=input.historical,b=input.baseline;
const digest=createHash('sha256').update(await fs.readFile(path.join(root,'results/v2',s.run_id,'manifest.json'))).digest('hex');
if(digest!==input.manifest_sha256)throw new Error('Presentation source run changed');
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const family=resolvePresentationFont();const P=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#172D37',teal:'#127C80',muted:'#52666E',pale:'#EFF5F5',amber:'#C77D22'};
const pct=x=>(100*x).toFixed(1)+'%';const paper='https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf';
const source='https://github.com/AntoniosMalak/Predicting-Hit-Songs-Using-Repeated-Chorus';
function text(sl,value,x,y,w,hh,size=28,color=C.ink,bold=false){const shape=sl.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:hh},fill:'none',line:{fill:'none',width:0}});shape.text=value;shape.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};return shape;}
function slide(title,notes){const sl=P.slides.add();sl.background.fill='#FFFFFF';if(title)text(sl,title,64,42,1152,96,43,C.ink,true);text(sl,String(P.slides.items.length),1170,668,44,28,18,C.muted);sl.speakerNotes.textFrame.setText(notes);return sl;}
function table(sl,rows,x,y,w,hh,widths){const t=sl.tables.add({rows:rows.length,columns:rows[0].length,left:x,top:y,width:w,height:hh,columnWidths:widths,values:rows});t.borders.assign({style:'solid',fill:'#D5E1E2',width:.7});t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:rows[0].length}).assign({fill:'#FFFFFF',textStyle:{typeface:family,fontSize:25,color:C.ink},margins:{left:12,right:10,top:8,bottom:8}});t.cells.block({row:0,column:0,rowCount:1,columnCount:rows[0].length}).assign({fill:C.pale,textStyle:{typeface:family,fontSize:25,color:C.ink,bold:true}});return t;}
{
const sl=slide('',`Topic supplied through Eric Liu's reference report: ${paper}. This October update preserves the original task and adds reproducible audit and development experiments. Both students should understand the implementation and evidence. Materials prepared with Codex assistance.`);
text(sl,'Predicting Hit Songs\nUsing Repeated Chorus',64,120,1152,190,59,C.ink,true);
text(sl,'Audit and controlled experiments',68,340,1100,56,34,C.teal);
text(sl,'Dhanush Sai Suprapadha   PES2UG24CS154\nDeepthi V   PES2UG24CS150',68,458,1090,86,28);
text(sl,'UE24CS352A   Machine Learning   October 2026',68,590,1090,42,23,C.muted);
}
{
const sl=slide('Research task and available data',`The task is unchanged: year-end hit versus other chart song. Both classes can have charted. Counts from validated legacy CSV and results/audit/v1_independent_verification.json. The public reproduction differs from the supplied paper. Source: ${source}, commit 838e76f. Label collection: its CollectData.ipynb. No new labels or source recordings were created.`);
text(sl,'Can 15-second chorus features distinguish\na year-end hit from another chart song?',64,156,1140,110,35,C.teal,true);
table(sl,[['Available data','Count'],['Songs / artist-name groups','751 / 71'],['Raw audio measurements',String(s.feature_count)],['Year-end hits / other chart songs','366 / 385']],64,305,1152,244,[820,332]);
text(sl,'Both classes contain charting songs. Original recordings are unavailable.',64,597,1135,60,26,C.muted);
}
{
const sl=slide('Audit results and engineering repairs',`Evidence: results/baseline/v1_manifest.json, results/audit/v1_reproduction_comparison.json and test logs. The complete V1 rerun matched all nine model metrics and 154 predictions. F06/F07 repairs identify actual source bytes and reject false upstream provenance. Shared artifact loader validates run, schema, label and hash identities. These engineering repairs do not by themselves improve prediction.`);
text(sl,'The original results reproduced exactly',64,161,1140,65,36,C.teal,true);
text(sl,'All nine models matched their recorded scores.\nAll 154 historical predictions matched.',64,240,1140,108,29);
text(sl,'Safer experiments and consistent outputs',64,399,1140,60,34,C.ink,true);
text(sl,'Actual-file hashes and explicit source provenance\nSeparate immutable runs with retained trial records\nOne validated model contract for the app, CLI and reports',64,480,1130,154,28);
}
{
const sl=slide('Development and historical evaluation',`V2 development uses the original training membership only. Evidence: configs/v2_thorough.json and results/v2/v2_nested_001/outer_folds.json. Five outer folds assess the procedure, with three inner grouped folds selecting across the whole candidate list. Original historical membership is fixed. Artist strings are normalized, not fully resolved canonical performers. No fresh test exists.`);
table(sl,[['Partition','Songs','Groups'],['Development','597','52'],['Historical benchmark','154','19']],64,154,1152,198,[720,216,216]);
text(sl,'Five outer folds assess the selection procedure',64,396,1140,62,34,C.teal,true);
text(sl,'Each outer training set uses three inner folds to choose a model.\nPreprocessing learns only from its fitting rows.\nThe final candidate then fits all development songs.',64,478,1140,153,28);
}
{
const sl=slide('The predeclared candidate search',`There are 35 declared configurations, with fixed seeds 42/43/44. Sources: configs/v2_quick.json, configs/v2_thorough.json and immutable ablation tables. The same candidate list preceded both runs. Original V1 candidate families remain as controls; Extra Trees is new. Mutual-information selection is seeded and learned within each fold. No adaptive search of favorable seeds occurred.`);
table(sl,[['Representation','Declared alternatives'],['Original feature axes','No reduction'],['PCA','30, 50, 100 components; 90%, 95% variance'],['Supervised selection','ANOVA or seeded mutual information'],['Feature-family subset','MFCC, energy and spectral features']],64,157,1152,307,[395,757]);
text(sl,'35 configurations with fixed seeds and compute budgets',64,510,1140,53,31,C.teal,true);
text(sl,'Original classifiers and dummy baseline remain. Extra Trees adds one family.',64,587,1140,63,26,C.muted);
}
{
const metrics=['accuracy','balanced_accuracy','precision','recall','f1'];const sl=slide('Nested evaluation and the 75% target',`Source: results/v2/v2_nested_001/summary.json and predictions.csv. Values describe the selection procedure's outer-fold predictions on 597 songs across 52 artist groups. The final development candidate is ${s.model_name}, logistic regression with 50 MI-selected features. Balanced accuracy interval: ${s.ci95.balanced_accuracy.map(pct).join(' to ')}. Bootstrap resamples whole artist groups and conditions on recorded predictions. No fresh-test confirmation. Strict checks use unrounded values.`);
const ch=sl.charts.add('bar',{position:{left:64,top:162,width:1152,height:373},categories:['Accuracy','Balanced accuracy','Precision','Recall','F1'],series:[{name:'Nested development',values:metrics.map(k=>Number((100*s.metrics[k]).toFixed(1))),fill:C.teal},{name:'75% boundary',values:metrics.map(()=>75),fill:'#D5E1E2'}],barOptions:{direction:'column',grouping:'clustered',gapWidth:85},hasLegend:true,xAxis:{textStyle:{fontSize:23}},yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:22},title:'Percent'},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:23},numberFormat:'0.0'}});applyPresentationChartFont(ch,{fontFamily:family});
text(sl,`Balanced accuracy interval: ${s.ci95.balanced_accuracy.map(pct).join(' to ')}`,64,563,1135,51,30,C.ink,true);
text(sl,'All five metrics must exceed 75%. This evaluation does not meet that target.',64,623,1140,43,24,C.muted);
}
{
const keys=['accuracy','balanced_accuracy','precision','recall','f1'];const sl=slide('Matched historical comparison',`Both columns evaluate the same 154 songs from 19 artist-string groups. V2 was frozen before this deliberate historical invocation. Sources: results/v2/v1_baseline/summary.json and results/evaluations/v2_nested_001_historical/summary.json. V1 selected polynomial SVM; V2 final candidate logistic regression plus 50 MI features. This is a historical comparison because these test labels and results were already inspected. The paired interval uses identical resampled groups for both predictions. No promotion based on historical scores.`);
table(sl,[['Metric','V1 original','V2 candidate'],...keys.map(k=>[k.replaceAll('_',' '),pct(b.metrics[k]),pct(h.metrics[k])])],64,155,1152,331,[650,251,251]);
const delta=h.paired_vs_v1;text(sl,`Balanced accuracy change: +${(delta.balanced_accuracy_difference*100).toFixed(1)} percentage points`,64,525,1140,50,30,C.teal,true);
text(sl,`Paired interval: ${(delta.ci95[0]*100).toFixed(1)} to +${(delta.ci95[1]*100).toFixed(1)} points. It includes zero.`,64,590,1140,68,27,C.muted);
}
{
const curves=JSON.parse(await fs.readFile(path.join(root,'docs/v2/learning_curve_chart.json'),'utf8'));const sl=slide('Training and validation gaps',`The curves describe the final selected configuration as a retrospective diagnostic. They are not fresh evaluation or a new estimate of the selection procedure. Source: results/v2/v2_nested_001/learning_curves.csv. Fitting subsets use deterministic whole-artist selections inside the same three grouped development folds. Wider group coverage improves measured validation in these points, but this does not prove a causal explanation or forecast 75% performance.`);
const ch=sl.charts.add('line',{position:{left:64,top:160,width:1152,height:379},categories:curves.map(r=>r.groups.toFixed(1)),series:[{name:'Training',values:curves.map(r=>Number((100*r.train).toFixed(1))),line:{fill:C.amber,width:3}},{name:'Validation',values:curves.map(r=>Number((100*r.validation).toFixed(1))),line:{fill:C.teal,width:3}}],hasLegend:true,xAxis:{title:'Mean fitting artist groups',textStyle:{fontSize:24}},yAxis:{min:0,max:100,majorUnit:25,title:'Balanced accuracy (%)',textStyle:{fontSize:23}}});applyPresentationChartFont(ch,{fontFamily:family});
text(sl,'A training/validation gap remains',64,573,1140,48,33,C.teal,true);
text(sl,'These diagnostics do not establish a single cause or guarantee a future score.',64,630,1140,36,24,C.muted);
}
{
const sl=slide('Working demo and blocked audio experiments',`App: app.py uses chorus_hit/artifacts.py for run/hash/schema checks. Default active run is v1_baseline; V2 is explicitly a research option. Ingestion/shared extraction/MERT cache interfaces exist and have synthetic software tests. There are zero eligible original recordings in data/recording_audit_v1.json. MERT execution and ablations have not run. Required: permitted real audio, recording/performer identities, verified labels/windows, reviewed pinned local model weights and compatible optional dependencies. Source model card: https://huggingface.co/m-a-p/MERT-v1-95M.`);
text(sl,'Demo',64,165,1100,53,34,C.teal,true);
text(sl,'Choose a historical song and compare its prediction.\nInspect the run identity, errors and evaluation status.\nThe original model remains the active default.',64,235,1130,152,28);
text(sl,'Inputs needed for the next audio study',64,430,1140,53,34,C.ink,true);
text(sl,'Permitted recordings and verified recording/label identities\nOne extraction policy for both training audio and uploads\nA genuinely fresh evaluation collection',64,500,1130,145,28);
}
{
const sl=slide('Conclusion and project evidence',`The 75% target was not reached. Nested BA ${pct(s.metrics.balanced_accuracy)} with interval including 50%; historical BA ${pct(h.metrics.balanced_accuracy)} with paired change interval including zero. No fresh confirmation or real embedding results. Source paper: ${paper}. Feature dataset: ${source}, Apache-2.0, commit 838e76f. Method: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html. Local branch audit/chorus-v2 contains the complete evidence and preserves original authorship/attribution. It has not been pushed.`);
text(sl,'The measured results remain below the target',64,165,1140,100,40,C.teal,true);
text(sl,`Nested balanced accuracy: ${pct(s.metrics.balanced_accuracy)}\nHistorical balanced accuracy: ${pct(h.metrics.balanced_accuracy)}\nFresh confirmation: unavailable`,64,310,1140,144,31);
text(sl,'Evidence included with the project',64,500,1130,50,32,C.ink,true);
text(sl,'Baseline snapshot, source audit, folds and every trial\nPredictions, uncertainty, executable notebook and study guide',64,570,1130,90,27);
}
await fs.mkdir(build,{recursive:true});const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(P)).save(candidate);
const result=await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath:path.join(root,'docs/v2/Project_Presentation.pptx'),pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,4,5,7].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[2,4,5,7],requiredNativeChartOwnerSlides:[6,8],materializeLiteralChartWorkbooks:true,explicitTotalSlideCount:10,fontPolicy:{basis:'design',families:[family]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify(result));
for(let i=0;i<P.slides.items.length;i++){const png=await P.export({slide:P.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));}
console.log('Built and rendered 10 slides');
