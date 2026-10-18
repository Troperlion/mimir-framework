"""Constrained, reproducible whole-model benchmark used by the Colab notebook."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import urllib.request

MNIST_URL='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
MNIST_SHA256='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_data(directory,hidden_sizes,seed,quick=False,archive=None):
    import numpy as np
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    archive=Path(archive) if archive else directory/'mnist.npz'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        temporary=archive.with_suffix('.download')
        with urllib.request.urlopen(MNIST_URL,timeout=120) as source,temporary.open('wb') as target:
            import shutil
            shutil.copyfileobj(source,target)
        temporary.replace(archive)
    if sha256(archive)!=MNIST_SHA256:raise RuntimeError('MNIST SHA256 mismatch; remove the corrupt archive')
    with np.load(archive,allow_pickle=False) as data:
        order=np.random.default_rng(seed).permutation(len(data['x_train']))
        splits={'train':(data['x_train'][order],data['y_train'][order]),
                'valid':(data['x_test'][:2000],data['y_test'][:2000]),
                'test':(data['x_test'][2000:],data['y_test'][2000:])}
        for split,(x,y) in splits.items():
            count={'train':10000,'valid':500,'test':1500}[split] if quick else len(x)
            x=np.asarray(x[:count],np.uint8);y=np.asarray(y[:count],np.uint8)
            if x.shape[1:]!=(28,28) or np.any(y>=10):raise RuntimeError('Invalid MNIST dataset')
            x.tofile(directory/f'{split}.images.u8');y.tofile(directory/f'{split}.labels.u8')
    for hidden in sorted(set(hidden_sizes)):
        rng=np.random.default_rng(seed+hidden)
        params=[(rng.standard_normal((hidden,784))*np.sqrt(2/784)).astype('<f4'),np.zeros(hidden,'<f4'),
                (rng.standard_normal((10,hidden))/np.sqrt(hidden)).astype('<f4'),np.zeros(10,'<f4')]
        for i,p in enumerate(params):p.tofile(directory/f'weights-{hidden}-{i}.f32')
    provenance=dict(source=MNIST_URL,source_sha256=MNIST_SHA256,seed=seed,
        split='original train shuffled once; original test[0:2000] validation; test[2000:] held out',
        quick=quick,normalization='uint8 -> float32 / 255; one-hot labels; no augmentation',
        files={p.name:sha256(p) for p in sorted(directory.iterdir()) if p.suffix in {'.u8','.f32'}})
    (directory/'dataset-manifest.json').write_text(json.dumps(provenance,indent=2))
    return provenance


def first_step_reference(config):
    import numpy as np
    d=Path(config['data_dir']);h=config['hidden'];b=config['batch'];lr=config['lr']
    shapes=[(h,784),(h,),(10,h),(10,)];p=[np.fromfile(d/f'weights-{h}-{i}.f32',dtype='<f4').reshape(shape).astype(np.float64) for i,shape in enumerate(shapes)]
    images=np.fromfile(d/'train.images.u8',dtype=np.uint8).reshape(-1,784)
    labels=np.fromfile(d/'train.labels.u8',dtype=np.uint8);index=np.arange(b)%len(images)
    # Match the float32 preprocessing, then calculate the reference in float64.
    x=(images[index].astype(np.float32)/np.float32(255)).astype(np.float64)
    target=np.eye(10)[labels[index]]
    z=x@p[0].T+p[1];activation=np.maximum(z,0);out=activation@p[2].T+p[3]
    delta=2*(out-target)/(b*10);dh=(delta@p[2])*(z>0)
    grads=[dh.T@x,dh.sum(0),delta.T@activation,delta.sum(0)]
    updated=np.concatenate([(v-lr*g).ravel() for v,g in zip(p,grads)])
    return dict(prediction=out.ravel(),loss=np.mean((out-target)**2),probe=updated[config['probe_indices']])


def process_tree_memory(root):
    """Sample RSS and swap for the child AND its workers (not the notebook)."""
    rows={}
    for status in Path('/proc').glob('[0-9]*/status'):
        try:
            fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.read_text().splitlines() if ':' in line}
            pid=int(status.parent.name)
            rows[pid]=(int(fields['PPid']),int(fields.get('VmRSS','0 kB').split()[0])*1024,
                       int(fields.get('VmSwap','0 kB').split()[0])*1024)
        except (OSError,KeyError,ValueError):continue
    descendants={root}
    while True:
        expanded=descendants|{pid for pid,(parent,_,_) in rows.items() if parent in descendants}
        if expanded==descendants:break
        descendants=expanded
    return sum(rows[p][1] for p in descendants if p in rows),sum(rows[p][2] for p in descendants if p in rows)


def constrained_run(command,environment,log_path,timeout_s,memory_mib,cwd=None,poll_s=.05):
    limit=int(memory_mib*1024**2);started=time.monotonic();reason=None;peak_rss=peak_total=0
    with Path(log_path).open('w') as log:
        process=subprocess.Popen(list(map(str,command)),cwd=cwd,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                rss,swap=process_tree_memory(process.pid);peak_rss=max(peak_rss,rss);peak_total=max(peak_total,rss+swap)
                elapsed=time.monotonic()-started
                if rss+swap>limit:reason='memory_limit'
                elif elapsed>timeout_s:reason='time_limit'
                if reason:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait();break
                if process.poll() is not None:break
                time.sleep(poll_s)
        except BaseException:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait();raise
        # Also clean up any lingering workers after normal exit.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    return dict(process_wall_s=time.monotonic()-started,returncode=process.returncode,
        budget_status=reason or ('completed' if process.returncode==0 else 'process_error'),
        peak_tree_rss_mib=peak_rss/1024**2,peak_tree_rss_swap_mib=peak_total/1024**2,
        host_memory_limit_mib=memory_mib,process_time_limit_s=timeout_s,memory_poll_s=poll_s)


def validate_result(result,config,gold,min_accuracy):
    import numpy as np
    if result.get('status')!='measured':raise RuntimeError(result.get('error','Measurement missing')+'\n'+result.get('traceback',''))
    if result.get('case')!=config['case']:raise RuntimeError('Wrong result identity')
    if result.get('framework')=='mimir':
        options=config.get('mimir',{})
        for phase in ('training_plan','inference_plan'):
            plan=result.get(phase,{})
            if not plan.get('planned') or plan.get('mode')!=options.get('planner_mode','cost'):
                raise RuntimeError('Mimir planner not active: '+phase)
            if plan.get('fusion')!=options.get('fusion',True):raise RuntimeError('Mimir fusion setting differs from protocol')

    for key,expected in [('first_prediction',gold['prediction']),('first_parameter_probe',gold['probe']),('first_loss',gold['loss'])]:
        actual=np.asarray(result[key],float);expected=np.asarray(expected,float)
        if actual.shape!=expected.shape or not np.all(np.isfinite(actual)) or not np.allclose(actual,expected,atol=3e-5,rtol=4e-4):
            raise RuntimeError('First real training step differs from independent NumPy reference: '+key)
    if result['steps_completed']!=config['steps'] or result['examples_trained']!=config['steps']*config['batch']:
        raise RuntimeError('Fixed work incomplete')
    for key,count in [('train_step_samples_ms',config['steps']),('train_losses',config['steps'])]:
        values=np.asarray(result[key],float)
        if len(values)!=count or not np.all(np.isfinite(values)) or (key.endswith('_ms') and np.any(values<=0)):
            raise RuntimeError('Invalid metrics: '+key)
    test=result['test'];latencies=np.asarray(test['samples_ms'],float)
    if not len(latencies) or not np.all(np.isfinite(latencies)) or np.any(latencies<=0):raise RuntimeError('Invalid inference latencies')
    if not (0<=test['accuracy']<=1 and np.isfinite(test['mse']) and test['mse']>=0):raise RuntimeError('Invalid held-out quality')
    if test['accuracy']<min_accuracy:raise RuntimeError(f'Quality gate: test accuracy {test["accuracy"]:.3f} < {min_accuracy}')
    final=np.asarray(result['final_parameter_probe'],float)
    if final.shape!=(len(config['probe_indices']),) or not np.all(np.isfinite(final)):raise RuntimeError('Invalid trained parameters')
    if test['examples']!=config['test_examples'] or len(latencies)!=(config['test_examples']+config['batch']-1)//config['batch']:raise RuntimeError('Inference work incomplete')
    for key in ('train_wall_s','inference_wall_s','cold_step_ms','cold_predict_ms','worker_wall_s'):
        if not np.isfinite(result[key]) or result[key]<=0:raise RuntimeError('Invalid elapsed time: '+key)


def run_suite(directory,configs,frameworks,pythons,binary,worker,environment,affinity,trials,
              timeout_s,memory_mib,min_accuracy,seed,install_errors=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);records=[]
    references={c['case']:first_step_reference(c) for c in configs}
    for config in configs:(directory/(config['case']+'.config.json')).write_text(json.dumps(config,indent=2))
    jobs=[(f,c,t) for t in range(trials) for c in configs for f in frameworks];random.Random(seed).shuffle(jobs)
    for i,(framework,config,trial) in enumerate(jobs,1):
        stem=f'{config["case"]}.{framework}.{trial}';path=directory/(stem+'.json');log=directory/(stem+'.log')
        path.unlink(missing_ok=True)
        Path(str(path)+'.progress.json').unlink(missing_ok=True)
        row=dict(framework=framework,backend=config['backend'],case=config['case'],trial=trial,status='error')
        print(f'Usage reel [{i}/{len(jobs)}] {stem}',flush=True)
        try:
            if framework in (install_errors or {}):raise RuntimeError(install_errors[framework])
            command=([ 'bash',Path(__file__).resolve().parents[2]/'run_mimir.sh',config['backend'],directory/(config['case']+'.config.json'),path] if framework=='mimir' else
                     [pythons[framework],worker,framework,config['backend'],directory/(config['case']+'.config.json'),path])
            command=['taskset','-c',','.join(map(str,affinity)),*command]
            job_environment=dict(environment)
            if framework=='mimir':
                job_environment['MIMIR_EXECUTABLE']=str(Path(binary).resolve())
                job_environment['GOMP_CPU_AFFINITY']=' '.join(map(str,affinity))
            budget=constrained_run(command,job_environment,log,timeout_s,memory_mib)
            row.update(budget)
            if budget['budget_status']!='completed':
                row['status']=budget['budget_status'];raise RuntimeError('Budget/process failure: '+budget['budget_status'])
            measured=json.loads(path.read_text())
            if measured.get('framework')!=framework or measured.get('backend')!=config['backend']:raise RuntimeError('Wrong framework/backend identity')
            row.update(measured)
            validate_result(row,config,references[config['case']],min_accuracy)
            row['status']='ok'
        except Exception as exc:
            if row['status'] in {'measured','running'}:row['status']='error'
            row['error']=str(exc)
            if log.exists():row['log_tail']=log.read_text(errors='replace')[-8000:]
            progress=Path(str(path)+'.progress.json')
            if progress.exists():
                try:row['partial_progress']=json.loads(progress.read_text())
                except (ValueError,OSError):pass
        row.update(trial=trial,case=config['case'],framework=framework,backend=config['backend'])
        path.write_text(json.dumps(row,indent=2,allow_nan=False));records.append(row)
        (directory/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False))
        print(' ->',row['status'],row.get('error',''),flush=True)
    return records


def summarize(records,configs,frameworks,trials):
    import numpy as np
    import pandas as pd
    rows=[]
    for f in frameworks:
        for config in configs:
            group=[r for r in records if r['framework']==f and r['case']==config['case']]
            good=len(group)==trials and all(r['status']=='ok' for r in group)
            row=dict(framework=f,case=config['case'],batch=config['batch'],hidden=config['hidden'],status='ok' if good else 'incomplete',
                trials_ok=sum(r['status']=='ok' for r in group),trials_requested=trials)
            if good:
                training=np.concatenate([r['train_step_samples_ms'] for r in group]);inference=np.concatenate([r['test']['samples_ms'] for r in group])
                row.update(train_examples_per_s=float(np.median([r['examples_trained']/r['train_wall_s'] for r in group])),
                    inference_examples_per_s=float(np.median([r['test']['examples']/r['inference_wall_s'] for r in group])),
                    step_median_ms=float(np.median(training)),step_p95_ms=float(np.percentile(training,95)),
                    step_iqr_ms=float(np.percentile(training,75)-np.percentile(training,25)),
                    train_wall_s_min=min(r['train_wall_s'] for r in group),train_wall_s_max=max(r['train_wall_s'] for r in group),
                    inference_p50_ms=float(np.median(inference)),inference_p95_ms=float(np.percentile(inference,95)),
                    cold_step_ms=float(np.median([r['cold_step_ms'] for r in group])),
                    worker_wall_s=float(np.median([r['worker_wall_s'] for r in group])),
                    process_wall_s=float(np.median([r['process_wall_s'] for r in group])),
                    peak_tree_rss_mib=max(r['peak_tree_rss_mib'] for r in group),
                    test_accuracy_min=min(r['test']['accuracy'] for r in group),test_mse_max=max(r['test']['mse'] for r in group),
                    goal_reached_all=all(r['time_to_goal_s'] is not None for r in group),
                    time_to_goal_s=float(np.median([r['time_to_goal_s'] for r in group])) if all(r['time_to_goal_s'] is not None for r in group) else None,
                    mode=group[0]['mode'],version=group[0]['version'],device=group[0]['device'])
            if good and f=='mimir':
                import re
                inference=group[0].get('inference_plan',{})
                training=group[0].get('training_plan',{})
                row['mimir_planner_mode']=inference.get('mode','unknown')
                row['mimir_reconstruction_loss_backends']=' / '.join(sorted({backend for r in group for backend,count in r.get('reconstruction_loss_runtime_counts',{}).items() if count>0}))
                row['mimir_optimizer_backends']=' / '.join(sorted({backend for r in group for backend,count in r.get('optimizer_runtime_counts',{}).items() if count>0}))
                row['mimir_training_graph_fused_layers']=sum(n.get('graph_fused',False) for n in training.get('nodes',[]))
                row['mimir_inference_graph_fused_layers']=sum(n.get('graph_fused',False) for n in inference.get('nodes',[]))
                row['mimir_inference_kernel_fused_layers']=sum(n.get('kernel_fused',False) for n in inference.get('nodes',[]))
                row['mimir_observed_forward_backends']=' / '.join(sorted(set(
                    backend for r in group for line in r.get('runtime_audit','').splitlines() if 'dispatchForwardLayer' in line for backend in re.findall(r'backend=([^\s]+)',line) if backend!='FUSED_SKIP')))
            if not good:row['error']=' | '.join(sorted(set(r.get('error','missing trial') for r in group if r['status']!='ok'))) or 'missing trial'
            rows.append(row)
    eligible=[f for f in frameworks if sum(r['framework']==f and r['status']=='ok' for r in rows)==len(configs)]
    board=[]
    for f in frameworks:
        own=[r for r in rows if r['framework']==f];score=None
        if f in eligible and len(eligible)>=2:
            ratios=[]
            for row in own:
                contenders=[r for r in rows if r['case']==row['case'] and r['framework'] in eligible]
                for metric in ('train_examples_per_s','inference_examples_per_s'):
                    ratios.append(row[metric]/max(r[metric] for r in contenders))
            score=float(100*np.exp(np.mean(np.log(ratios))))
        board.append(dict(framework=f,coverage=f'{sum(r["status"]=="ok" for r in own)}/{len(configs)}',score_100=score,
                          status='ranked' if score is not None else 'incomplete' if f not in eligible else 'not enough complete contenders'))
    board=pd.DataFrame(board).sort_values('score_100',ascending=False,na_position='last').reset_index(drop=True)
    board['rank']=board['score_100'].rank(method='min',ascending=False)
    detail=pd.DataFrame(rows)
    for config in configs:
        baseline=[r for r in rows if r['framework']=='mimir' and r['case']==config['case'] and r['status']=='ok']
        if baseline:
            mask=(detail['case']==config['case']) & (detail['status']=='ok')
            for metric in ('train','inference'):
                column=metric+'_examples_per_s'
                detail.loc[mask,metric+'_speedup_vs_mimir']=detail.loc[mask,column]/baseline[0][column]
    return detail,board
