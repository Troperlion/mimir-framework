#!/usr/bin/env python3
"""Install/refresh the real-workload section without replacing existing cells."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def cell(kind,source):
    source=source.strip()+'\n'
    result=dict(cell_type=kind,id=hashlib.sha256(source.encode()).hexdigest()[:12],
                metadata={'mimir_real_benchmark':True},source=source.splitlines(True))
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result

CONFIG='''
# Real workload: independent from the microbenchmark score.
RUN_REAL_BENCH=True
REAL_PROFILE='standard'  # quick = smoke on real data; standard = full MNIST split
REAL_TRIALS=2 if REAL_PROFILE=='standard' else 1
REAL_BATCHES=[32,128,512] if REAL_PROFILE=='standard' else [64]
REAL_HIDDEN=128
REAL_STEPS=400 if REAL_PROFILE=='standard' else 80
REAL_EVAL_EVERY=50 if REAL_PROFILE=='standard' else 40
REAL_LEARNING_RATE=.2
REAL_TARGET_ACCURACY=.85
REAL_MIN_TEST_ACCURACY=.75 if REAL_PROFILE=='standard' else .4
REAL_PROCESS_TIMEOUT_S=180
REAL_HOST_MEMORY_MIB=4096  # monitored sum RSS+swap across process and its workers
REAL_MIMIR_OPTIONS=dict(planner_mode='cost',fusion=True,fusion_train=True,
    buffer_reuse=True,device_residency=True,allow_cpu_fallback=True)
assert REAL_PROFILE in {'quick','standard'} and REAL_TRIALS>=1
assert all(isinstance(b,int) and b>0 for b in REAL_BATCHES) and len(set(REAL_BATCHES))==len(REAL_BATCHES)
assert REAL_STEPS>0 and REAL_EVAL_EVERY>0 and REAL_HIDDEN>0 and REAL_LEARNING_RATE>0
assert 0<=REAL_MIN_TEST_ACCURACY<=1 and 0<=REAL_TARGET_ACCURACY<=1
assert REAL_PROCESS_TIMEOUT_S>0 and REAL_HOST_MEMORY_MIB>0
'''
PREPARE='''
if RUN_REAL_BENCH:
    import importlib.util
    real_source=REPO/'scripts/benchmarks'
    required_real=['colab_real_suite.py','colab_real_worker.py','colab_real_model.cpp']
    missing_real=[name for name in required_real if not (real_source/name).is_file()]
    if missing_real:
        raise RuntimeError('Ancien instantane Mimir : '+str(missing_real)+'. Recharger la version actuelle du notebook puis relancer depuis la premiere cellule (configuration et extraction des sources comprises).')
    spec=importlib.util.spec_from_file_location('mimir_colab_real_suite',real_source/'colab_real_suite.py')
    REAL_SUITE=importlib.util.module_from_spec(spec);spec.loader.exec_module(REAL_SUITE)
    REAL_DIR=RUN_DIR/('real-workload-'+datetime.now(timezone.utc).strftime('%H%M%S')+'-'+os.urandom(3).hex());REAL_DIR.mkdir(exist_ok=False)
    REAL_DATA=REAL_DIR/'data'
    REAL_DATA_MANIFEST=REAL_SUITE.prepare_data(REAL_DATA,[REAL_HIDDEN],SEED,
        quick=REAL_PROFILE=='quick',archive=Path('/content/mimir-bench-cache/mnist.npz'))
    sizes=[REAL_HIDDEN*784,REAL_HIDDEN,10*REAL_HIDDEN,10]
    probe_indices=[];offset=0
    for size in sizes:
        probe_indices.extend((offset+np.unique(np.linspace(0,size-1,min(32,size),dtype=int))).tolist());offset+=size
    REAL_CASES=[dict(case=f'mnist-mlp-{REAL_HIDDEN}-batch-{batch}',backend=BACKEND,
        data_dir=str(REAL_DATA),mimir_version=(REPO/'VERSION').read_text().strip(),hidden=REAL_HIDDEN,batch=batch,steps=REAL_STEPS,lr=REAL_LEARNING_RATE,
        eval_every=REAL_EVAL_EVERY,target_accuracy=REAL_TARGET_ACCURACY,probe_indices=probe_indices,mimir=REAL_MIMIR_OPTIONS,
        train_examples=(REAL_DATA/'train.labels.u8').stat().st_size,
        test_examples=(REAL_DATA/'test.labels.u8').stat().st_size) for batch in REAL_BATCHES]
    # Add a native public-Model driver to the existing deferred CMake hook.
    real_hook=''' + repr('''
function(mimir_colab_add_real_benchmark)
  if(TARGET mimir_core AND NOT TARGET mimir_colab_real)
    add_executable(mimir_colab_real "${CMAKE_SOURCE_DIR}/scripts/benchmarks/colab_real_model.cpp")
    target_compile_features(mimir_colab_real PRIVATE cxx_std_17)
    target_link_libraries(mimir_colab_real PRIVATE mimir_core)
    set_target_properties(mimir_colab_real PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  endif()
endfunction()
cmake_language(DEFER CALL mimir_colab_add_real_benchmark)
''') + '''
    original_hook=INJECT.read_text()
    if 'function(mimir_colab_add_real_benchmark)' not in original_hook:INJECT.write_text(original_hook+real_hook)
    run(CMAKE_ARGS+['-DCMAKE_PROJECT_INCLUDE='+str(INJECT)])
    run(['cmake','--build',BUILD,'-j',JOBS,'--target','mimir_colab_real'],timeout=2400)
    REAL_BINARY=BUILD/'bin/mimir_colab_real'
    REAL_WORKER=REAL_DIR/'sources/colab_real_worker.py'
    provenance=dict(launcher_sha256=REAL_SUITE.sha256(REPO/'run_mimir.sh'),profile=REAL_PROFILE,trials=REAL_TRIALS,cases=REAL_CASES,
        process_timeout_s=REAL_PROCESS_TIMEOUT_S,host_rss_plus_swap_limit_mib=REAL_HOST_MEMORY_MIB,
        min_test_accuracy=REAL_MIN_TEST_ACCURACY,cpu_affinity=CPU_AFFINITY,threads=THREADS,
        seed=SEED,dataset=REAL_DATA_MANIFEST,source_snapshot_sha256=globals().get('SOURCE_SHA256'),
        sources={p.name:REAL_SUITE.sha256(p) for p in real_source.glob('colab_real_*') if p.is_file()},
        contract='persistent weights; public whole-model APIs; host batch preprocessing and device transfer; synchronized SGD; fixed validation schedule; final held-out inference',
        fp32=True,tf32=False,mimir_options=REAL_MIMIR_OPTIONS,cmake_args=list(map(str,CMAKE_ARGS)),
        mimir_layers='production planner; CPU/native accelerator placement allowed; native host loss; SGD dispatched by runtime router; see plans and runtime audit',
        gpu_memory_limit='physical capacity only; no artificial VRAM/HBM cap',memory_poll_s=.05)
    (REAL_DIR/'protocol.json').write_text(json.dumps(provenance,indent=2))
    (REAL_DIR/'sources').mkdir(exist_ok=True)
    for p in real_source.glob('colab_real_*'):
        if p.is_file():shutil.copy2(p,REAL_DIR/'sources'/p.name)
    print('Usage reel :',len(REAL_CASES),'cas,',REAL_TRIALS,'essais,',REAL_STEPS,'updates/cas ;',REAL_DIR)
'''
MEASURE='''
if RUN_REAL_BENCH:
    if not VALIDATED:raise RuntimeError('Validation Mimir requise avant les essais reels')
    REAL_RECORDS=REAL_SUITE.run_suite(REAL_DIR,REAL_CASES,FRAMEWORKS,PYTHONS,REAL_BINARY,REAL_WORKER,
        RUNTIME_ENV,CPU_AFFINITY,REAL_TRIALS,REAL_PROCESS_TIMEOUT_S,REAL_HOST_MEMORY_MIB,
        REAL_MIN_TEST_ACCURACY,SEED,INSTALL_ERRORS)
    assert provenance['sources']=={p.name:REAL_SUITE.sha256(p) for p in real_source.glob('colab_real_*') if p.is_file()}, 'Sources du benchmark modifiees pendant les essais'
    assert REAL_DATA_MANIFEST['files']=={p.name:REAL_SUITE.sha256(p) for p in sorted(REAL_DATA.iterdir()) if p.suffix in {'.u8','.f32'}}, 'Donnees modifiees pendant les essais'
'''
REPORT='''
if RUN_REAL_BENCH:
    REAL_DETAIL,REAL_BOARD=REAL_SUITE.summarize(REAL_RECORDS,REAL_CASES,FRAMEWORKS,REAL_TRIALS)
    display(REAL_BOARD[['rank','framework','coverage','score_100','status']])
    with pd.option_context('display.max_colwidth',None):display(REAL_DETAIL)
    REAL_DETAIL.to_csv(REAL_DIR/'real-detail.csv',index=False)
    REAL_BOARD.to_csv(REAL_DIR/'real-leaderboard.csv',index=False)
    errors='\\n\\n'.join(f"{r['framework']} | {r['case']} | essai {r['trial']+1}\\n{r.get('error','')}\\n{r.get('log_tail','')}"
        for r in REAL_RECORDS if r['status']!='ok')
    (REAL_DIR/'errors.txt').write_text(errors)
    if errors:print(errors)
    report='<html><head><meta charset="utf-8"></head><body><h1>Usage reel sous contraintes</h1>'
    report+='<p>Score = moyenne geometrique des debits relatifs au meilleur, entrainement et inference, poids egaux par batch. Qualite minimale et budgets obligatoires sur tous les essais. Ce score est distinct des microbenchmarks.</p>'
    report+=REAL_BOARD.to_html(index=False,escape=True)+REAL_DETAIL.to_html(index=False,escape=True)+'</body></html>'
    (REAL_DIR/'report.html').write_text(report)
    print('Rapport usage reel :',REAL_DIR/'report.html')
    print('Speedup_vs_mimir > 1 : le concurrent est plus rapide ; < 1 : Mimir est plus rapide.')
    print('time_to_goal absent : seuil non atteint aux evaluations observees ; jamais impute a zero.')
    valid=REAL_DETAIL[REAL_DETAIL.status=='ok']
    if not valid.empty:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,metric,title in zip(axes,['train_examples_per_s','inference_examples_per_s'],['Entrainement + validation','Inference tenue a part']):
            valid.pivot(index='batch',columns='framework',values=metric).plot.bar(ax=ax,logy=True)
            ax.set_title(title);ax.set_ylabel('Images/s (log)')
        plt.tight_layout();plt.savefig(REAL_DIR/'throughput.png',dpi=160);plt.show()
    # The existing final archive includes this entire subdirectory and its logs.
'''

def main():
    path=ROOT/'Colab-bench-multi-framework.ipynb';notebook=json.loads(path.read_text())
    notebook['cells']=[c for c in notebook['cells'] if not c.get('metadata',{}).get('mimir_real_benchmark')]
    for existing in notebook['cells']:
        if existing['cell_type']=='code':
            text=''.join(existing['source'])
            if "'-DMIMIR_PORTABLE_BUILD=ON'" in text:
                text=text.replace("'-DMIMIR_PORTABLE_BUILD=ON'", "'-DMIMIR_PORTABLE_BUILD=OFF','-DENABLE_SIMD=ON','-DENABLE_OPENMP=ON'")
                existing['source']=text.splitlines(True)
    index=next(i for i,c in enumerate(notebook['cells']) if c['cell_type']=='markdown' and ''.join(c['source']).startswith('## Visualisation et export'))
    cells=[cell('markdown','''## Usage réel : entraînement et inférence MNIST sous contraintes

Ce second tableau mesure une utilisation complète des API : **Mímir `Model.trainStep`/`forwardPass`**, PyTorch `nn.Sequential`/autograd/SGD, Keras `Sequential`/`GradientTape`/SGD et JAX `jit(value_and_grad)`. Les poids persistent entre les mises à jour. Il ne réinitialise pas le modèle à chaque pas.

- MLP `784 → 128 → ReLU → 10`, FP32, MSE sur labels one-hot, SGD sans momentum ni weight decay. Même initialisation, mêmes images, mêmes batches cycliques après un mélange initial commun. Batchs comparés séparément ; aucun score ne mélange des charges différentes au sein d'un cas.
- MNIST est téléchargé et contrôlé par le [SHA256 publié par Keras](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py). Le train officiel est conservé ; les 2 000 premières images du test officiel servent à la validation périodique, les 8 000 autres à l'évaluation finale. Le profil `quick` utilise des sous-ensembles et sert au contrôle fonctionnel.
- **Contraintes appliquées** : même affinité et nombre de threads CPU, plafond mémoire hôte RSS+swap cumulé du processus et de ses workers, délai total par essai (imports, compilation et données compris). Un watchdog externe termine tout le groupe de processus en cas de dépassement. Échantillonnage à 50 ms : une pointe entre deux mesures peut échapper au moniteur ; les pages partagées peuvent être comptées plusieurs fois. Pas de quota VRAM/HBM artificiel ; les limites physiques et OOM restent visibles.
- Les étapes chaudes incluent préparation du batch, transferts, forward, loss, backward, SGD et synchronisation. Les frameworks conservent leurs paramètres selon leur API native. Chez Mímir, loss/SGD restent côté hôte et les transferts du runtime TPU sont inclus ; le planner `cost` peut choisir des routes CPU et accélérateur, comme en utilisation réelle. Fusions, réutilisation des buffers et résidence sur le device sont autorisées ; la compilation native active SIMD et OpenMP. Les plans et la trace de diagnostic indiquent ce qui est applicable : une option activée ne garantit pas une fusion de kernel ou une résidence effective sur chaque runtime. Les fusions d'entraînement restent limitées à celles prises en charge par le framework. Ce scénario mesure ces chemins publics et ce MLP ; il ne résume pas tous les usages ou optimisations des frameworks.
- Première étape vérifiée contre une référence NumPy float64 (prédictions, loss et sondes de paramètres après update, biais inclus). Une précision minimale sur le test tenu à part est obligatoire pour classer un essai. L'entraînement continue jusqu'au travail fixe demandé, même si le seuil de validation est atteint.
- Mesures : débit d'entraînement **validation périodique incluse**, débit d'inférence, p50/p95/IQR des étapes, latences à froid, temps total, pic mémoire du groupe, précision/MSE et temps observé pour atteindre la précision cible. Les deux modes s'évaluent sur le même unique accélérateur, dans des processus isolés et un ordre aléatoire.
- La note d'usage réel est indépendante de celle des opérations : moyenne géométrique des débits relatifs au meilleur, poids égaux entre entraînement, inférence et batchs. Tous les cas et essais doivent passer les budgets, la vérification numérique et le seuil de qualité ; il faut deux frameworks complets. Dépassements, OOM, erreurs et qualité insuffisante restent non classés. Les médianes et p95 ne constituent pas à eux seuls une preuve de significativité statistique.
'''),cell('code',CONFIG),cell('markdown','### Données communes, modèle natif et provenance'),cell('code',PREPARE),cell('markdown','### Essais isolés sous budgets effectifs'),cell('code',MEASURE),cell('markdown','### Comparaison d’usage réel et écarts avec Mímir'),cell('code',REPORT)]
    notebook['cells'][index:index]=cells
    intro=notebook['cells'][0]
    text=''.join(intro['source'])
    text=text.replace("Le périmètre est un ensemble déclaré d'opérations FP32", "Le premier tableau porte sur un ensemble déclaré d'opérations FP32")
    marker='Deux tableaux indépendants : opérations élémentaires, puis entraînement et inférence MNIST sous contraintes effectives.\n'
    if marker not in text:text+='\n'+marker
    intro['source']=text.splitlines(True)
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
