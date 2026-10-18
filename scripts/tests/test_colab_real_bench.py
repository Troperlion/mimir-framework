#!/usr/bin/env python3
"""Focused integrity and resource-budget regression checks (no accelerator needed)."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('colab_real_suite',ROOT/'scripts/benchmarks/colab_real_suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
try:
    import numpy as np
    import pandas
except ImportError:
    np=None


class Budgets(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','requires Linux /proc and process groups')
    def test_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            result=suite.constrained_run([sys.executable,'-c','import time; time.sleep(10)'],dict(os.environ),Path(d)/'log',.2,512)
            self.assertEqual(result['budget_status'],'time_limit')
            self.assertLess(result['process_wall_s'],5)

    @unittest.skipUnless(sys.platform=='linux','requires Linux /proc and process groups')
    def test_child_memory_counted(self):
        script="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; b=bytearray(96*1024*1024); time.sleep(10)']); time.sleep(10)"
        with tempfile.TemporaryDirectory() as d:
            result=suite.constrained_run([sys.executable,'-c',script],dict(os.environ),Path(d)/'log',5,48)
            self.assertEqual(result['budget_status'],'memory_limit')
            self.assertGreater(result['peak_tree_rss_mib'],48)


@unittest.skipIf(np is None,'numpy/pandas not installed')
class Ranking(unittest.TestCase):
    def test_incomplete_excluded_and_geometric_score(self):
        def record(framework,trial,train_s,infer_s):
            return dict(framework=framework,case='toy',trial=trial,status='ok',train_step_samples_ms=[1.,2.],
                train_wall_s=train_s,examples_trained=128,cold_step_ms=1.,worker_wall_s=2.,process_wall_s=2.1,
                peak_tree_rss_mib=10.,time_to_goal_s=None,mode='unit-test fixture',version='fixture',device='cpu',
                test=dict(samples_ms=[1.,2.],examples=64,accuracy=.9,mse=.1),inference_wall_s=infer_s)
        records=[record(f,t,train,infer) for f,train,infer in [('mimir',1.28,.64),('pytorch',.64,.16)] for t in (0,1)]
        records.append(dict(framework='jax',case='toy',trial=0,status='memory_limit',error='fixture limit'))
        detail,board=suite.summarize(records,[dict(case='toy',batch=64,hidden=4)],['mimir','pytorch','jax'],2)
        scores=board.set_index('framework')
        self.assertAlmostEqual(scores.loc['mimir','score_100'],100*np.sqrt(.5*.25))
        self.assertAlmostEqual(scores.loc['pytorch','score_100'],100)
        self.assertTrue(np.isnan(scores.loc['jax','score_100']))
        self.assertTrue(np.isnan(scores.loc['jax','rank']))
        self.assertTrue(detail[detail.framework=='mimir'].time_to_goal_s.isna().all())

    def test_mimir_requires_the_requested_planner(self):
        config=dict(case='toy',mimir=dict(planner_mode='cost'))
        result=dict(case='toy',framework='mimir',status='measured',training_plan=dict(planned=False))
        with self.assertRaisesRegex(RuntimeError,'planner not active'):
            suite.validate_result(result,config,{},.8)
        result['training_plan']=dict(planned=True,mode='static',fusion=True)
        with self.assertRaisesRegex(RuntimeError,'planner not active'):
            suite.validate_result(result,config,{},.8)

    def test_quality_and_elapsed_times_cannot_bypass_gate(self):
        config=dict(case='toy',batch=2,steps=2,probe_indices=[0,1],test_examples=2)
        gold=dict(prediction=np.zeros(20),probe=np.zeros(2),loss=0.)
        measured=dict(case='toy',status='measured',first_prediction=[0.]*20,first_parameter_probe=[0.,0.],first_loss=0.,
            steps_completed=2,examples_trained=4,train_step_samples_ms=[1.,2.],train_losses=[0.,0.],
            test=dict(accuracy=.9,mse=.1,samples_ms=[1.],examples=2),final_parameter_probe=[0.,0.],
            train_wall_s=1.,inference_wall_s=1.,cold_step_ms=1.,cold_predict_ms=1.,worker_wall_s=2.)
        suite.validate_result(measured,config,gold,.8)
        measured['test']['accuracy']=.1
        with self.assertRaisesRegex(RuntimeError,'Quality gate'):suite.validate_result(measured,config,gold,.8)
        measured['test']['accuracy']=.9;measured['train_wall_s']=float('nan')
        with self.assertRaisesRegex(RuntimeError,'elapsed'):suite.validate_result(measured,config,gold,.8)


if __name__=='__main__':unittest.main()
