"""Exercise config.sh feature selection and dependency installation without sudo."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ConfigureCliTest(unittest.TestCase):
    def run_config(self,*arguments):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);bin_dir=directory/'bin';bin_dir.mkdir()
            log=directory/'calls.jsonl'
            tool='''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
with open(os.environ['MIMIR_CONFIG_TEST_CALLS'],'a') as log:
 log.write(json.dumps([Path(sys.argv[0]).name,*sys.argv[1:]])+'\\n')
if Path(sys.argv[0]).name=='sudo':
 os.execvp(sys.argv[1],sys.argv[1:])
print('mock tool 1.0')
'''
            for name in ['sudo','apt-get','cmake','c++','lua','pkg-config']:
                p=bin_dir/name;p.write_text(tool);p.chmod(0o755)
            environment={**os.environ,'PATH':str(bin_dir)+os.pathsep+os.environ['PATH'],'MIMIR_CONFIG_TEST_CALLS':str(log),'RUSTICL_ENABLE':'','SKIP_SYSTEM_DEPS':'0'}
            result=subprocess.run(['bash',str(ROOT/'config.sh'),'--gui','NONE','--non-interactive','--build-dir',str(directory/'build'),'--disable','ENABLE_VULKAN','--disable','ENABLE_OPENCL','--disable','ENABLE_FFMPEG',*arguments],env=environment,capture_output=True,text=True)
            calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result,calls

    def test_features_drive_dependencies_and_cmake(self):
        result,calls=self.run_config('--enable','MIMIR_ENABLE_TESTS','--enable','ENABLE_SCRIPTING_JS','--generator','Ninja')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn(['apt-get','update'],calls)
        install=next(c for c in calls if c[:2]==['apt-get','install'])
        for dependency in ['ninja-build','liblua5.3-dev','liblz4-dev']:
            self.assertIn(dependency,install)
        self.assertNotIn('libavcodec-dev',install)
        cmake=next(c for c in calls if c[0]=='cmake' and '-S' in c)
        for option in ['-DMIMIR_ENABLE_TESTS=ON','-DENABLE_SCRIPTING_JS=ON','-DENABLE_VULKAN=OFF','-DMIMIR_VIZ_BACKEND=NONE','Ninja']:
            self.assertIn(option,cmake)

    def test_skip_packages_and_preserve_cross_arguments(self):
        result,calls=self.run_config('--skip-system-deps','--cmake-arg','-DCMAKE_SYSTEM_NAME=Windows','--cmake-arg','-DCMAKE_TOOLCHAIN_FILE=/directory with spaces/toolchain.cmake','--build-type','Debug')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse(any(c[0] in ['apt-get','sudo'] for c in calls))
        cmake=next(c for c in calls if c[0]=='cmake' and '-S' in c)
        for value in ['-DCMAKE_SYSTEM_NAME=Windows','-DCMAKE_TOOLCHAIN_FILE=/directory with spaces/toolchain.cmake','-DCMAKE_BUILD_TYPE=Debug']:
            self.assertIn(value,cmake)

    def test_invalid_selection_exits_before_installation(self):
        for arguments in [('--enable','UNKNOWN_FEATURE'),('--disable',),('--build-type','Invalid'),('--generator',)]:
            result,calls=self.run_config(*arguments)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(any(c[0] in ['apt-get','sudo'] for c in calls))


if __name__=='__main__':unittest.main()
