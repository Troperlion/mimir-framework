"""Private, exact-host-version NVIDIA Vulkan userspace for Colab.

Never run nvidia-installer, install a kernel module, or replace system CUDA libs.
Only the NVIDIA official self-extractor is run with --extract-only.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import urllib.request

GRAPHICS_PREFIXES=('libEGL_nvidia.so','libGLX_nvidia.so','libnvidia-eglcore.so',
                   'libnvidia-glcore.so','libnvidia-glsi.so','libnvidia-gpucomp.so',
                   'libnvidia-allocator.so','libnvidia-tls.so','libnvidia-glvkspirv.so','libnvidia-vulkan-producer.so')


def nvidia_device(output):
    # Inspect physical-device blocks, rather than loader messages mentioning NVIDIA.
    return any(re.search(r'vendorID\s*=\s*(?:0x10de|4318)\b',block,re.I) and
               re.search(r'deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(?:DISCRETE|INTEGRATED)_GPU',block)
               for block in re.split(r'GPU\d+\s*:',output))


def probe(environment):
    p=subprocess.run(['vulkaninfo','--summary'],env=environment,capture_output=True,text=True,timeout=90)
    return p.returncode==0 and nvidia_device(p.stdout),p.stdout+p.stderr


def stage_libraries(extracted,destination):
    """Copy native x86_64 graphics ELF files and reconstruct their actual SONAMEs."""
    destination.mkdir(parents=True,exist_ok=True)
    copied=[]
    for source in sorted(Path(extracted).rglob('*.so*')):
        if source.is_symlink() or not source.is_file() or not source.name.startswith(GRAPHICS_PREFIXES):continue
        with source.open('rb') as stream:header=stream.read(20)
        if len(header)<20 or header[:5]!=b'\x7fELF\x02' or header[18:20]!=b'\x3e\x00':continue
        target=destination/source.name
        shutil.copy2(source,target)
        dynamic=subprocess.run(['readelf','-d',str(target)],capture_output=True,text=True,check=True)
        soname=re.search(r'\(SONAME\).*?\[([^\]]+)\]',dynamic.stdout)
        if soname:
            name=soname.group(1)
            if Path(name).name!=name:raise RuntimeError('Invalid library SONAME')
            link=destination/name
            if link!=target:
                link.unlink(missing_ok=True);link.symlink_to(target.name)
        copied.append(target.name)
    if not any(name.startswith('libEGL_nvidia.so') for name in copied):
        raise RuntimeError('NVIDIA archive has no native libEGL_nvidia')
    return copied


def configure(environment,root,run):
    """Return (child environment, audit report); callers keep availability policy."""
    original=dict(environment)
    report=dict(status='unavailable',method='private NVIDIA graphics userspace')
    try:
        ready,output=probe(original)
        if ready:
            report.update(status='host_ready',probe=output)
            return original,report
        report['host_probe']=output
        if platform.machine()!='x86_64':raise RuntimeError('Automatic NVIDIA extraction requires x86_64')
        version_output=run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],timeout=30)
        versions=set(version_output.strip().splitlines())
        if len(versions)!=1:raise RuntimeError('Cannot determine a single NVIDIA host driver version')
        version=versions.pop().strip()
        if not re.fullmatch(r'\d{3}\.\d{2}(?:\.\d{2})?',version):raise RuntimeError('Invalid NVIDIA version: '+version)
        root=Path(root).resolve()/version;root.mkdir(parents=True,exist_ok=True)
        archive=root/f'NVIDIA-Linux-x86_64-{version}.run'
        url=f'https://download.nvidia.com/XFree86/Linux-x86_64/{version}/{archive.name}'
        report.update(driver_version=version,url=url)
        if not archive.is_file():
            partial=archive.with_suffix('.partial')
            print('Téléchargement des composants NVIDIA officiels :',url,flush=True)
            try:
                with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as target:
                    shutil.copyfileobj(response,target)
                partial.replace(archive)
            finally:partial.unlink(missing_ok=True)
        digest=hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        report['archive_sha256']=digest.hexdigest()
        extracted=root/'extracted'
        if not (extracted/'.mimir-complete').exists():
            if extracted.exists():shutil.rmtree(extracted)
            run(['sh',archive,'--extract-only','--target',extracted],timeout=600)
            (extracted/'.mimir-complete').touch()
        library_dir=root/'lib'
        report['libraries']=stage_libraries(extracted,library_dir)
        manifests=list(extracted.glob('nvidia_icd.json'))
        if not manifests:raise RuntimeError('Official NVIDIA ICD manifest absent from archive')
        icd=json.loads(manifests[0].read_text())
        egl=next(library_dir/name for name in report['libraries'] if name.startswith('libEGL_nvidia.so'))
        icd['ICD']['library_path']=str(egl)
        manifest=root/'nvidia_icd.json';manifest.write_text(json.dumps(icd,indent=2))
        candidate=dict(original,VK_DRIVER_FILES=str(manifest),VK_ICD_FILENAMES=str(manifest),
                       LD_LIBRARY_PATH=str(library_dir)+(':'+original['LD_LIBRARY_PATH'] if original.get('LD_LIBRARY_PATH') else ''))
        ready,output=probe(candidate)
        report.update(probe=output,icd=str(manifest))
        if not ready:raise RuntimeError('Private driver did not enumerate an NVIDIA Vulkan GPU; native tests remain authoritative')
        report['status']='private_ready'
        return candidate,report
    except Exception as exc:
        report['error']=str(exc)
        print('Vulkan NVIDIA :',str(exc),flush=True)
        return original,report
