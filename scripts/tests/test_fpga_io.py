#!/usr/bin/env python3
"""Real Mímir dataset + SafeTensors/RawFolder I/O, via PTY firmware or --hardware."""
import argparse
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import tempfile
import threading
import zlib

ROOT=Path(__file__).resolve().parents[2]
LUA=r'''
assert(Mimir.Model.set_hardware(false))
local caps=Mimir.Model.hardware_caps().fpga
assert(caps.connected and caps.dataset_buffer and caps.checkpoint_controller and caps.tensor_conversion, 'FPGA I/O firmware >=1.3 required')
local input=os.getenv('MIMIR_FPGA_TEST_IMAGE')
local directory=os.getenv('MIMIR_FPGA_TEST_DIR')
local ok,n=Mimir.Dataset.load(os.getenv('MIMIR_FPGA_TEST_DATASET'),8,8,1,false,'',16,true)
assert(ok,n);assert(n==1)
local item=assert(Mimir.Dataset.get(1,true))
local direct=assert(Mimir.IO.readImageRGBU8(input,8,8))
assert(#item.image==192 and #direct.image==192)
for i=1,192 do assert(item.image[i]==direct.image[i]) end
assert(Mimir.Model.create('basic_mlp',{input_dim=4,hidden_dim=3,output_dim=2}))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights('xavier',42))
local values={.1,-.2,.3,.4}
for _,dtype in ipairs({'float16','bfloat16'}) do
  for _,format in ipairs({'raw_folder','safetensors'}) do
    assert(Mimir.Model.dtype(dtype))
    local before=assert(Mimir.Model.forward(values,false))
    local path=directory..'/'..dtype..'-'..format..(format=='safetensors' and '.safetensors' or '')
    assert(Mimir.Serialization.save(path,format,{save_tokenizer=false,save_encoder=false}))
    assert(Mimir.Model.init_weights('xavier',99))
    assert(Mimir.Serialization.load(path,format,{load_tokenizer=false,load_encoder=false,strict_mode=true}))
    local after=assert(Mimir.Model.forward(values,false))
    assert(#before==#after)
    for i=1,#before do assert(math.abs(before[i]-after[i])<.02,'checkpoint roundtrip changed prediction') end
  end
end
print('FPGA_DATASET_CHECKPOINT_OK')
'''


def fixture(path):
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    raw=b''.join(b'\0'+b''.join(bytes([x*24,y*24,80]) for x in range(8)) for y in range(8))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',8,8,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


class Firmware:
    def __init__(self,master,legacy=False,corrupt=False):
        self.legacy=legacy;self.corrupt=corrupt;self.master=master;self.stop=threading.Event();self.operation=0;self.stream=bytearray();self.modes=set();self.error=None
        self.worker=threading.Thread(target=self.run)
    def read(self,count):
        data=bytearray()
        while len(data)<count:
            if self.stop.is_set():
                raise EOFError
            ready,_,_=select.select([self.master],[],[],.1)
            if ready:
                part=os.read(self.master,count-len(data))
                if not part: raise EOFError
                data.extend(part)
        return bytes(data)
    def write(self,data):
        while data:
            count=os.write(self.master,data)
            data=data[count:]
    def run(self):
        try:
            while not self.stop.is_set():
                command=self.read(4)
                if command==b'MIMI':
                    assert self.read(4)==b'R?\x01\n'
                    self.write(b'MIMIRFPG'+struct.pack('<BBIH',1,2 if self.legacy else 3,7 if self.legacy else 15,64))
                elif command==b'DOT8':
                    count=self.read(1)[0];raw=self.read(count*2)
                    values=[v if v<128 else v-256 for v in raw]
                    self.write(b'DRES'+struct.pack('<i',sum(a*b for a,b in zip(values[::2],values[1::2]))))
                elif command==b'@BEG':
                    self.operation=self.read(1)[0];self.stream.clear();self.write(b'CACK')
                elif command==b'@END':
                    self.write(b'CRES'+bytes([self.operation])+struct.pack('<QI',len(self.stream),zlib.crc32(self.stream)))
                elif command==b'@BUF':
                    mode,count,crc=struct.unpack('<BHI',self.read(7));raw=self.read(count)
                    assert zlib.crc32(raw)==crc
                    self.modes.add(mode)
                    if self.operation: self.stream.extend(raw)
                    if mode in (0,5): out=raw
                    elif mode==1:
                        words=struct.unpack('<'+'I'*(len(raw)//4),raw)
                        out=struct.pack('<'+'H'*len(words),*[((v+0x7fff+((v>>16)&1))>>16)&65535 for v in words])
                    elif mode==2:
                        words=struct.unpack('<'+'H'*(len(raw)//2),raw)
                        out=struct.pack('<'+'I'*len(words),*[v<<16 for v in words])
                    elif mode==3:
                        words=struct.unpack('<'+'f'*(len(raw)//4),raw)
                        out=struct.pack('<'+'e'*len(words),*words)
                    elif mode==4:
                        words=struct.unpack('<'+'e'*(len(raw)//2),raw)
                        out=struct.pack('<'+'f'*len(words),*words)
                    else: raise AssertionError(mode)
                    self.write(b'XRES'+bytes([mode])+struct.pack('<H',len(out))+out+struct.pack('<I',zlib.crc32(out) ^ (1 if self.corrupt else 0)))
                else: raise AssertionError(command)
        except EOFError: pass
        except BaseException as error: self.error=error


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--hardware',action='store_true');parser.add_argument('--legacy',action='store_true');parser.add_argument('--corrupt-crc',action='store_true');args=parser.parse_args()
    assert not (args.hardware and (args.legacy or args.corrupt_crc))
    with tempfile.TemporaryDirectory(prefix='mimir-fpga-io-test-') as tmp:
        base=Path(tmp);dataset=base/'dataset';dataset.mkdir();image=dataset/'input.png';fixture(image)
        lua=base/'test.lua';source=LUA
        if args.legacy:
            source=source.replace("assert(caps.connected and caps.dataset_buffer and caps.checkpoint_controller and caps.tensor_conversion, 'FPGA I/O firmware >=1.3 required')","assert(caps.connected and not caps.dataset_buffer, 'legacy firmware must use CPU I/O')")
        lua.write_text(source)
        env={**os.environ,'OMP_NUM_THREADS':'1','MIMIR_DISABLE_FPGA':'0','MIMIR_FPGA_IO':'1','MIMIR_FPGA_TEST_IMAGE':str(image),'MIMIR_FPGA_TEST_DIR':str(base),'MIMIR_FPGA_TEST_DATASET':str(dataset)}
        for runtime in ['VULKAN','OPENCL','ROCM','CUDA']: env['MIMIR_DISABLE_'+runtime]='1'
        firmware=None
        if not args.hardware:
            master,slave=pty.openpty()
            root=base/'sysfs';board=root/'board';(board/'ttyACM0').mkdir(parents=True)
            for name,value in [('idVendor','1d50'),('idProduct','602b'),('serial','0710test')]: (board/name).write_text(value+'\n')
            env['MIMIR_FPGA_SYSFS_ROOT']=str(root);env['MIMIR_FPGA_DEVICE']=os.ttyname(slave)
            firmware=Firmware(master,args.legacy,args.corrupt_crc);firmware.worker.start()
        try:
            result=subprocess.run([str(ROOT/'bin/mimir'),'--lua',str(lua)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        finally:
            if firmware:
                firmware.stop.set();firmware.worker.join(2);os.close(master);os.close(slave)
        if args.corrupt_crc:
            assert result.returncode!=0, 'corrupt UART CRC was accepted'
            assert 'FPGA_DATASET_CHECKPOINT_OK' not in result.stdout
            print('FPGA_IO_BAD_CRC_REJECTED');return
        assert result.returncode==0,result.stdout+result.stderr
        assert 'FPGA_DATASET_CHECKPOINT_OK' in result.stdout
        if args.legacy:
            assert firmware.error is None,firmware.error
            assert not firmware.modes,firmware.modes
            print('FPGA_IO_LEGACY_CPU_FALLBACK_OK');return
        for marker in ['dataset_buffer bytes=192','checkpoint_save','checkpoint_load','F32->F16','F16->F32','F32->BF16','BF16->F32']:
            assert marker in result.stderr,marker+' missing\n'+result.stderr
        if firmware:
            assert firmware.error is None,firmware.error
            assert firmware.modes=={0,1,2,3,4,5},firmware.modes
        print('FPGA_IO_HOST_OK: dataset readback, native conversions, CRC controller, SafeTensors/RawFolder roundtrips ('+('hardware' if args.hardware else 'PTY')+')')
        if args.hardware:
            print('\n'.join(line for line in result.stderr.splitlines() if '[FPGA]' in line))

if __name__=='__main__': main()
