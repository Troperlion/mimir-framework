#!/usr/bin/env python3
"""Simulate the actual UART staging/controller RTL and IEEE conversions."""
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import zlib

ROOT=Path(__file__).resolve().parents[2]
BASE=r'''
`timescale 1ns/1ps
module testbench;
reg clk=0,rx=1; wire tx;
always #5 clk=~clk;
top #(.CLOCK_HZ(8000),.BAUD(1000),.REQUEST_TIMEOUT_CYCLES(1000))
 dut(.clk25(clk),.serial_rx(rx),.serial_tx(tx));
reg [7:0] received[0:65535];integer count=0,cursor=0,bit_index;
reg [7:0] byte_value;
always begin
 @(negedge tx);repeat(12) @(posedge clk);#1;
 for(bit_index=0;bit_index<8;bit_index=bit_index+1) begin
  byte_value[bit_index]=tx;repeat(8) @(posedge clk);#1;
 end
 if(tx!==1'b1) $fatal(1,"UART stop bit");
 received[count]=byte_value;count=count+1;
end
task send;input [7:0] value;integer b;begin
 @(negedge clk);rx=0;repeat(8) @(negedge clk);
 for(b=0;b<8;b=b+1) begin rx=value[b];repeat(8) @(negedge clk);end
 rx=1;repeat(8) @(negedge clk);
end endtask
task expect_byte;input [7:0] value;integer cycles;begin
 cycles=0;while(cursor>=count && cycles<20000) begin @(negedge clk);cycles=cycles+1;end
 if(cursor>=count) $fatal(1,"timeout at %0d",cursor);
 if(received[cursor]!==value) $fatal(1,"byte %0d: expected %02x got %02x",cursor,value,received[cursor]);
 cursor=cursor+1;
end endtask
reg [31:0] half_expected[0:65535];
reg [31:0] float_samples[0:4095];
reg [15:0] half_encoded[0:4095];
reg [15:0] bf_encoded[0:4095];
integer i;
initial begin
 // INITIAL
end
initial begin #100000000; $fatal(1,"simulation deadline"); end
endmodule
module TRELLIS_IO #(parameter DIR="OUTPUT") (inout B,input I,input T,output O);
assign B=T?1'bz:I;assign O=B;
endmodule
'''

def bf16(bits):
    if bits & 0x7f800000 == 0x7f800000 and bits & 0x7fffff:
        return (bits >> 16) | 0x40
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def half(bits):
    value=struct.unpack('<f',struct.pack('<I',bits))[0]
    if bits & 0x7f800000 == 0x7f800000 and bits & 0x7fffff:
        return ((bits >> 16) & 0x8000) | 0x7c00 | ((bits >> 13) & 0x3ff) | 0x200
    try:
        return struct.unpack('<H',struct.pack('<e',value))[0]
    except OverflowError:
        return 0xfc00 if bits >> 31 else 0x7c00


with tempfile.TemporaryDirectory(prefix='mimir-rtl-io-') as tmp:
    directory=Path(tmp)
    oracle=[]
    for value in range(65536):
        if value & 0x7c00 == 0x7c00 and value & 0x3ff:
            bits=((value & 0x8000) << 16) | 0x7f800000 | ((value & 0x3ff) << 13) | 0x400000
        else:
            bits=struct.unpack('<I',struct.pack('<f',struct.unpack('<e',struct.pack('<H',value))[0]))[0]
        oracle.append(bits)
    rng=random.Random(42)
    samples=[0,0x80000000,0x7f800000,0xff800000,0x7fc00000,0x7f800001,
             0x33800000,0x33000000,0x387fc000,0x38800000,0x477fe000,0x47800000,
             0x3f801000,0x3f803000,0x00000001,0x007fffff]
    samples += [rng.getrandbits(32) for _ in range(4096-len(samples))]
    for filename,values,width in [('half.txt',oracle,8),('float.txt',samples,8),('encoded.txt',map(half,samples),4),('bf.txt',map(bf16,samples),4)]:
        (directory/filename).write_text(''.join(f'{v:0{width}x}\n' for v in values))
    statements=[]
    def send(data):
        statements.extend(f'send(8\'h{x:02x});' for x in data)
    def expect(data):
        statements.extend(f'expect_byte(8\'h{x:02x});' for x in data)
    for memory,filename in [('half_expected','half.txt'),('float_samples','float.txt'),('half_encoded','encoded.txt'),('bf_encoded','bf.txt')]:
        statements.append(f'$readmemh("{directory/filename}",{memory});')
    statements += ['repeat(20) @(negedge clk);',
                   'for(i=0;i<65536;i=i+1) if(dut.f16_to_f32(i)!==half_expected[i]) $fatal(1,"F16 decode %04x",i);',
                   'for(i=0;i<4096;i=i+1) begin',
                   'if(dut.f32_to_f16(float_samples[i])!==half_encoded[i]) $fatal(1,"F16 encode %08x expected %04x got %04x",float_samples[i],half_encoded[i],dut.f32_to_f16(float_samples[i]));',
                   'if(dut.f32_to_bf16(float_samples[i])!==bf_encoded[i]) $fatal(1,"BF16 encode %08x",float_samples[i]);end']
    send(b'MIMIR?\x01\n');expect(b'MIMIRFPG'+struct.pack('<BBIH',1,3,15,64))
    send(b'@BEG\x02');expect(b'CACK')
    stream=bytearray()
    for mode in (0,1,2,3,4,5):
        if mode in (0,5):
            data=bytes(range(256));out=data
        elif mode in (1,3):
            bits=samples[:64];data=struct.pack('<'+'I'*64,*bits)
            out=struct.pack('<'+'H'*64,*[(bf16(v) if mode==1 else half(v)) for v in bits])
        else:
            shorts=[0,1,0x3ff,0x400,0x3c00,0x7bff,0x7c00,0xfc00,0x7e00,0x8000]*12+[0]*8
            data=struct.pack('<'+'H'*128,*shorts)
            out=struct.pack('<'+'I'*128,*[(v << 16 if mode==2 else oracle[v]) for v in shorts])
        stream.extend(data)
        send(b'@BUF'+bytes([mode])+struct.pack('<HI',len(data),zlib.crc32(data))+data)
        expect(b'XRES'+bytes([mode])+struct.pack('<H',len(out))+out+struct.pack('<I',zlib.crc32(out)))
    send(b'@END');expect(b'CRES\x02'+struct.pack('<QI',len(stream),zlib.crc32(stream)))
    send(b'@BEG\0');expect(b'CACK')
    # Corrupt input CRC: no output and no successful checkpoint operation.
    send(b'@BUF\0'+struct.pack('<HI',3,0)+b'bad')
    expect(b'XRES\xff\0\0'+bytes(4))
    send(b'@END');expect(b'CRES\0'+bytes(12))
    statements += ['$display("FPGA_IO_RTL_OK: 65536 F16 decodes, 4096 F16/BF16 encodes, UART copy/conversion, CRC controller");','$finish;']
    bench=directory/'test.v';bench.write_text(BASE.replace('// INITIAL','\n'.join(statements)))
    executable=directory/'test.out'
    subprocess.run(['iverilog','-g2012','-s','testbench','-o',str(executable),str(ROOT/'fpga/icesugar-pro/mimir_runtime/top.v'),str(bench)],check=True)
    subprocess.run(['vvp',str(executable)],check=True,timeout=30)
