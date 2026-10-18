#!/usr/bin/env node
// Real browser integration; Node 22 and Chrome/Chromium, no npm dependencies.
import {spawn} from 'node:child_process';
import {mkdtemp,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import assert from 'node:assert/strict';
const executable=process.argv[2];
if(!executable)throw Error('Usage: node Tests/viz_web_canvas_smoke.mjs /path/to/mimir_viz_smoke');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(predicate){for(let i=0;i<150;i++){const value=await predicate();if(value)return value;await delay(100)}throw Error('Timed out')}
const profile=await mkdtemp(join(tmpdir(),'mimir-canvas-'));
const model=spawn(executable,['30']);let log='';model.stdout.on('data',d=>log+=d);model.stderr.on('data',d=>log+=d);
let browser,ws;let browserLog='';
try{
 const url=await until(()=>log.match(/http:\/\/127\.0\.0\.1:\d+\/[a-f0-9]+\//)?.[0]);
 browser=spawn(process.env.CHROME||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--remote-debugging-port=0','--remote-allow-origins=*',`--user-data-dir=${profile}`,'about:blank']);
 browser.stderr.on('data',d=>browserLog+=d);
 const port=await until(()=>browserLog.match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)/)?.[1]);
 const page=await (await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`,{method:'PUT'})).json();
 ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 let sequence=0;const pending=new Map();
 ws.onmessage=event=>{const message=JSON.parse(event.data);if(message.id){const {resolve,reject}=pending.get(message.id);pending.delete(message.id);message.error?reject(Error(JSON.stringify(message.error))):resolve(message.result)}};
 function cdp(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
 async function evaluate(expression){const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
 await until(()=>evaluate(`document.querySelector('#status')?.textContent==='Connecté'`));
 const samples=await evaluate(`[[20,30],[110,30],[250,190],[290,220]].map(([x,y])=>Array.from(document.querySelector('canvas').getContext('2d').getImageData(x,y,1,1).data))`);
 assert.deepEqual(samples,[[231,42,63,255],[45,210,90,255],[90,100,240,255],[17,34,51,255]]);
 const screenshot=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(join(tmpdir(),'mimir-web-native.png'),Buffer.from(screenshot.data,'base64'));
 await cdp('Input.dispatchMouseEvent',{type:'mousePressed',x:20,y:64,button:'left',clickCount:1});
 await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x:20,y:64,button:'left',clickCount:1});
 await evaluate(`canvas.dispatchEvent(new KeyboardEvent('keydown',{key:'h',bubbles:true}));canvas.dispatchEvent(new KeyboardEvent('keyup',{key:'h',bubbles:true}));document.querySelector('[data-key="82"]').click()`);
 await until(()=>evaluate(`document.querySelector('canvas').width===400`));
 assert(log.includes('DOWN 20 30')&&log.includes('TEXT 104'),log);
 const response=await fetch(url+'event?t=bad',{method:'POST'});assert.equal(response.status,400);
 await evaluate(`document.querySelector('#stop').click()`);
 await until(()=>model.exitCode!==null);
 assert.equal(model.exitCode,0,log);assert(log.includes('CLOSE'),log);
 console.log('WEB real Canvas2D pixels, textures, clipping, DOM toolbar, mouse, keyboard, resize, close OK');
}finally{
 ws?.close();if(model.exitCode===null)model.kill();if(browser&&browser.exitCode===null){browser.kill();await delay(500)}
 await rm(profile,{recursive:true,force:true,maxRetries:5,retryDelay:100});
}
