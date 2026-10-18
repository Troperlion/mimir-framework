#pragma once
namespace mimir::viz::web {
inline const char* page=R"HTML(<!doctype html><meta charset="utf-8"><title>Mímir Viz</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#20242b;color:#dfe5ee;font:12px system-ui,sans-serif}
header{height:34px;display:flex;align-items:center;gap:5px;padding:0 8px;background:#282d36;border-bottom:1px solid #101318;position:sticky;top:0;z-index:2}
.brand{color:#e89144;font-weight:700;margin-right:14px;letter-spacing:1px}button{font:inherit;font-weight:500;color:#dfe5ee;border:1px solid #4d5562;border-radius:5px;padding:4px 10px;background:#313843;cursor:pointer;box-shadow:0 2px 2px #0002;transition:background .12s,border-color .12s}button:hover{background:#3e4b5c;border-color:#7e97b2}button:active{background:#26394d;box-shadow:inset 0 1px 3px #0004}button:focus-visible{outline:2px solid #80b4e9;outline-offset:1px}#status{margin-left:auto;color:#aaa;padding:0 8px}#stop{color:#edaa79}main{overflow:auto}canvas{display:block;outline:none}footer{padding:5px 10px;background:#282d36;border-top:1px solid #101318;color:#aaa}
</style>
<header><span class="brand">MÍMIR</span><button data-key="72" title="H">Aide</button><button data-key="67" title="C">Propriétés</button><button data-key="78" title="N">Graphe</button><button data-key="69" title="E">Capture</button><button data-key="82" title="R">Actualiser</button><span id="status" role="status">Connexion…</span><button id="stop">Fermer</button></header><main><canvas tabindex="0" aria-label="Espace de travail du modèle"></canvas></main><footer>Modèle · Tenseurs · Validation <span style="float:right">Canvas2D / WEB</span></footer>
<script>
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),status=document.querySelector('#status');
const base=location.pathname.replace(/\/$/,'');let active=true,chain=Promise.resolve(),x=0,y=0;
function send(t,v=0,m=0,d=0){chain=chain.then(async()=>{const r=await fetch(`${base}/event?t=${t}&x=${x}&y=${y}&v=${v}&m=${m}&d=${d}`,{method:'POST'});if(!r.ok)throw Error('événement refusé');}).catch(()=>{status.textContent='Connexion perdue';active=false;});}
function point(e){const r=canvas.getBoundingClientRect();x=Math.round((e.clientX-r.left)*canvas.width/r.width);y=Math.round((e.clientY-r.top)*canvas.height/r.height);}
canvas.onpointermove=e=>{point(e);send(2)};
canvas.onpointerdown=e=>{if(e.button>2)return;canvas.focus();canvas.setPointerCapture(e.pointerId);point(e);send(3,e.button===2?1:e.button===1?2:0);e.preventDefault()};
canvas.onpointerup=e=>{point(e);send(4,e.button===2?1:e.button===1?2:0);if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId)};
canvas.onpointercancel=()=>send(4);canvas.oncontextmenu=e=>e.preventDefault();
canvas.addEventListener('wheel',e=>{point(e);send(5,0,0,-e.deltaY/(e.deltaMode===0?100:3));e.preventDefault()},{passive:false});
function key(e){const k=e.key;const map={Escape:256,Enter:257,Tab:258,Backspace:259,ArrowLeft:260,ArrowRight:261,ArrowUp:262,ArrowDown:263};if(map[k])return map[k];if(/^F([1-9]|1[012])$/.test(k))return 280+Number(k.slice(1));if(/^[a-z0-9]$/i.test(k))return k.toUpperCase().charCodeAt(0);return 0;}
canvas.onkeydown=e=>{send(6,key(e),(e.ctrlKey?1:0)|(e.shiftKey?2:0)|(e.altKey?4:0)|(e.metaKey?8:0));if(e.key==='Backspace')send(8,8);else if(!e.ctrlKey&&!e.metaKey&&!e.isComposing&&[...e.key].length===1)send(8,e.key.codePointAt(0));e.preventDefault()};
canvas.onkeyup=e=>{send(7,key(e));e.preventDefault()};canvas.oncompositionend=e=>{for(const c of e.data)send(8,c.codePointAt(0))};
window.onblur=()=>{send(4);send(9)};
document.querySelector('#stop').onclick=()=>{send(1);active=false;status.textContent='Viz fermée'};
document.querySelectorAll('[data-key]').forEach(button=>button.onclick=()=>{const k=Number(button.dataset.key);send(6,k);send(7,k);canvas.focus()});
async function refresh(){if(!active)return;try{const r=await fetch(base+'/frame');if(r.status===204){setTimeout(refresh,50);return}if(!r.ok)throw Error();const frame=await r.json();if(canvas.width!==frame.width||canvas.height!==frame.height){canvas.width=frame.width;canvas.height=frame.height}await window.renderVizScene(canvas,frame.scene);canvas.style.cursor=['default','pointer','crosshair','nwse-resize'][Number(r.headers.get('X-Viz-Cursor'))]||'default';status.textContent='Connecté';setTimeout(refresh,33)}catch(e){status.textContent='Connexion perdue';setTimeout(refresh,1000)}}
const renderer=document.createElement('script');renderer.src=base+'/renderer.js';renderer.onload=()=>{canvas.focus();refresh()};renderer.onerror=()=>{status.textContent='Rendu indisponible'};document.head.appendChild(renderer);
</script>)HTML";
}
