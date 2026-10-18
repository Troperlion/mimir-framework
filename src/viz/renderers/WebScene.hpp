#pragma once
namespace mimir::viz::web {
inline constexpr const char* renderer = R"JS(
// Native Canvas2D renderer: the server sends geometry and image assets, never UI screenshots.
window.renderVizScene=async function(canvas,scene){
 const ctx=canvas.getContext('2d');
 const rgba=v=>`rgba(${v[0]},${v[1]},${v[2]},${v[3]/255})`;
 const images=await Promise.all(scene.images.map(data=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src='data:image/png;base64,'+data})));
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle=rgba(scene.background);ctx.fillRect(0,0,canvas.width,canvas.height);
 for(const cmd of scene.commands){
  ctx.save();const v=cmd.view,x=v[4]*canvas.width,y=v[5]*canvas.height,w=v[6]*canvas.width,h=v[7]*canvas.height;
  ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();ctx.translate(x,y);ctx.scale(w/Math.max(1,v[2]),h/Math.max(1,v[3]));ctx.translate(-v[0],-v[1]);
  if(cmd.kind==='rect'){
   const r=cmd.rect;ctx.fillStyle=rgba(cmd.fill);ctx.fillRect(...r);const t=Math.abs(cmd.thickness);
   if(t){ctx.lineWidth=t;ctx.strokeStyle=rgba(cmd.outline);ctx.strokeRect(r[0]+t/2,r[1]+t/2,Math.max(0,r[2]-t),Math.max(0,r[3]-t))}
  }else if(cmd.kind==='text'){
   ctx.font=`${cmd.bold?'bold ':''}${cmd.size}px "Open Sans",sans-serif`;ctx.fillStyle=rgba(cmd.fill);ctx.fillText(cmd.text,cmd.position[0],cmd.position[1]+cmd.size*.82);
  }else if(cmd.kind==='line'){
   ctx.lineWidth=1;for(let i=1;i<cmd.vertices.length;i++){const a=cmd.vertices[i-1],b=cmd.vertices[i];ctx.strokeStyle=rgba(b.slice(2));ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}
  }else if(cmd.kind==='image'&&images[cmd.asset]){
   ctx.translate(...cmd.position);ctx.scale(...cmd.scale);ctx.globalAlpha=cmd.alpha;ctx.imageSmoothingEnabled=cmd.smooth;ctx.drawImage(images[cmd.asset],0,0);
  }
  ctx.restore();
 }
};
)JS";
}
