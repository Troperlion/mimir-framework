/* Geometry-only preview; never changes annotations or claims semantic certainty. */
(function (root) {
  'use strict';
  const colors = {plausible:'#7ce1ba', ambiguous:'#ffd58a', misplaced:'#ffa591', unknown:'#a1b0bf'};
  function nearest(position, box) {
    return [Math.max(box[0],Math.min(box[2],position[0])),Math.max(box[1],Math.min(box[3],position[1]))];
  }
  function assess(annotation, width, height, tolerance = .03) {
    const results=[], entities=annotation?.entities || [];
    const validPosition = p => Array.isArray(p) && p.length===2 && p.every(v=>Number.isFinite(v) && v>=0 && v<=1);
    const validBox = b => Array.isArray(b) && b.length===4 && b.every(v=>Number.isFinite(v) && v>=0 && v<=1) && b[0]<b[2] && b[1]<b[3];
    const diagonal=Math.hypot(width,height);
    const distance=(p,b)=>{const q=nearest(p,b);return Math.hypot((p[0]-q[0])*width,(p[1]-q[1])*height)/diagonal;};
    for(const owner of entities) for(const point of owner.skeleton?.points || []) {
      // Root points represent the attachment of the skeleton to its owner. Other
      // points can extend outside the body, including tails and custom appendages.
      const rootPoint=!point.parent && !(owner.skeleton.edges || []).some(edge=>edge.to===point.name);
      const targets=[...new Set([...(rootPoint ? [owner.id] : []),...(point.entities || [])])];
      for(const targetId of targets) {
        const target=entities.find(e=>e.id===targetId);
        const result={owner:owner.id, point:point.name, target:targetId, implicit:!(point.entities || []).includes(targetId), status:'unknown', reason:'', position:point.position, endpoint:null, suggestion:null};
        if(!target)result.reason='Entité cible introuvable.';
        else if(!validPosition(point.position))result.reason='Placez ce point pour vérifier son attache.';
        else if(!validBox(target.bbox))result.reason='Dessinez une boîte pour l’entité cible.';
        else if(!(width>0 && height>0 && Number.isFinite(diagonal)))result.reason='Chargez l’image pour vérifier les distances.';
        else {
          const d=distance(point.position,target.bbox);result.distance=d;result.endpoint=nearest(point.position,target.bbox);
          const alternatives=entities.filter(e=>e.id!==owner.id && !targets.includes(e.id) && validBox(e.bbox))
            .map(e=>({id:e.id,distance:distance(point.position,e.bbox)})).filter(e=>e.distance<=tolerance).sort((a,b)=>a.distance-b.distance);
          if(d>tolerance) {
            result.status='misplaced';result.reason=`Point trop éloigné de la boîte cible (${(d*100).toFixed(1)} % de la diagonale ; tolérance ${(tolerance*100).toFixed(1)} %).`;
            if(alternatives.length) {result.suggestion=alternatives[0].id;result.reason+=` Cible proche possible : ${alternatives[0].id}.`;}
          } else if(alternatives.some(e=>e.distance<=d)) {
            result.status='ambiguous';result.reason=`Plusieurs boîtes sont aussi proches : ${alternatives.map(e=>e.id).join(', ')}. Vérifiez l’identité de la cible.`;
          } else {
            result.status='plausible';result.reason=d===0?'Point situé dans la boîte cible.':'Point proche du bord de la boîte cible.';
          }
        }
        results.push(result);
      }
    }
    return results;
  }
  const api={assess,colors};root.SkeletonLinks=api;
  if(typeof module!=='undefined' && module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
