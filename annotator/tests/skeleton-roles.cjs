'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require.resolve('../assets/app.js'),'utf8');
const context=vm.createContext({changed(){},renderEntities(){},message(){}});
vm.runInContext(source.slice(source.indexOf('const jointRoles ='),source.indexOf('function renderSkeleton(')),context);
vm.runInContext(`
for(const [name,expected] of [['tête','tête'],['main gauche','saisie'],['pied droit','support'],['hanche gauche','articulation majeure'],['queue','point']]) {
  if(jointRole({name})!==expected) throw Error(name);
}
if(jointRole({name:'main gauche',role:'support'})!=='support') throw Error('Explicit role ignored');
var head={name:'tête',position:null}, entity={skeleton:{points:[head]}};
placeFace(entity,head);
`,context);
assert.equal(context.entity.skeleton.points.length,7);
assert.equal(context.skeletonTarget.face.length,6);
assert.ok(context.entity.skeleton.points.slice(1).every(p=>p.parent==='tête' && p.role==='détail facial'));
vm.runInContext('placeFace(entity,head)',context);
assert.equal(context.entity.skeleton.points.length,7,'Repositioning reuses facial points');
// Exercise the same placement branch used by a canvas click.
const start=source.indexOf('    const {entity, joint, face} = skeletonTarget;');
const end=source.indexOf('    skeletonTarget = null; draw(); return;',start);
context.p=[.5,.5];context.img={naturalWidth:800,naturalHeight:400};context.annotation={entities:[context.entity]};
vm.runInContext('{'+source.slice(start,end)+'}',context);
assert.deepEqual(Array.from(context.entity.skeleton.points[0].position),[.5,.5]);
assert.ok(context.entity.skeleton.points.every(p=>p.position.every(v=>v>=0 && v<=1)));
assert.ok(context.entity.skeleton.points[1].position[0]<context.entity.skeleton.points[2].position[0]);
console.log('PASS: joint roles, explicit overrides, reusable facial layer, canvas placement and normalized positions.');
