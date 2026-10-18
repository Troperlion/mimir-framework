'use strict';
const assert=require('node:assert/strict');
const {assess}=require('../assets/skeleton-links.js');
const fixture=()=>({entities:[
  {id:'body',bbox:[0,0,.4,1],skeleton:{name:'custom',points:[
    {name:'root',position:[.2,.5],parent:null},
    {name:'hand',position:[.75,.5],parent:'root',entities:['held']},
    {name:'tail tip',position:[1,1],parent:'root'}]}},
  {id:'held',bbox:[.7,.4,.8,.6]},
  {id:'other',bbox:[.9,.4,1,.6]}
]});
let a=fixture(), before=JSON.stringify(a), results=assess(a,800,400);
assert.deepEqual(results.map(r=>r.status),['plausible','plausible']);
assert.equal(results[0].implicit,true);
assert.equal(results.length,2); // A custom appendage outside the owner is allowed.
assert.equal(JSON.stringify(a),before);
a.entities[0].skeleton.points[1].position=[.95,.5];
results=assess(a,800,400);assert.equal(results[1].status,'misplaced');assert.equal(results[1].suggestion,'other');assert.deepEqual(results[1].endpoint,[.8,.5]);
a.entities[0].skeleton.points[1].position=[.81,.5];
assert.equal(assess(a,800,400)[1].status,'plausible');
assert.equal(assess(a,800,400,0)[1].status,'misplaced');
a=fixture();a.entities.push({id:'overlap',bbox:[.72,.4,.78,.6]});
assert.equal(assess(a,800,400)[1].status,'ambiguous');
a.entities[0].skeleton.points[1].entities.push('overlap');
assert.deepEqual(assess(a,800,400).map(r=>r.status),['plausible','plausible','plausible']);
a=fixture();delete a.entities[0].skeleton.points[1].position;
assert.equal(assess(a,800,400)[1].status,'unknown');
a=fixture();delete a.entities[1].bbox;
assert.equal(assess(a,800,400)[1].status,'unknown');
a=fixture();a.entities[0].skeleton.points[1].entities=['absent'];
assert.equal(assess(a,800,400)[1].status,'unknown');
a=fixture();a.entities[0].skeleton.points[1].position=[.75,.65];
assert.equal(assess(a,1000,100)[1].status,'plausible');
assert.equal(assess(a,100,1000)[1].status,'misplaced');
assert.equal(assess(fixture(),0,0)[1].status,'unknown');
a=fixture();delete a.entities[0].skeleton.points[2].parent;a.entities[0].skeleton.edges=[{from:'root',to:'tail tip'}];
assert.equal(assess(a,800,400).length,2); // Free edges also define the skeleton roots.

console.log('PASS: plausible, ambiguous, misplaced, missing data, suggestions, custom tails, explicit multiple targets, tolerance, image aspect ratio, annotation immutability.');
