const avail = Architectures.available();
console.log('[bridge-js] available count:', avail.length);

const dtypes = Architectures.dtypes();
console.log('[bridge-js] dtypes count:', dtypes.length);

// Normalisation: aliases est un array de strings (pas CSV comme avant)
const f32 = dtypes.find(d => d.name === 'float32');
console.log('[bridge-js] f32 name:', f32.name, '| bytes:', f32.bytes, '| kind:', f32.kind);
console.log('[bridge-js] f32 aliases (array):', Array.isArray(f32.aliases), '| count:', f32.aliases.length);
console.log('[bridge-js] f32 aliases[0]:', f32.aliases[0]);

const cfg = Architectures.default_config('basic_mlp');
console.log('[bridge-js] default_config keys:', Object.keys(cfg).length);

Model.create('basic_mlp', {});
Model.allocate_params();
Model.init_weights('he', 42);
const p = Model.total_params();
console.log('[bridge-js] total_params (live):', p);
console.log('[bridge-js] ok');
