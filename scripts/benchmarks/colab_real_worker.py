"""Real MNIST training/inference adapters. One framework and device per process."""
import json
import os
from pathlib import Path
import sys
import time
import traceback


def make_adapter(framework, backend, config, initial):
    import numpy as np
    h=config['hidden'];lr=config['lr']
    if framework=='pytorch':
        if backend=='tpu':os.environ['PJRT_DEVICE']='TPU'
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        xm=None
        if backend=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
            device=torch.device('cuda:0')
        elif backend=='tpu':
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr
            if xr.device_type()!='TPU':raise RuntimeError('XLA is not TPU')
            device=xm.xla_device()
        else:device=torch.device('cpu')
        class Adapter:
            mode='torch.nn.Sequential + autograd + torch.optim.SGD'
            version=torch.__version__
            device_name=str(device)
            def __init__(self):
                self.model=torch.nn.Sequential(torch.nn.Linear(784,h),torch.nn.ReLU(),torch.nn.Linear(h,10)).to(device)
                self.optimizer=torch.optim.SGD(self.model.parameters(),lr=lr,momentum=0,weight_decay=0)
                if any(p.device.type!=device.type for p in self.model.parameters()):raise RuntimeError('Wrong parameter device')
                self.reset()
            def sync(self):
                if xm is not None:xm.mark_step();xm.wait_device_ops()
                elif backend=='cuda':torch.cuda.synchronize(device)
            def reset(self):
                with torch.no_grad():
                    for p,v in zip(self.model.parameters(),initial):p.copy_(torch.from_numpy(v.copy()).to(device))
                self.optimizer.zero_grad(set_to_none=True);self.sync()
            def train(self,x,y,probe=False):
                self.model.train();self.optimizer.zero_grad(set_to_none=True)
                xx=torch.from_numpy(x).to(device);yy=torch.from_numpy(y).to(device)
                out=self.model(xx);loss=((out-yy)**2).mean();loss.backward();self.optimizer.step();self.sync()
                return float(loss.detach().cpu()),out.detach().cpu().numpy() if probe else None
            def predict(self,x):
                self.model.eval()
                with torch.no_grad():out=self.model(torch.from_numpy(x).to(device))
                self.sync();return out.detach().cpu().numpy()
            def parameters(self):return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]
        return Adapter()
    if framework=='jax':
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_default_matmul_precision','highest')
        wanted={'cpu':'cpu','cuda':'gpu','tpu':'tpu'}[backend];device=jax.devices(wanted)[0]
        if device.platform!=wanted:raise RuntimeError('Wrong JAX device')
        def forward(p,x):return jax.nn.relu(x@p[0].T+p[1])@p[2].T+p[3]
        def train(p,x,y):
            def objective(v):
                out=forward(v,x);return jnp.mean((out-y)**2),out
            (loss,out),grads=jax.value_and_grad(objective,has_aux=True)(p)
            return tuple(v-lr*g for v,g in zip(p,grads)),loss,out
        compiled_train=jax.jit(train);compiled_predict=jax.jit(forward)
        class Adapter:
            mode='JAX persistent parameter pytree + jit(value_and_grad) + SGD'
            version=jax.__version__;device_name=str(device)
            def __init__(self):self.reset()
            def reset(self):
                self.p=tuple(jax.device_put(v.copy(),device) for v in initial);jax.block_until_ready(self.p)
            def train(self,x,y,probe=False):
                self.p,loss,out=compiled_train(self.p,jax.device_put(x,device),jax.device_put(y,device))
                jax.block_until_ready((self.p,loss))
                return float(loss),np.asarray(out) if probe else None
            def predict(self,x):
                out=compiled_predict(self.p,jax.device_put(x,device));out.block_until_ready()
                if next(iter(out.devices())).platform!=wanted:raise RuntimeError('Wrong JAX output device')
                return np.asarray(out)
            def parameters(self):return [np.asarray(v).copy() for v in self.p]
        return Adapter()
    if framework=='tensorflow':
        import tensorflow as tf
        tf.config.set_soft_device_placement(False)
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        strategy=None
        if backend=='tpu':
            resolver=tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy=tf.distribute.TPUStrategy(resolver)
            if strategy.num_replicas_in_sync!=1:raise RuntimeError('Exactly one TPU replica required')
            if any('TPU' not in d.upper() for d in strategy.extended.worker_devices):raise RuntimeError('TPUStrategy not placed on TPU')
            scope=strategy.scope()
        else:
            if backend=='cuda' and not tf.config.list_physical_devices('GPU'):raise RuntimeError('CUDA unavailable; no CPU fallback')
            if backend=='cuda':
                for d in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(d,True)
            scope=tf.device('/GPU:0' if backend=='cuda' else '/CPU:0')
        with scope:
            model=tf.keras.Sequential([tf.keras.Input(shape=(784,)),tf.keras.layers.Dense(h,activation='relu'),tf.keras.layers.Dense(10)])
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.0)
            optimizer.build(model.trainable_variables)
        wanted='TPU' if backend=='tpu' else 'GPU' if backend=='cuda' else 'CPU'
        if backend!='tpu' and any(wanted not in v.value.device.upper() for v in model.trainable_variables):raise RuntimeError('Wrong TensorFlow parameter device')
        def train(x,y):
            with tf.GradientTape() as tape:
                out=model(x,training=True);loss=tf.reduce_mean(tf.square(out-y))
            optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables))
            return loss,out
        def predict(x):return model(x,training=False)
        if strategy:
            @tf.function
            def compiled_train(x,y):return strategy.run(train,args=(x,y))
            @tf.function
            def compiled_predict(x):return strategy.run(predict,args=(x,))
        else:
            @tf.function
            def compiled_train(x,y):
                with tf.device('/GPU:0' if backend=='cuda' else '/CPU:0'):return train(x,y)
            @tf.function
            def compiled_predict(x):
                with tf.device('/GPU:0' if backend=='cuda' else '/CPU:0'):return predict(x)
        def local(value):return strategy.experimental_local_results(value)[0] if strategy else value
        class Adapter:
            mode='Keras Sequential + tf.function/GradientTape + Keras SGD'+(' + TPUStrategy' if strategy else '')
            version=tf.__version__;device_name=wanted
            def __init__(self):self.placement_checked=False;self.reset()
            def reset(self):
                model.set_weights([initial[0].T,initial[1],initial[2].T,initial[3]])
                optimizer.iterations.assign(0)
            def train(self,x,y,probe=False):
                xx,yy=tf.convert_to_tensor(x),tf.convert_to_tensor(y)
                loss,out=compiled_train(xx,yy)
                if strategy and not self.placement_checked:
                    graph=compiled_train.get_concrete_function(xx,yy).graph.as_graph_def()
                    nodes=list(graph.node)+[node for function in graph.library.function for node in function.node_def]
                    if not any('TPU' in node.op or '_tpu_replicate' in node.attr for node in nodes):raise RuntimeError('TPU execution graph proof missing')
                    self.placement_checked=True
                # Reading iterations after the compiled update is a completion barrier.
                _=optimizer.iterations.numpy()
                return float(local(loss).numpy()),local(out).numpy() if probe else None
            def predict(self,x):
                out=local(compiled_predict(tf.convert_to_tensor(x)))
                if not strategy and wanted not in out.device.upper():raise RuntimeError('Wrong TF output device')
                return out.numpy()
            def parameters(self):
                p=model.get_weights();return [p[0].T.copy(),p[1],p[2].T.copy(),p[3]]
        return Adapter()
    raise ValueError(framework)


def main(framework,backend,config_path,result_path):
    import numpy as np
    config=json.loads(Path(config_path).read_text());directory=Path(config['data_dir'])
    initial=[]
    for i,shape in enumerate([(config['hidden'],784),(config['hidden'],),(10,config['hidden']),(10,)]):
        initial.append(np.fromfile(directory/f'weights-{config["hidden"]}-{i}.f32',dtype='<f4').reshape(shape))
    data={split:(np.fromfile(directory/f'{split}.images.u8',dtype=np.uint8).reshape(-1,784),
                 np.fromfile(directory/f'{split}.labels.u8',dtype=np.uint8)) for split in ('train','valid','test')}
    batch=config['batch'];steps=config['steps'];eye=np.eye(10,dtype=np.float32)
    def get_batch(split,offset):
        images,labels=data[split];index=(np.arange(batch)+offset)%len(images)
        return images[index].astype(np.float32)/np.float32(255),eye[labels[index]]
    adapter=make_adapter(framework,backend,config,initial)
    report=dict(framework=framework,backend=backend,case=config['case'],status='measured',
                mode=adapter.mode,version=adapter.version,device=adapter.device_name,
                optimizer='SGD(momentum=0, weight_decay=0)',loss='mean squared error over batch*10')
    first_x,first_y=get_batch('train',0)
    started=time.perf_counter();first_loss,first_prediction=adapter.train(first_x,first_y,True)
    report['cold_step_ms']=(time.perf_counter()-started)*1000
    flat=np.concatenate([v.ravel() for v in adapter.parameters()]);indices=config['probe_indices']
    report.update(first_loss=first_loss,first_prediction=first_prediction.ravel().tolist(),first_parameter_probe=flat[indices].tolist())
    adapter.reset()
    # Compile inference before timed training; this warmup never changes weights.
    started=time.perf_counter();adapter.predict(first_x)
    report['cold_predict_ms']=(time.perf_counter()-started)*1000
    def evaluate(split):
        images,labels=data[split];correct=0;loss=0.;latencies=[]
        for offset in range(0,len(images),batch):
            count=min(batch,len(images)-offset)
            started=time.perf_counter();x,y=get_batch(split,offset);out=adapter.predict(x)
            latencies.append((time.perf_counter()-started)*1000)
            correct+=int((out[:count].argmax(1)==labels[offset:offset+count]).sum())
            loss+=float(np.square(out[:count]-y[:count]).sum())
        return dict(accuracy=correct/len(images),mse=loss/(len(images)*10),samples_ms=latencies,examples=len(images))
    samples=[];losses=[];history=[];time_to_goal=None
    started=time.perf_counter()
    for step in range(steps):
        tick=time.perf_counter();x,y=get_batch('train',step*batch);loss,_=adapter.train(x,y)
        samples.append((time.perf_counter()-tick)*1000);losses.append(loss)
        if not np.isfinite(loss):raise RuntimeError('Non-finite training loss')
        if (step+1)%config['eval_every']==0 or step+1==steps:
            validation=evaluate('valid');elapsed=time.perf_counter()-started
            history.append(dict(step=step+1,elapsed_s=elapsed,accuracy=validation['accuracy'],mse=validation['mse']))
            if time_to_goal is None and validation['accuracy']>=config['target_accuracy']:time_to_goal=elapsed
            progress=dict(steps_completed=step+1,history=history,status='running')
            Path(result_path+'.progress.json').write_text(json.dumps(progress,allow_nan=False))
    report.update(train_wall_s=time.perf_counter()-started,train_step_samples_ms=samples,train_losses=losses,
                  history=history,time_to_goal_s=time_to_goal,steps_completed=steps,examples_trained=steps*batch)
    started=time.perf_counter();report['test']=evaluate('test');report['inference_wall_s']=time.perf_counter()-started
    report['final_parameter_probe']=np.concatenate([p.ravel() for p in adapter.parameters()])[indices].tolist()
    return report


if __name__=='__main__':
    started=time.perf_counter();framework,backend,config_path,result_path=sys.argv[1:]
    try:
        report=main(framework,backend,config_path,result_path)
    except Exception as exc:
        report=dict(framework=framework,backend=backend,status='error',error=str(exc),traceback=traceback.format_exc())
    report['worker_wall_s']=time.perf_counter()-started
    Path(result_path).write_text(json.dumps(report,allow_nan=False))
