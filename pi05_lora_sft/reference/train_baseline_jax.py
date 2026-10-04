"""Full-parameter OpenPI flow matching with global-batch gradient accumulation.
The model's compute_loss is upstream, unchanged. Clip/AdamW/EMA run once per global batch.
"""
import argparse,dataclasses,functools,gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'configs'),str(ROOT/'openpi/scripts')]
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from openpi.training import sharding
import sequential_checkpoints as checkpoints
from notify_status import notify
from openpi.shared import array_typing as at
from pi05_ecot_libero90_action_only import get_config,RUN
from dataset_adapter import Libero90Dataset,training_batch,make_transform
import train as upstream

def accumulated_step(config,microbatch,rng,state,batch):
    observation,actions=batch
    accum_steps=actions.shape[0]//microbatch
    assert actions.shape[0]%microbatch==0
    obs=jax.tree.map(lambda x:x.reshape((accum_steps,microbatch)+x.shape[1:]),observation)
    act=actions.reshape((accum_steps,microbatch)+actions.shape[1:])
    train_rng=jax.random.fold_in(rng,state.step)
    diff_state=nnx.DiffState(0,config.trainable_filter)
    def loss_fn(model,key,o,a):
        return jnp.mean(model.compute_loss(key,o,a,train=True))
    def gradient(i):
        model=nnx.merge(state.model_def,state.params);model.train()
        return nnx.value_and_grad(loss_fn,argnums=diff_state)(model,jax.random.fold_in(train_rng,i),jax.tree.map(lambda x:x[i],obs),act[i])
    # First microbatch initializes buffers, preventing an extra full gradient-sized zero allocation.
    loss,grad=gradient(0)
    def body(carry,i):
        l,g=gradient(i)
        return (carry[0]+l,jax.tree.map(lambda a,b:a+b,carry[1],g)),None
    (loss,grad),_=jax.lax.scan(body,(loss,grad),jnp.arange(1,accum_steps))
    loss=loss/accum_steps
    grad=jax.tree.map(lambda x:x/accum_steps,grad)
    params=state.params.filter(config.trainable_filter)
    updates,opt_state=state.tx.update(grad,state.opt_state,params)
    model=nnx.merge(state.model_def,state.params)
    nnx.update(model,optax.apply_updates(params,updates))
    params=nnx.state(model)
    ema=jax.tree.map(lambda old,new:state.ema_decay*old+(1-state.ema_decay)*new,state.ema_params,params)
    new=dataclasses.replace(state,step=state.step+1,params=params,opt_state=opt_state,ema_params=ema)
    return new,{'flow_matching_loss':loss,'gradient_norm':optax.global_norm(grad)}

def gpu_metrics():
    import subprocess
    try:
        line=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
        a,b,c=map(float,line.split(','));return {'gpu_utilization':a,'gpu_memory_mib':b,'gpu_total_mib':c}
    except Exception:return {}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tiny','smoke','train'],required=True)
    p.add_argument('--microbatch',type=int,default=32);p.add_argument('--resume',action='store_true');p.add_argument('--restore-only',action='store_true')
    args=p.parse_args();cfg=get_config()
    assert cfg.batch_size%args.microbatch==0
    if args.phase=='train':
        for filename in ['adapter_validation.json','base_validation.json','tiny_validation.json','smoke_validation.json','accumulation_validation.json']:
            report=json.loads((ROOT/'reports'/filename).read_text());assert report['status']=='passed',filename
        assert json.loads((ROOT/'reports/smoke_validation.json').read_text())['reloaded']
        smoke=json.loads((ROOT/'reports/smoke_validation.json').read_text())
        assert smoke['effective_batch_size']==cfg.batch_size and smoke['microbatch']==args.microbatch
    if args.phase=='tiny':
        # Diagnostic-only schedule; the full run retains its configured warmup.
        cfg=dataclasses.replace(cfg,num_train_steps=20,lr_schedule=dataclasses.replace(cfg.lr_schedule,warmup_steps=1))
    elif args.phase=='smoke':
        # Multiple 100-step runs already validated finite full-model gradients.
        # Keep retries short while exercising the unresolved full checkpoint
        # save and reload path with a nonzero AdamW/scheduler state.
        cfg=dataclasses.replace(cfg,num_train_steps=5)
    run_name=RUN if args.phase=='train' else RUN+'-'+args.phase
    path=ROOT/'checkpoints'/run_name
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'data/cache/jax'))
    rng=jax.random.key(cfg.seed);train_rng,init_rng=jax.random.split(rng)
    mesh=sharding.make_mesh(cfg.fsdp_devices)
    manager,resuming=checkpoints.initialize_checkpoint_dir(path,keep_period=cfg.keep_period,overwrite=False,resume=args.resume)
    ds=Libero90Dataset();transform=make_transform(cfg)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    class Assets:
        def data_config(self):return dc
    assets=Assets()
    state,state_sharding=upstream.init_train_state(cfg,init_rng,mesh,resume=resuming)
    if resuming:state=checkpoints.restore_state(manager,state,assets)
    jax.block_until_ready(state)
    norm_path=cfg.assets_dirs/'ecot_libero90/norm_stats.json'
    provenance={'phase':args.phase,'config':dataclasses.asdict(cfg),'microbatch':args.microbatch,
                'accumulation_steps':cfg.batch_size//args.microbatch,'effective_batch_size':cfg.batch_size,
                'train_rng':jax.random.key_data(train_rng).tolist(),
                'sampler':'numpy.default_rng(SeedSequence([seed, optimizer_step])); uniform valid chunks with replacement',
                'norm_stats_sha256':digest(norm_path),'dataset_manifest_sha256':digest(ROOT/'reports/dataset_checksums.json'),
                'episode_index_sha256':digest(ROOT/'data/index/episodes.json'),'trainer_sha256':digest(Path(__file__)),
                'checkpoint_code_sha256':digest(ROOT/'scripts/sequential_checkpoints.py'),
                'openpi_commit':'215abfb217dbac7d5f1273282331b9b1866c0479'}
    if resuming:
        saved=json.loads((path/str(int(state.step))/'resume_metadata.json').read_text())
        for key in ['microbatch','accumulation_steps','effective_batch_size','norm_stats_sha256','dataset_manifest_sha256','episode_index_sha256','trainer_sha256','checkpoint_code_sha256','train_rng','phase']:
            assert saved[key]==provenance[key],f'Resume mismatch: {key}'
    else:
        (path/'configuration.json').write_text(json.dumps(provenance,default=str,indent=2))
    if args.restore_only:
        step=int(state.step);assert step==cfg.num_train_steps
        leaves=jax.tree.leaves(state.params)
        assert all(bool(jnp.isfinite(x).all()) for x in leaves)
        rp=ROOT/'reports'/f'{args.phase}_validation.json'
        d=json.loads(rp.read_text());d.update(reloaded=True,reloaded_step=step)
        rp.write_text(json.dumps(d,indent=2));print('Restored full state at',step,flush=True);return
    wandb_id_path=path/'wandb_id.txt'
    kwargs=dict(project=cfg.project_name,entity='yus047-',name=run_name,config=provenance,dir=str(ROOT/'logs'))
    if resuming and wandb_id_path.exists():kwargs.update(id=wandb_id_path.read_text().strip(),resume='allow')
    try:run=wandb.init(**kwargs,settings=wandb.Settings(init_timeout=30))
    except Exception:
        kwargs.pop('resume',None);run=wandb.init(**kwargs,mode='offline')
    wandb_id_path.write_text(run.id)
    (ROOT/'reports'/f'{args.phase}_wandb.json').write_text(json.dumps({'url':run.url,'id':run.id,'mode':run.settings.mode,'directory':run.dir},indent=2))
    direct=args.microbatch==cfg.batch_size
    step_fn=functools.partial(upstream.train_step,cfg) if direct else functools.partial(accumulated_step,cfg,args.microbatch)
    fn=jax.jit(step_fn,donate_argnums=(1,))
    fixed_ids=np.resize(ds.indices(0,8,cfg.seed),cfg.batch_size) if args.phase=='tiny' else None
    tiny_probe=None;initial_probe_loss=None
    if args.phase=='tiny':
        tiny_probe,_=training_batch(ds,cfg,0,indices=fixed_ids[:args.microbatch],transform=transform)
        def probe(params,batch):
            model=nnx.merge(state.model_def,params)
            return jnp.mean(model.compute_loss(jax.random.key(12345),batch[0],batch[1],train=False))
        probe_fn=jax.jit(probe)
        initial_probe_loss=float(probe_fn(state.params,tiny_probe))
    history=[];start=int(state.step);started=time.time()
    for i in range(start,cfg.num_train_steps):
        before=time.time();batch,ids=training_batch(ds,cfg,i,indices=fixed_ids,transform=transform)
        with sharding.set_mesh(mesh):state,info=fn(train_rng,state,batch)
        info=jax.device_get(info);elapsed=time.time()-before
        if direct:info={'flow_matching_loss':info['loss'],'gradient_norm':info['grad_norm']}
        assert np.isfinite(float(info['flow_matching_loss'])) and np.isfinite(float(info['gradient_norm'])),info
        metrics={k:float(v) for k,v in info.items()}
        metrics.update(optimizer_step=i+1,learning_rate=float(cfg.lr_schedule.create()(i)),sampled_examples=(i+1)*cfg.batch_size,
                       equivalent_epochs=(i+1)*cfg.batch_size/len(ds),examples_per_second=cfg.batch_size/elapsed,
                       step_seconds=elapsed,ema_enabled=True,ema_decay=cfg.ema_decay,**gpu_metrics())
        with (ROOT/'logs'/f'{run_name}.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        run.log(metrics,step=i+1);history.append(metrics)
        print(json.dumps(metrics),flush=True)
        if args.phase=='train' and i==start:
            notify('π0.5 FULL TRAINING STARTED',f'First optimizer step completed: {i+1}/12000. Physical batch 32. W&B: {run.url}')
        (ROOT/'reports'/f'{args.phase}_progress.json').write_text(json.dumps({'run_name':run_name,**metrics},indent=2))
        if args.phase!='tiny' and ((i+1)%cfg.save_interval==0 or i+1==cfg.num_train_steps):
            step=i+1;checkpoints.save_state(manager,state,assets,step);manager.wait_until_finished()
            metadata={**provenance,'optimizer_step':step,'next_sampler_step':step,'ema_export':'params contains EMA; train_state contains live params and AdamW state, including schedule counter'}
            (path/str(step)/'resume_metadata.json').write_text(json.dumps(metadata,default=str,indent=2))
            link=path/f'step_{step:06d}'
            if not link.exists():link.symlink_to(str(step),target_is_directory=True)
    summary={'status':'passed','steps':int(state.step),'effective_batch_size':cfg.batch_size,
             'microbatch':args.microbatch,'elapsed_seconds':time.time()-started,'reloaded':False,
             'first_loss':history[0]['flow_matching_loss'],'last_loss':history[-1]['flow_matching_loss'],
             'checkpoint':str(path/f'step_{int(state.step):06d}') if args.phase!='tiny' else None}
    if args.phase=='tiny':
        final_probe_loss=float(probe_fn(state.params,tiny_probe))
        summary.update(initial_probe_loss=initial_probe_loss,final_probe_loss=final_probe_loss,
                       diagnostic_schedule='1-step warmup, configured 2e-5 peak; not used for full run')
        summary['status']='passed' if final_probe_loss<initial_probe_loss else 'failed'
    (ROOT/'reports'/f'{args.phase}_validation.json').write_text(json.dumps(summary,indent=2))
    assert summary['status']=='passed',summary
    manager.wait_until_finished();run.finish()
if __name__=='__main__':main()
