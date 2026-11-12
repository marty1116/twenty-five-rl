"""Independent acceptance gates for the approved October 26 repair decisions."""
from __future__ import annotations

import ast
from collections import defaultdict
import contextlib
import copy
import io
import inspect
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from twenty_five import Card, Phase, TwentyFiveGame, InvalidAction
from twenty_five.envs import TwentyFiveAECEnv, TwentyFiveEnv
from twenty_five.gui import GameSession, build_opponents, _asset_bytes
from twenty_five.observations import PlayerView, NEURAL_OBSERVATION_SIZE, OBSERVATION_LAYOUT
from twenty_five.training.replay import ReplayBuffer
from tests.fixtures import rob_deck


class PrivateInterfacesTests(unittest.TestCase):
    def test_recipient_events_and_own_knowledge(self):
        env=TwentyFiveAECEnv(4)
        env.reset(seed=1,options={'initial_deck':rob_deck(2)})
        discard=env.game.legal_cards()[0].index
        env.step(discard)
        for seat,name in enumerate(env.possible_agents):
            self.assertEqual(env.infos[name]['info_schema_version'],2)
            self.assertEqual(env.infos[name]['transition']['action'],discard if seat==2 else None)
            self.assertEqual(env.raw_view(name).own_rob_discard,Card.from_index(discard) if seat==2 else None)

    def test_gym_reset_and_step_redact_opponent_robs(self):
        env=TwentyFiveEnv(4,controlled_seat=0,illegal_action_mode='raise')
        obs,info=env.reset(seed=1,options={'initial_deck':rob_deck(2)})
        self.assertTrue(any(e['phase'] is Phase.ROB for e in info['events']))
        for seed in range(10):
            obs,info=env.reset(seed=seed)
            while True:
                for e in info['events']:
                    if e['phase'] is Phase.ROB and e['actor']!=0:
                        self.assertIsNone(e['action'])
                obs,_,done,truncated,info=env.step(int(np.flatnonzero(obs['action_mask'])[0]))
                self.assertFalse(truncated)
                if done:break

    def test_private_seed_debug_label_and_postgame_replay(self):
        session=GameSession(build_opponents([]),players=2,seed=None)
        self.assertNotIn('seed',session.state()['game'])
        self.assertFalse(session.state()['game']['debug_seeded'])
        with self.assertRaises(InvalidAction):session.replay_bytes()
        while not session.game.terminal:session.act(session.game.legal_cards()[0].index)
        self.assertEqual(session.state()['game']['seed'],session.seed)
        replay=json.loads(session.replay_bytes())
        restored=TwentyFiveGame.from_dict(replay['initial'])
        for event in replay['actions']:restored.step(event['seat'],event['action'])
        self.assertEqual(restored.to_json(),session.game.to_json())
        debug=GameSession(build_opponents([]),players=2,seed=11)
        self.assertTrue(debug.state()['game']['debug_seeded'])
        self.assertEqual(debug.state()['game']['seed'],11)
        self.assertIn(b'Private deal',_asset_bytes('app.js'))
        self.assertIn(b'reveals deal',_asset_bytes('index.html'))

    def test_live_cli_hides_ai_discard(self):
        from twenty_five.training import play
        game=TwentyFiveGame(4,initial_deck=rob_deck(2))
        text=io.StringIO()
        with patch.object(play,'TwentyFiveGame',return_value=game),patch.object(play,'_human_action',side_effect=KeyboardInterrupt),contextlib.redirect_stdout(text):
            play.play_human_game(4,None,'counting',0,1,0)
        self.assertNotIn(f'P2 plays {game.discards[0]}',text.getvalue())
        self.assertIn('face down',text.getvalue())

    def test_invalid_opponent_truncation_propagates(self):
        class Broken:
            name='invalid'
            def act(self,view,rng):return 52
        env=TwentyFiveEnv(4,controlled_seat=0,opponents={s:Broken() for s in range(1,4)})
        obs,info=env.reset(seed=1)
        if info['truncated']:
            self.assertFalse(obs['action_mask'].any())
        else:
            obs,_,term,trunc,_=env.step(int(np.flatnonzero(obs['action_mask'])[0]))
            self.assertFalse(term);self.assertTrue(trunc)
            self.assertFalse(obs['action_mask'].any())
        with self.assertRaises(RuntimeError):env.step(0)

    def test_both_counting_inferences_repaired(self):
        # Reuse the historical legal fixtures, but inspect the current policy;
        # the old assertions are retained in the audit and intentionally not run.
        from twenty_five.policies import CountingPolicy
        from twenty_five.cards import Suit
        code=ast.parse((Path(__file__).parents[1]/'docs/audit/checks/check_6_counting_inferences.py').read_text())
        captured={}
        def trace(frame,event,arg):
            if frame.f_code is CountingPolicy.act.__code__:
                local=frame.f_locals
                if 'voids' in local:captured['voids']={s:set(v) for s,v in local['voids'].items()}
                if 'unseen_trumps' in local:captured['unseen']=list(local['unseen_trumps'])
            return trace
        # Run fixture setup/real policy calls, omit only the old defect assertions,
        # stdout and old tracer manipulation. Desired properties are asserted below.
        nodes=[]
        for n in code.body:
            if isinstance(n,ast.Assert) or (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) in {'print','sys.settrace','captured.clear'}):continue
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='captured' for t in n.targets):continue
            if isinstance(n,ast.FunctionDef) and n.name=='trace':continue
            if isinstance(n,ast.Try):
                nodes+=n.body
            else:nodes.append(n)
        namespace={'__file__':str(Path(__file__).parents[1]/'docs/audit/checks/check_6_counting_inferences.py'),'captured':captured,'trace':trace}
        sys.settrace(trace)
        try:exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),'historical-fixture-only','exec'),namespace)
        finally:sys.settrace(None)
        self.assertNotIn(Suit.HEARTS,captured['voids'].get(1,set()))
        self.assertIn(namespace['turned'],captured['unseen'])


class ReplayRepairTests(unittest.TestCase):
    def test_invalid_priority_inputs_fail_explicitly(self):
        replay=ReplayBuffer(2,1)
        replay.add(np.zeros(1),0,0.,np.zeros(1),False,np.ones(52),1.)
        for alpha,beta in ((-1,.4),(.6,2),(float('nan'),.4)):
            with self.assertRaises(ValueError):replay.sample(1,np.random.default_rng(0),alpha,beta)
        for indices,errors in ((np.array([0]),np.array([float('nan')])),(np.array([-1]),np.array([1.])),
                               (np.array([0.5]),np.array([1.])),(np.array([0]),np.array([1.,2.]))):
            with self.assertRaises(ValueError):replay.update_priorities(indices,errors)

    def replay(self):
        replay=ReplayBuffer(2,1)
        for i in range(2):replay.add(np.zeros(1),i,float(i),np.zeros(1),True,np.zeros(52),1.)
        replay.priorities[:]=[1.,9.]
        return replay

    def test_sampling_distribution_and_global_importance_correction(self):
        replay=self.replay();rng=np.random.default_rng(11)
        losses=[];frequencies=np.zeros(2);weights=defaultdict(set);duplicates=False
        for _ in range(8000):
            b=replay.sample(2,rng,alpha=1,beta=1)
            np.add.at(frequencies,b.indices,1)
            losses.append(float(np.mean(b.rewards*b.weights)))
            for i,w in zip(b.indices,b.weights):weights[int(i)].add(float(w))
            duplicates|=b.indices[0]==b.indices[1]
        np.testing.assert_allclose(frequencies/frequencies.sum(),[.1,.9],atol=.012)
        # Buffer-wide normalization scales uniform mean .5 by N*min(P)=.2.
        self.assertAlmostEqual(np.mean(losses),.1,delta=.005)
        self.assertTrue(duplicates)
        self.assertTrue(all(len(values)==1 for values in weights.values()))
        self.assertTrue(all(w<.12 for w in weights[1]))  # singleton batches must not renormalize this to1
        for _ in range(10):
            b=replay.sample(1,rng,alpha=1,beta=1)
            self.assertIn(float(b.weights[0]),weights[int(b.indices[0])])

    def test_duplicate_updates_are_order_independent_maximum(self):
        replay=self.replay()
        replay.update_priorities(np.asarray([0,0]),np.asarray([9.,2.]))
        self.assertAlmostEqual(float(replay.priorities[0]),9.00001,places=4)
        replay.update_priorities(np.asarray([0,0]),np.asarray([2.,9.]))
        self.assertAlmostEqual(float(replay.priorities[0]),9.00001,places=4)


@unittest.skipIf(torch is None,'PyTorch unavailable')
class GradientRepairTests(unittest.TestCase):
    def model(self):
        from twenty_five.agents.neural import LinearActorCritic
        model=LinearActorCritic(NEURAL_OBSERVATION_SIZE,hidden=4,phase_heads=True)
        for p in model.parameters():p.data.zero_()
        obs=np.zeros((2,NEURAL_OBSERVATION_SIZE),np.float32)
        obs[:,OBSERVATION_LAYOUT.phase.start+1]=1
        masks=np.zeros((2,52),bool);masks[0,[0,1]]=1;masks[1,2]=1
        return model,obs,masks

    def test_actual_pg_m01_both_outcomes_baselines_and_equal_returns(self):
        from twenty_five.training.policy_gradient import _update
        for baseline in (False,True):
            for outcome,action in ((1.,1),(-1.,0)):
                model,obs,masks=self.model()
                opt=torch.optim.SGD(model.parameters(),lr=0)
                _update(model,opt,list(obs),list(masks),[action,2],[outcome,outcome],baseline,torch.device('cpu'))
                self.assertLess(float(model.actor.bias.grad[1]),0)
                self.assertEqual(float(model.actor.bias.grad[2]),0)
                if baseline:self.assertLess(float(model.critic.bias.grad[0])*outcome,0)

    def test_actual_ppo_m01_and_detached_rollout_contract(self):
        from twenty_five.training.losses import ppo_objective,compute_gae
        for outcome,action in ((1.,1),(-1.,0)):
            model,obs,masks=self.model()
            logits,values=model.policy_and_value(torch.as_tensor(obs),torch.as_tensor(masks))
            actions=torch.tensor([action,2])
            old=torch.distributions.Categorical(logits=logits).log_prob(actions).detach()
            rows=[SimpleNamespace(env_id=0,episode_id=0,seat=0,reward=r,value=0.,next_value=0.,done=i==1)
                  for i,r in enumerate([0.,outcome])]
            adv,targets=compute_gae(rows,1.,1.)
            loss,*_=ppo_objective(logits,values,actions,old,torch.as_tensor(adv),torch.as_tensor(targets))
            loss.backward()
            self.assertLess(float(model.actor.bias.grad[1]),0)
        with self.assertRaises(ValueError):
            ppo_objective(logits,values,actions,old.requires_grad_(),torch.as_tensor(adv),torch.as_tensor(targets))

    def test_interleaved_gae_and_clipping_independent_oracles(self):
        from twenty_five.training.losses import compute_gae,ppo_objective
        rows=[SimpleNamespace(env_id=e,episode_id=ep,seat=s,reward=r,value=v,next_value=nv,done=d) for e,ep,s,r,v,nv,d in
            [(0,0,0,1.,.5,.25,False),(1,0,1,2.,1.,.5,False),(0,0,1,7.,0.,0.,True),
             (0,0,0,3.,.25,0.,True),(1,0,1,-1.,.5,0.,True),(0,1,0,9.,0.,0.,True)]]
        a,y=compute_gae(rows,.5,.8)
        np.testing.assert_allclose(a,[1.725,.65,7,2.75,-1.5,9],atol=1e-6)
        np.testing.assert_allclose(y,[2.225,1.65,7,3,-1,9],atol=1e-6)
        # Valid binary distributions realizing ratios .7/1.3 from old p=.5.
        p=torch.tensor([.35,.65,.35,.65],requires_grad=True)
        logits=torch.stack([p.log(),(1-p).log()],dim=1)
        values=torch.tensor([.2,-.2,.4,-.4],requires_grad=True)
        loss,actor,critic,entropy,logratio,ratio=ppo_objective(logits,values,torch.zeros(4,dtype=torch.long),
            torch.full((4,),math.log(.5)),torch.tensor([1.,1.,-1.,-1.]),torch.zeros(4),entropy_coefficient=.01)
        self.assertAlmostEqual(float(actor),.05,places=6)
        self.assertAlmostEqual(float(critic),.1,places=6)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_delayed_known_optimum_pg_and_ppo_three_seeds(self):
        from twenty_five.training.policy_gradient import _update
        from twenty_five.training.losses import ppo_objective,compute_gae
        results=[]
        for seed in (0,1,2):
            for family in ('reinforce','reinforce_baseline','n_step_actor_critic','ppo'):
                torch.manual_seed(seed)
                model,obs,masks=self.model()
                torch.manual_seed(seed)
                optimizer=torch.optim.Adam(model.parameters(),lr=.03)
                if family!='ppo':
                    for _ in range(180):
                        with torch.no_grad():action=int(model.distribution(torch.as_tensor(obs[:1]),torch.as_tensor(masks[:1])).sample()[0])
                        outcome=1. if action==1 else -1.
                        _update(model,optimizer,list(obs),list(masks),[action,2],[outcome,outcome],family!='reinforce',torch.device('cpu'))
                else:
                    for _ in range(50):
                        states=[];legal=[];actions=[];old_log=[];rows=[]
                        for ep in range(8):
                            with torch.no_grad():
                                logits,values=model.policy_and_value(torch.as_tensor(obs),torch.as_tensor(masks))
                                dist=torch.distributions.Categorical(logits=logits)
                                chosen=dist.sample();logs=dist.log_prob(chosen)
                            outcome=1. if int(chosen[0])==1 else -1.
                            for k in range(2):
                                rows.append(SimpleNamespace(env_id=0,episode_id=ep,seat=0,reward=0. if k==0 else outcome,
                                    value=float(values[k]),next_value=float(values[1]) if k==0 else 0.,done=k==1))
                                states.append(obs[k]);legal.append(masks[k]);actions.append(int(chosen[k]));old_log.append(float(logs[k]))
                        a,y=compute_gae(rows,1.,1.)
                        logits,values=model.policy_and_value(torch.as_tensor(np.stack(states)),torch.as_tensor(np.stack(legal)))
                        loss,*_=ppo_objective(logits,values,torch.tensor(actions),torch.tensor(old_log),torch.as_tensor(a),torch.as_tensor(y),
                            policy_weights=torch.full((16,),2.))
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                with torch.no_grad():prob=float(model.distribution(torch.as_tensor(obs[:1]),torch.as_tensor(masks[:1])).probs[0,1])
                results.append((family,seed,prob))
                self.assertGreater(prob,.9,results[-1])
        print('Delayed-choice known optima through production loss paths:',results)


@unittest.skipIf(torch is None,'PyTorch unavailable')
class TrainingRepairTests(unittest.TestCase):
    def test_batched_extraction_preserves_actions_parameters_and_optimizer(self):
        from twenty_five.training import dqn,ppo
        sys.path.insert(0,str(Path(__file__).parents[1]/'verification/baseline'))
        from integration_checks import capture_actions,nested_equal
        for module in (dqn,ppo):
            source=inspect.getsource(module.train)
            if module is dqn:
                source=source.replace('greedy_actions = online(states).masked_fill(~masks, torch.finfo(torch.float32).min).argmax(dim=-1).cpu().tolist()',
                    'q_values = online(states).masked_fill(~masks, torch.finfo(torch.float32).min).cpu().numpy()')
                source=source.replace('else greedy_actions[env_id]','else int(q_values[env_id].argmax())')
                kwargs=dict(episodes=3,num_envs=2,batch_size=4,warmup_steps=4,replay_capacity=64,train_frequency=1)
            else:
                source=source.replace('sampled_actions = sampled.cpu().tolist()','sampled_actions = [int(x.item()) for x in sampled]')
                source=source.replace('sampled_values = values.cpu().tolist()','sampled_values = [float(x.item()) for x in values]')
                source=source.replace('sampled_log_probabilities = log_probabilities.cpu().tolist()',
                                      'sampled_log_probabilities = [float(x.item()) for x in log_probabilities]')
                kwargs=dict(updates=2,num_envs=2,horizon=8,batch_size=64,epochs=1,target_kl=None,progress=False)
            namespace=dict(module.train.__globals__)
            exec(compile(source,'<previous scalar extraction>','exec'),namespace)
            with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
                old_path,new_path=Path(td)/'old.pt',Path(td)/'new.pt'
                common=dict(seed=37,players=4,device='cpu',eval_games=0,checkpoint_interval=0,log_interval=0,**kwargs)
                _,old_actions=capture_actions(namespace['train'],output=old_path,**common)
                _,new_actions=capture_actions(module.train,output=new_path,**common)
                self.assertEqual(old_actions,new_actions)
                old=torch.load(old_path,weights_only=True);new=torch.load(new_path,weights_only=True)
                self.assertTrue(nested_equal(old['state_dict'],new['state_dict']))
                self.assertTrue(nested_equal(old['training_state']['optimizer_state_dict'],new['training_state']['optimizer_state_dict']))

    def test_dqn_readiness_collect_only_and_real_optimizer(self):
        from twenty_five.training.dqn import train
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'dqn.pt'
            common=dict(episodes=1,seed=4,output=p,players=2,device='cpu',num_envs=1,batch_size=2,
                        replay_capacity=8,eval_games=0,checkpoint_interval=0)
            with self.assertRaises(ValueError):train(**common,warmup_steps=100)
            with self.assertRaises(ValueError):train(**common,warmup_steps=2,gradient_steps=0)
            collected=train(**common,warmup_steps=100,collect_only=True)
            self.assertEqual(collected['optimizer_steps'],0)
            trained=train(**common,warmup_steps=2)
            self.assertGreater(trained['optimizer_steps'],0)
            data=torch.load(p,weights_only=True)
            self.assertIn('target_state_dict',data['training_state'])
            self.assertIn('cuda_rng_states',data['training_state'])
            self.assertEqual(data['metadata']['gamma'],1.)

    def test_dqn_contract_and_sidecar_digest_reject_mixed_runs(self):
        from twenty_five.training.dqn import train
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            first,second=Path(td)/'first.pt',Path(td)/'second.pt'
            common=dict(episodes=1,seed=7,players=2,device='cpu',num_envs=1,batch_size=2,
                        warmup_steps=2,replay_capacity=8,eval_games=0,checkpoint_interval=0,log_interval=0)
            train(output=first,**common)
            with self.assertRaisesRegex(ValueError,'contract'):train(output=second,resume=first,gamma=.9,**common)
            replay,metadata=ReplayBuffer.load(first.with_suffix('.replay.npz'))
            metadata['checkpoint_sha256']='wrong checkpoint'
            replay.save(first.with_suffix('.replay.npz'),metadata=metadata)
            with self.assertRaisesRegex(ValueError,'digest mismatch'):train(output=second,resume=first,**common)
            self.assertEqual(train(output=second,resume=first,resume_mode='warm_start',**common)['episodes'],1)

    def test_ppo_kl_guard_precedes_optimizer_step(self):
        from twenty_five.training import ppo
        objective=ppo.ppo_objective
        def divergent(logits,values,actions,old_logp,*args,**kwargs):
            return objective(logits,values,actions,old_logp+3.,*args,**kwargs)
        with tempfile.TemporaryDirectory() as td,patch.object(ppo,'ppo_objective',divergent),patch.object(torch.optim.Adam,'step') as step:
            path=Path(td)/'ppo.pt'
            result=ppo.train(1,1,path,players=2,num_envs=1,horizon=1,device='cpu',epochs=2,
                             eval_games=0,checkpoint_interval=0,log_interval=0,progress=False,target_kl=.01)
            self.assertEqual(result['optimizer_steps'],0);step.assert_not_called()
            metrics=json.loads(path.with_suffix('.metrics.jsonl').read_text().splitlines()[-1])
            self.assertTrue(metrics['train/stopped_for_target_kl'])

    def test_ppo_exact_split_with_history_and_fixed_neural_opponent(self):
        # Existing comparison helper invokes real trainers and compares actions,
        # model/Adam tensors, RNG, counters and historical opponent tensors.
        sys.path.insert(0,str(Path(__file__).parents[1]/'verification/baseline'))
        from integration_checks import ppo_split_comparison
        for neural in (False,True):
            with tempfile.TemporaryDirectory() as td:result=ppo_split_comparison(td,neural)
            self.assertTrue(all(v for k,v in result.items() if k.endswith('_equal')),result)

    def test_pg_and_tabular_exact_split_with_mixed_opponents(self):
        sys.path.insert(0,str(Path(__file__).parents[1]/'verification/baseline'))
        from integration_checks import capture_actions,nested_equal
        from twenty_five.training import policy_gradient,tabular_twentyfive
        jobs=[(v,policy_gradient.train,{'variant':v,'device':'cpu','n_steps':2})
              for v in ('reinforce','reinforce_baseline','n_step_actor_critic')]
        jobs += [(v,tabular_twentyfive.train,{'algorithm':v}) for v in ('mc','sarsa','q_learning','expected_sarsa')]
        for name,function,extra in jobs:
            with tempfile.TemporaryDirectory() as td:
                ext='.pt' if function is policy_gradient.train else '.json'
                full,first,split=[Path(td)/(s+ext) for s in ('full','first','split')]
                common=dict(seed=19,players=2,opponent_names=['random','highest'],**extra)
                _,a=capture_actions(function,episodes=4,output=full,**common)
                _,b=capture_actions(function,episodes=2,output=first,**common)
                _,c=capture_actions(function,episodes=2,output=split,resume=first,**common)
                self.assertEqual(a,b+c,name)
                if ext=='.pt':
                    f=torch.load(full,weights_only=True);s=torch.load(split,weights_only=True)
                    self.assertTrue(nested_equal(f['state_dict'],s['state_dict']),name)
                    self.assertTrue(nested_equal(f['training_state']['optimizer_state_dict'],s['training_state']['optimizer_state_dict']),name)
                else:
                    f=json.loads(full.read_text());s=json.loads(split.read_text())
                    for key in ('q','counts','rng_state'):self.assertEqual(f[key],s[key],name)

    def test_exact_contract_rejects_changes_and_lr_is_explicit(self):
        from twenty_five.training.policy_gradient import train
        with tempfile.TemporaryDirectory() as td:
            first,second=Path(td)/'first.pt',Path(td)/'second.pt'
            train('reinforce_baseline',1,11,first,players=2,device='cpu',learning_rate=.0003)
            with self.assertRaises(ValueError):train('reinforce_baseline',1,11,second,players=3,device='cpu',resume=first)
            train('reinforce_baseline',1,11,second,players=2,device='cpu',resume=first,learning_rate=.003)
            payload=torch.load(second,weights_only=True)
            self.assertEqual(payload['metadata']['requested_learning_rate'],.003)
            self.assertEqual(payload['metadata']['effective_learning_rates'],[.0003])
            train('reinforce_baseline',1,11,second,players=2,device='cpu',resume=first,learning_rate=.003,override_learning_rate=True)
            self.assertEqual(torch.load(second,weights_only=True)['metadata']['effective_learning_rates'],[.003])
            train('reinforce_baseline',1,11,second,players=3,device='cpu',resume=first,resume_mode='warm_start')
            self.assertEqual(torch.load(second,weights_only=True)['training_state']['episodes'],1)

    def test_gridworld_cap_has_nonterminal_tail(self):
        from twenty_five.agents.tabular import discounted_returns
        from twenty_five.training.gridworld_mc import train
        self.assertEqual(discounted_returns([1.,2.],.5,bootstrap=4.),[3.,4.])
        with tempfile.TemporaryDirectory() as td:
            result=train(3,1,Path(td)/'grid.json',max_steps=1)
            self.assertEqual(result['truncated_episodes'],3)
            self.assertIn('bootstrap',result['cutoff_semantics'])


class EvaluationRepairTests(unittest.TestCase):
    def test_failed_policy_accounting_and_interval_labels(self):
        from twenty_five.evaluation import evaluate
        from twenty_five.policies import make_policy
        class Invalid:
            name='bad'
            def act(self,view,rng):return 52
        with tempfile.TemporaryDirectory() as td:
            result=evaluate({'bad':Invalid(),'good':make_policy('random')},4,[-100],Path(td),2,matchup_games=1)
        bad=next(r for r in result['rows'] if r['policy']=='bad')
        self.assertEqual(bad['attempted_games'],4);self.assertEqual(bad['completed_games'],0)
        self.assertEqual(bad['invalid_action_count'],4);self.assertEqual(bad['failed_games'],4)
        self.assertFalse(bad['valid_for_comparison']);self.assertIsNone(bad['average_final_score'])
        self.assertIn('Wilson',bad['ci95_method']);self.assertIn('Wilson',result['confidence_interval'])
        self.assertTrue(result['diagnostic_failures'])

    @unittest.skipIf(torch is None,'PyTorch unavailable')
    def test_sampled_checkpoint_policy_is_distinct_and_uses_eval_rng(self):
        from twenty_five.agents.neural import LinearActorCritic
        from twenty_five.policies import FrozenCheckpointPolicy
        with tempfile.TemporaryDirectory() as td:
            model=LinearActorCritic(NEURAL_OBSERVATION_SIZE,hidden=4)
            for p in model.parameters():p.data.zero_()
            path=Path(td)/'actor.pt';model.save(path)
            greedy=FrozenCheckpointPolicy(path);sampled=FrozenCheckpointPolicy(path,mode='sampled')
            game=TwentyFiveGame(2,1);view=PlayerView.from_game(game,game.current_player)
            rng=random.Random(7);before=torch.get_rng_state().clone()
            actions=[sampled.act(view,rng) for _ in range(80)]
            self.assertGreater(len(set(actions)),1)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertEqual(len({greedy.act(view,rng) for _ in range(10)}),1)
