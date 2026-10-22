"""Known collective states, largest-group ranks and starter plots."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import unittest
from types import SimpleNamespace
import numpy as np
from chimesim import metrics
from chimesim.environment import Environment

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    pd=plt=None


class StateTests(unittest.TestCase):
    def test_perfect_mill_has_rotation_without_polarisation(self):
        angles=np.arange(4)*np.pi/2
        agents=[SimpleNamespace(x=50+10*np.cos(a),y=50+10*np.sin(a),dx=-np.sin(a),dy=np.cos(a)) for a in angles]
        state=metrics.group_state(agents,Environment(100,100))
        self.assertAlmostEqual(state['milling'],1)
        self.assertAlmostEqual(state['polarisation'],0)
        self.assertAlmostEqual(state['centroid_distance'],10)
        self.assertAlmostEqual(state['elongation'],0)

    def test_opposite_mills_in_separate_groups_do_not_cancel(self):
        agents=[]
        labels=[]
        kind=SimpleNamespace(name='A1')
        for label,(cx,spin) in enumerate(((30,1),(130,-1))):
            for angle in np.arange(8)*np.pi/4:
                agents.append(SimpleNamespace(type=kind,x=cx+10*np.cos(angle),y=50+10*np.sin(angle),
                                              dx=-spin*np.sin(angle),dy=spin*np.cos(angle)))
                labels.append(label)
        value=metrics.milling_by_type(agents,Environment(200,100),labels)
        self.assertAlmostEqual(value['A1'],1)
        # Counter-rotation inside a single physical component reduces its order.
        for i,agent in enumerate(agents[:8]):
            if i%2:
                agent.dx*=-1;agent.dy*=-1
        value=metrics.milling_by_type(agents[:8],Environment(200,100),[0]*8)
        self.assertAlmostEqual(value['A1'],0)
        self.assertTrue(np.isnan(metrics.milling_by_type(agents[:1],Environment(200,100),[-1])['A1']))


@unittest.skipIf(pd is None or plt is None,'Optional analysis dependencies are not installed')
class AnalysisTests(unittest.TestCase):
    def source(self, reverse=False):
        trajectories=[]
        for frame in (1,2):
            for uid in range(5):
                x=(uid*2 if uid<3 else 60+(uid-3)*2)
                if frame==2 and reverse and uid<3:x=(2-uid)*2
                trajectories.append(dict(set='c0_r0',condition=0,rep=0,frame=frame,t=frame,id=uid,
                                         x=10+x+frame,y=10,vx=1,vy=0,speed=1,heading_rad=0))
        return dict(settings={'environment':{'width':100,'height':100,'bounded':True},'agents':{'A1':{}}},
                    trajectories=trajectories,individuals=[dict(set='c0_r0',condition=0,rep=0,id=i,agent_type='A1',speed_scale=1+i*.1) for i in range(5)])

    def test_largest_group_scores_average_to_half_when_positions_swap(self):
        from chimesim.analysis import process
        result=process(self.source(reverse=True),link_distance=3)
        self.assertEqual(set(result['positions'].id),{0,1,2})
        np.testing.assert_allclose(result['individuals'].front_rank,.5)
        self.assertEqual(set(result['individuals'].front_samples),{2})
        self.assertEqual(len(result['groups']),2)
        self.assertAlmostEqual(result['groups'].centroid_speed.iloc[1],1)

    def test_persistent_positions_do_not_get_artificially_reset_to_half(self):
        from chimesim.analysis import process
        result=process(self.source(),link_distance=3)
        np.testing.assert_allclose(result['individuals'].front_rank,[0,.5,1])
        self.assertIn('speed_scale',result['individuals'])
        self.assertGreater(result['structuring'].front_rank_sd.iloc[0],0)
        all_groups=process(self.source(),groups='all',link_distance=3)
        self.assertEqual(set(all_groups['positions'].id),set(range(5)))

    def test_tied_largest_groups_are_all_included(self):
        from chimesim.analysis import process
        source=self.source()
        for frame in (1,2):
            source['trajectories'].append(dict(set='c0_r0',condition=0,rep=0,frame=frame,t=frame,id=5,
                                               x=74+frame,y=10,vx=1,vy=0,speed=1,heading_rad=0))
        source['individuals'].append(dict(set='c0_r0',condition=0,rep=0,id=5,agent_type='A1',speed_scale=1))
        result=process(source,link_distance=3)
        self.assertEqual(set(result['positions'].id),set(range(6)))
        self.assertEqual(len(result['groups']),4)

    def test_actual_run_folder_and_existing_summary_plots(self):
        import tempfile
        from chimesim import settings, simulate
        from chimesim.analysis import process, load
        from chimesim.plotting import plot
        scenario=settings(source='defaults', environment={'width':100,'height':100,'spawn_radius':10},
                          agents={'A1':{'n':5,'noise':0}})
        with tempfile.TemporaryDirectory() as folder:
            saved=simulate(settings=scenario,n_steps=3,burn_in=0,seed=4,save_trajectories=True,output_dir=folder)
            result=process(saved,link_distance=30)
            self.assertEqual(len(result['positions']),15)
            loaded=load(saved['output_dir'])
            self.assertEqual(len(loaded['trajectories']),15)
            ax=plot(saved,'speed',kind='box',category='condition')
            plt.close(ax.figure)
            ax=plot(saved,'speed',kind='box',table='repeats',category='type')
            plt.close(ax.figure)

    def test_time_box_density_and_trait_scatter_return_editable_axes(self):
        from chimesim.analysis import process
        from chimesim.plotting import plot
        result=process(self.source(),link_distance=3)
        for kind in ('time','box','density'):
            ax=plot(result,'speed',kind=kind,category=['condition','agent_type'])
            self.assertIsNotNone(ax)
            self.assertEqual(ax.get_title(),'')
            plt.close(ax.figure)
        ax=plot(result,'front_rank',kind='scatter',x='speed_scale',table='individuals')
        self.assertEqual(len(ax.collections[0].get_offsets()),3)
        plt.close(ax.figure)
        ax=plot(result,'front_rank',kind='box',table='individuals',unit='individual')
        plt.close(ax.figure)

    def test_time_bins_average_separately_within_repeats(self):
        from chimesim.plotting import plot
        frame=pd.DataFrame(dict(set=['first']*4+['second']*4, rep=[0]*4+[1]*4,
                                t=[1,2,3,4]*2, speed=[0,2,4,6,10,12,14,16]))
        ax=plot(frame,'speed',category='rep',time_bin=2)
        self.assertEqual(len(ax.lines),2)
        np.testing.assert_allclose(ax.lines[0].get_ydata(),[0,3,6])
        np.testing.assert_allclose(ax.lines[1].get_ydata(),[10,13,16])
        np.testing.assert_allclose(ax.lines[0].get_xdata(),[1,2.5,4])
        self.assertEqual(len(frame),8)
        plt.close(ax.figure)
        for value in (0,-1,float('nan'),float('inf'),True,'25'):
            with self.assertRaises(ValueError):
                plot(frame,'speed',category='rep',time_bin=value)
        with self.assertRaises(ValueError):
            plot(frame,'speed',kind='box',category='rep',time_bin=2)

    def test_singleton_or_weakly_aligned_agents_have_no_front_rank(self):
        from chimesim.analysis import process
        source=self.source()
        for row in source['trajectories']:
            row['vx']=0;row['vy']=0
        result=process(source,link_distance=3)
        self.assertTrue(result['positions'].front_rank.isna().all())
        self.assertTrue(result['individuals'].front_rank.isna().all())
