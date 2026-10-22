"""Mouse recording headings and velocities use sampled model-step motion."""
import unittest
from chimesim.recording import MouseTrajectory


class MouseRecordingTests(unittest.TestCase):
    def test_positions_heading_and_model_step_velocity(self):
        track=MouseTrajectory()
        self.assertEqual(track.sample((10,20),0),(1,10,20,None,None,None))
        self.assertEqual(track.sample((14,20),2),(1,14,20,0,2,0))
        self.assertEqual(track.sample((14,23),3),(1,14,23,90,0,3))
        self.assertEqual(track.sample((14,23),4),(1,14,23,90,0,0))

    def test_inactive_samples_and_restart_do_not_bridge_gaps(self):
        track=MouseTrajectory()
        track.sample((10,20),0);track.sample((10,10),1)
        self.assertEqual(track.heading,270)
        self.assertEqual(track.sample(None,2),(0,None,None,None,None,None))
        self.assertEqual(track.sample((80,90),3),(1,80,90,None,None,None))
        track.sample((70,90),4)
        self.assertEqual(track.heading,180)
        track.reset()
        self.assertEqual(track.sample((100,100),100),(1,100,100,None,None,None))
