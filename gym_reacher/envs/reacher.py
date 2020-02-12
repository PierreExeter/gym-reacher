from gym_reacher.envs.robot_bases import MJCFBasedRobot
import numpy as np


class Reacher1(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'reacher_1dof.xml', 'body0', action_dim=1, obs_dim=6)

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["target_y"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)

        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]

        self.joint0 = self.jdict["joint0"]

        self.joint0.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.joint0.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        self.angle0, self.angle0_dot = self.joint0.current_relative_position()

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()

        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())

        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            self.angle0,
            self.angle0_dot
        ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)



class Reacher2(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'reacher_2dof.xml', 'body0', action_dim=2, obs_dim=8)

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["target_y"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)

        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]

        self.joint0 = self.jdict["joint0"]
        self.joint1 = self.jdict["joint1"]

        self.joint0.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint1.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.joint0.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.joint1.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        self.angle0, self.angle0_dot = self.joint0.current_relative_position()
        self.angle1, self.angle1_dot = self.joint1.current_relative_position()

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()

        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())

        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            self.angle0,
            self.angle0_dot,
            self.angle1,
            self.angle1_dot
        ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)



class Reacher3(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'reacher_3dof.xml', 'body0', action_dim=3, obs_dim=10)

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["target_y"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)

        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]

        self.joint0 = self.jdict["joint0"]
        self.joint1 = self.jdict["joint1"]
        self.joint2 = self.jdict["joint2"]

        self.joint0.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint1.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint2.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.joint0.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.joint1.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))
        self.joint2.set_motor_torque(0.05 * float(np.clip(a[2], -1, +1)))

    def calc_state(self):
        self.angle0, self.angle0_dot = self.joint0.current_relative_position()
        self.angle1, self.angle1_dot = self.joint1.current_relative_position()
        self.angle2, self.angle2_dot = self.joint2.current_relative_position()

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()

        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())

        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            self.angle0,
            self.angle0_dot,
            self.angle1,
            self.angle1_dot,
            self.angle2,
            self.angle2_dot
        ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)


class Reacher4(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'reacher_4dof.xml', 'body0', action_dim=4, obs_dim=12)

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["target_y"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)

        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]

        self.joint0 = self.jdict["joint0"]
        self.joint1 = self.jdict["joint1"]
        self.joint2 = self.jdict["joint2"]
        self.joint3 = self.jdict["joint3"]

        self.joint0.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint1.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint2.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint3.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        
    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.joint0.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.joint1.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))
        self.joint2.set_motor_torque(0.05 * float(np.clip(a[2], -1, +1)))
        self.joint3.set_motor_torque(0.05 * float(np.clip(a[3], -1, +1)))

    def calc_state(self):
        self.angle0, self.angle0_dot = self.joint0.current_relative_position()
        self.angle1, self.angle1_dot = self.joint1.current_relative_position()
        self.angle2, self.angle2_dot = self.joint2.current_relative_position()
        self.angle3, self.angle3_dot = self.joint3.current_relative_position()

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()

        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())

        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            self.angle0,
            self.angle0_dot,
            self.angle1,
            self.angle1_dot,
            self.angle2,
            self.angle2_dot,
            self.angle3,
            self.angle3_dot
        ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)


class Reacher5(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'reacher_5dof.xml', 'body0', action_dim=5, obs_dim=14)

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["target_y"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)

        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]

        self.joint0 = self.jdict["joint0"]
        self.joint1 = self.jdict["joint1"]
        self.joint2 = self.jdict["joint2"]
        self.joint3 = self.jdict["joint3"]
        self.joint4 = self.jdict["joint4"]

        self.joint0.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint1.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint2.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint3.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint4.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        
    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.joint0.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.joint1.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))
        self.joint2.set_motor_torque(0.05 * float(np.clip(a[2], -1, +1)))
        self.joint3.set_motor_torque(0.05 * float(np.clip(a[3], -1, +1)))
        self.joint4.set_motor_torque(0.05 * float(np.clip(a[4], -1, +1)))

    def calc_state(self):
        self.angle0, self.angle0_dot = self.joint0.current_relative_position()
        self.angle1, self.angle1_dot = self.joint1.current_relative_position()
        self.angle2, self.angle2_dot = self.joint2.current_relative_position()
        self.angle3, self.angle3_dot = self.joint3.current_relative_position()
        self.angle4, self.angle4_dot = self.joint4.current_relative_position()

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()

        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())

        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            self.angle0,
            self.angle0_dot,
            self.angle1,
            self.angle1_dot,
            self.angle2,
            self.angle2_dot,
            self.angle3,
            self.angle3_dot,
            self.angle4,
            self.angle4_dot
        ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)


class Reacher6(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'reacher_6dof.xml', 'body0', action_dim=6, obs_dim=16)

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["target_y"].reset_current_position(
            self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)

        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]

        self.joint0 = self.jdict["joint0"]
        self.joint1 = self.jdict["joint1"]
        self.joint2 = self.jdict["joint2"]
        self.joint3 = self.jdict["joint3"]
        self.joint4 = self.jdict["joint4"]
        self.joint5 = self.jdict["joint5"]

        self.joint0.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint1.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint2.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint3.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint4.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.joint5.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        
    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.joint0.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.joint1.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))
        self.joint2.set_motor_torque(0.05 * float(np.clip(a[2], -1, +1)))
        self.joint3.set_motor_torque(0.05 * float(np.clip(a[3], -1, +1)))
        self.joint4.set_motor_torque(0.05 * float(np.clip(a[4], -1, +1)))
        self.joint5.set_motor_torque(0.05 * float(np.clip(a[5], -1, +1)))

    def calc_state(self):
        self.angle0, self.angle0_dot = self.joint0.current_relative_position()
        self.angle1, self.angle1_dot = self.joint1.current_relative_position()
        self.angle2, self.angle2_dot = self.joint2.current_relative_position()
        self.angle3, self.angle3_dot = self.joint3.current_relative_position()
        self.angle4, self.angle4_dot = self.joint4.current_relative_position()
        self.angle5, self.angle5_dot = self.joint5.current_relative_position()

        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()

        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())

        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            self.angle0,
            self.angle0_dot,
            self.angle1,
            self.angle1_dot,
            self.angle2,
            self.angle2_dot,
            self.angle3,
            self.angle3_dot,
            self.angle4,
            self.angle4_dot,
            self.angle5,
            self.angle5_dot
        ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)



