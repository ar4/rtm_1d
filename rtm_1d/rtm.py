import numpy as np
from wave_1d_fd_pml.propagators import Pml2

class Rtm(object):
    def __init__(self, dx, dt=None, pml_width=10, profile=None):
        self.dx = dx
        self.dt = dt
        self.pml_width = pml_width
        self.profile = profile

    def migrate_shot(self, model, source, source_x, receivers, receivers_x,
                     imaging_condition_interval=1):
        assert source.ndim == 1
        assert receivers.ndim == 2
        source = source[np.newaxis, :]
        source_x = np.array([source_x])
        num_imaging_steps = int((receivers.shape[1] - 1) / imaging_condition_interval)

        prop = Pml2(model, self.dx, self.dt, self.pml_width, self.profile)
        nx = len(model)

        source_snapshots = self._forward_source(source, source_x,
                                                imaging_condition_interval,
                                                num_imaging_steps, prop, nx)

        image = self._backward_receivers(receivers, receivers_x,
                                         imaging_condition_interval,
                                         num_imaging_steps,
                                         source_snapshots, prop, nx)

        return image

    def _forward_source(self, source, source_x,
                        imaging_condition_interval,
                        num_imaging_steps, prop, nx):

        source_snapshots = np.zeros([num_imaging_steps, nx], np.float32)
        for imaging_step in range(0, num_imaging_steps):
            start_time_step = imaging_step * imaging_condition_interval
            end_time_step = start_time_step + imaging_condition_interval
            if end_time_step < source.shape[1]:
                source_snapshots[imaging_step, :] = \
                        prop.step(imaging_condition_interval,
                                  source[:, start_time_step:end_time_step],
                                  source_x)
            elif start_time_step < source.shape[1]:
                remaining_source_steps = source.shape[1] - start_time_step
                steps_after_source = (imaging_condition_interval -
                                      remaining_source_steps)
                prop.step(remaining_source_steps,
                          source[:, start_time_step:],
                          source_x)
                source_snapshots[imaging_step, :] = \
                        prop.step(steps_after_source)
            else:
                source_snapshots[imaging_step, :] = \
                        prop.step(imaging_condition_interval)

        return source_snapshots

    def _backward_receivers(self, receivers, receivers_x,
                            imaging_condition_interval,
                            num_imaging_steps,
                            source_snapshots, prop, nx):

        image = np.zeros([nx], np.float32)
        for imaging_step in range(num_imaging_steps - 1, -1, -1):
            start_time_step = (imaging_step + 1) * imaging_condition_interval
            end_time_step = start_time_step - imaging_condition_interval
            if start_time_step > receivers.shape[1]:
                start_time_step = receiver.shape[1]
            receiver_snapshot = \
                    prop.step(start_time_step - end_time_step,
                              receivers[:, start_time_step:end_time_step:-1],
                              receivers_x)
            image += (source_snapshots[imaging_step, :] *
                      receiver_snapshot[:] * imaging_condition_interval)

        return image

    def model_shot(self, model, source, source_x, receivers_x, max_time):
        assert source.ndim == 1
        source = source[np.newaxis, :]
        source_x = np.array([source_x])
        num_receivers = len(receivers_x)

        prop = Pml2(model, self.dx, self.dt, self.pml_width, self.profile)

        nt = int(max_time / self.dt)
        receivers = np.zeros([num_receivers, nt], np.float32)
        for step in range(nt):
            wavefield = prop.step(1,
                                  source[:, step:step+1],
                                  source_x)
            receivers[:, step] = wavefield[receivers_x]

        return receivers
