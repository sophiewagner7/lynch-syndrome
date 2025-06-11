import src.configs as c
import src.configs.global_configs as global_configs
import src.configs.screening_configs as screening

import numpy as np
import pandas as pd
import logging


class StateSpace:
    def __init__(self, state_names):
        self.states = state_names
        self.index = {name: i for i, name in enumerate(state_names)}


class CohortModel:
    def __init__(self, cohort_spec, transition_matrix):
        """
        Initialize the cohort model with a specification and a transition matrix.
        T = transition matrix
        D = distribution vector for health states (where population is when)
        I = incidence vector for health states (when people enter the state)
        P = prevalence vector, normalized per alive population
        R = incidence vector, normalized per alive population
        """
        self.spec = cohort_spec
        self.state_space = transition_matrix.states
        self.T = transition_matrix.matrix
        self.D = np.zeros((self.spec.cycles, global_configs.n_states))
        self.I = np.zeros((self.spec.cycles, global_configs.n_states))
        self.P = None
        self.R = None
        self.screen_detected = np.zeros(
            (4, int(global_configs.NUM_CYCLES))
        )  # Detected cancer cases
        self.is_screening_cycle = (
            False  # Flag to indicate if current cycle is a screening cycle
        )
        self.screening_log = pd.DataFrame(
            index=range(self.spec.cycles),
            columns=self.spec.screening_protocol.keys(),
            dtype=float,
        )

    def update_screening_cycle(self, t):
        if t in self.spec.screening_cycles:
            self.is_screening_cycle = True
        else:
            self.is_screening_cycle = False

    def apply_screening(self, t):
        # Placeholder for screening logic
        # This function should modify the state_vector based on screening results
        # For now, we just return the state_vector unchanged
        state_vector = self.D[t].copy()
        state_inflow = self.I[t].copy()

        # Get screening protocol for the current cycle
        test = self.spec.screening_protocol[t]
        test_specs = screening.SCREENING_TEST_SPECS[test]
        self.screening_log[t][test] = sum(
            state_vector[self.state_space.index[alive_state]]
            for alive_state in global_configs.alive_states
        )

        # Healthy
        false_positive = (
            state_vector[self.state_space.index["healthy"]] * test_specs["fpr"]
        )
        # TODO: What to do with false positives?

        # Polyps
        lr_polyp_detected = (
            state_vector[self.state_space.index["lr_polyp"]]
            * test_specs["sens_lr_polyp"]
        )
        hr_polyp_detected = (
            state_vector[self.state_space.index["hr_polyp"]]
            * test_specs["sens_hr_polyp"]
        )

        # TODO: Apply surveillance protocol to detected polyps? Or just remove them?
        # LR polyps assumed to be removed, go to healthy state
        state_vector[self.state_space.index["healthy"]] += lr_polyp_detected
        state_vector[self.state_space.index["healthy"]] += lr_polyp_detected
        state_vector[self.state_space.index["lr_polyp"]] -= lr_polyp_detected
        state_vector[self.state_space.index["hr_polyp"]] -= hr_polyp_detected

        # Cancer
        self.screen_detected[0, t] = (
            state_vector[self.state_space.index["u_stage_1"]]
            * test_specs["sens_stage_1"]
        )
        self.screen_detected[1, t] = (
            state_vector[self.state_space.index["u_stage_2"]]
            * test_specs["sens_stage_2"]
        )
        self.screen_detected[2, t] = (
            state_vector[self.state_space.index["u_stage_3"]]
            * test_specs["sens_stage_3"]
        )
        self.screen_detected[3, t] = (
            state_vector[self.state_space.index["u_stage_4"]]
            * test_specs["sens_stage_4"]
        )
        # Update incidence vector with detected cancer cases
        self.I[t, self.state_space.index["d_stage_1"]] += self.screen_detected[0, t]
        self.I[t, self.state_space.index["d_stage_2"]] += self.screen_detected[1, t]
        self.I[t, self.state_space.index["d_stage_3"]] += self.screen_detected[2, t]
        self.I[t, self.state_space.index["d_stage_4"]] += self.screen_detected[3, t]
        # Update state_vector based on screening results
        state_vector[self.state_space.index["u_stage_1"]] -= self.screen_detected[0, t]
        state_vector[self.state_space.index["u_stage_2"]] -= self.screen_detected[1, t]
        state_vector[self.state_space.index["u_stage_3"]] -= self.screen_detected[2, t]
        state_vector[self.state_space.index["u_stage_4"]] -= self.screen_detected[3, t]

    def normalize_distribtution_vectors(self):
        """
        Normalize the distribution vectors to ensure they sum to 1.
        This is done for both the distribution and incidence vectors.
        """
        # Incidence and prevalence denominator is out of living population
        dead_factor = np.divide(
            global_configs.POPULATION_SIZE,
            (
                global_configs.POPULATION_SIZE
                - self.D[len(global_configs.alive_states)].sum(axis=0)
            ),
        )
        self.P = np.zeros_like(self.D)
        self.R = self.I.copy()
        for state in range(len(self.state_space.states)):
            if self.state_space.states[state] in global_configs.alive_states:
                self.P[:, state] = np.multiply(self.D[:, state], dead_factor)
                self.I[:, state] = np.multiply(self.I[:, state], dead_factor)

    def run(self):
        start_state = np.zeros(len(self.state_space.states))
        start_state[self.state_space.index["healthy"]] = 1.0
        self.D[0] = start_state
        self.curr_age = self.spec.start_age
        self.curr_age_layer = 0
        t = 0
        mat = self.T[0]
        inflow_mat = np.tril(mat, k=-1)

        while t <= self.spec.cycles:
            self.D[t] = self.D[t - 1] @ mat
            self.I[t] = self.D[t - 1] @ inflow_mat

            # Apply screening if applicable
            if self.spec.interval > 0 and self.is_screening_cycle:
                self.apply_screening(t)

            t += 1

            # Update age and transition matrix when necessary
            if t % 12 == 0:
                self.curr_age += 1
                self.update_screening_cycle(t)
                if self.curr_age_layer in global_configs.AGE_LAYERS:
                    mat = self.T[self.curr_age_layer]
                    inflow_mat = np.tril(mat, k=-1)
                self.curr_age_layer += 1
                if self.curr_age > self.spec.stop_age:
                    break

        self.normalize_distribtution_vectors()
