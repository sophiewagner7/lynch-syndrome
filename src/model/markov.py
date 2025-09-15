import logging
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import pandas as pd

from configs import global_configs as c
from configs import strategy as strategy
from configs import inputs
from .runspec import RunSpec


@dataclass
class MarkovModel:

    def __init__(self, run_spec: RunSpec):
        """
        Arguments: run_spec (RunSpec): details of this run. Allows access to transition
        matrix and screening protocol.

        Initialize core matrices and logs for the model.
        T: age_layered transition matrix, aligned as T[age_layer, to, from]
        D:  distribution over health states at each cycle (population currently in each state)
        I:  incidence/entries into states at each cycle (flows entering each state)
        P:  prevalence normalized by alive population (computed in normalize_distribution_vectors)
        R:  incidence normalized by alive population (computed in normalize_distribution_vectors)

        Screening:
        - screening_protocol[t] gives the *first-line* test (e.g., 'FIT' or 'colo') at cycle t, or None.
        - screening_log[t, test] logs count screened by 'test' at cycle t.
        - complications_log tracks colonoscopy complications/deaths.
        - screen_detected[t, stage_idx] logs screen-detected cancers by (1..4).
        - symptom_detected[t, stage_idx] logs symptom-detected cancers by (1..4).

        Notes:
        - All vectors passed into helpers are GLOBAL-length (size = c.n_states).
        - We only modify indices relevant to each operation using global indices from configs.
        """

        self.spec = run_spec  # holds strategy and cohort
        self.T = run_spec.tmat.transpose(0, 2, 1)

        # Main vectors
        self.D = np.zeros((self.spec.cycles, c.n_states), dtype=float)
        self.I = np.zeros((self.spec.cycles, c.n_states), dtype=float)
        self.P = np.zeros_like(self.D)  # filled in normalize_distribution_vectors()
        self.R = np.zeros_like(self.I)  # filled in normalize_distribution_vectors()

        # Screening protocol, cycle flag
        self.screening_protocol = self.spec.get_screening_protocol()
        self.is_screening_cycle = False
        tests = self.spec.strategy.tests() if not self.spec.strategy.is_NH else []

        # Logs
        self.screening_log = (
            pd.DataFrame(0.0, index=range(self.spec.cycles), columns=tests)
            if tests
            else pd.DataFrame()
        )
        self.complications_log = pd.DataFrame(
            0.0,
            index=range(self.spec.cycles),
            columns=["colo_complications", "complication_deaths"],
        )
        self.screen_detected = np.zeros((self.spec.cycles, 4), dtype=float)
        self.symptom_detected = np.zeros((self.spec.cycles, 4), dtype=float)

    # ---------- helpers ----------
    def _update_screening_cycle(self, t) -> None:
        """Update screening cycle flag based on current time step."""
        self.is_screening_cycle = self.screening_protocol[t] is not None

    def _apply_colonoscopy_complications(
        self, population: np.ndarray, t: int, is_screening: bool = False
    ):
        """
        Apply colonoscopy complications to those who undergo/underwent colonoscopy.

        Arguments:
            population (np.ndarray):
                GLOBAL-length vector at cycle t whose entries represent *people undergoing
                colonoscopy* (Mass should be nonzero only in appropriate states.)
            t (int):
                Cycle index.
            is_screening (bool):
                If True, apply complications to screening states; else to detected states
                (symptom-detected cohort).

        Returns:
            np.ndarray: GLOBAL-length vector of colonoscopy complication deaths (zeros elsewhere).

        Side effects:
            - Modifies self.D[t] and self.I[t] in place.
            - Updates self.complications_log.
        """
        state_idx = c.health_states_stoi
        colo_specs = inputs.SCREENING_TEST_SPECS["colo"]

        # States that may experience colo complications in this call
        colo_risk_idx = (
            c.screening_states_idx if is_screening else c.detected_states_idx
        )

        # Compute complications & deaths (keep full global-length arrays)
        complications = np.zeros_like(population, dtype=float)
        complication_deaths = np.zeros_like(population, dtype=float)

        complications[colo_risk_idx] = (
            population[colo_risk_idx] * colo_specs["p_complication"]
        )
        complication_deaths[colo_risk_idx] = (
            complications[colo_risk_idx] * colo_specs["p_death_complication"]
        )

        # Log complications & deaths
        total_colo_complications = complications.sum()
        total_colo_deaths = complication_deaths.sum()
        self.complications_log.at[t, "colo_complications"] += total_colo_complications
        self.complications_log.at[t, "complication_deaths"] += total_colo_deaths

        # Remove those deaths from alive states in distribution D
        self.D[t, colo_risk_idx] -= complication_deaths[colo_risk_idx]

        # Route deaths to colonoscopy-death absorbing state
        self.D[t, state_idx["death_colo"]] += total_colo_deaths
        self.I[t, state_idx["death_colo"]] += total_colo_deaths

        return complication_deaths

    def _apply_colonoscopy(
        self, screening_pop: np.ndarray, t: int, followup: bool = False
    ) -> None:
        """
        Apply colonoscopy to the given population vector (GLOBAL-length).

        Arguments:
            screening_pop (np.ndarray):
                GLOBAL-length vector indicating who actually receives colonoscopy at cycle t
                (nonzero only in screening states for first-line or positives for follow-up).
            t (int):
                Cycle index.
            followup (bool):
                Whether this is a follow-up colonoscopy after a non-colo screen.

        Side effects:
            - Updates D[t] and I[t] for complications, polyp removals, and cancer detection.
            - Logs counts in screening_log['colo'] and screen_detected[t, :].
        """
        state_idx = c.health_states_stoi

        # Log number of people getting follow-up colonoscopy
        self.screening_log.at[t, "colo"] += screening_pop.sum()

        # Get colonoscopy specs based on colo application type
        colo_specs = (
            inputs.SCREENING_TEST_SPECS["colo_followup"]
            if followup
            else inputs.SCREENING_TEST_SPECS["colo"]
        )

        # --- Complications ---
        complication_deaths = self._apply_colonoscopy_complications(
            screening_pop, t, is_screening=True
        )

        # Remaining alive portion proceeds to detection logic
        alive_after_colo = screening_pop - complication_deaths

        # --- Polyp detection / removal (return to healthy) ---
        # LR polyp → healthy
        lr_dx = alive_after_colo[state_idx["lr_polyp"]] * colo_specs["sens_lr_polyp"]
        self.D[t, state_idx["lr_polyp"]] -= lr_dx
        self.D[t, state_idx["healthy"]] += lr_dx
        self.I[t, state_idx["healthy"]] += lr_dx

        # HR polyp → healthy
        hr_dx = alive_after_colo[state_idx["hr_polyp"]] * colo_specs["sens_hr_polyp"]
        self.D[t, state_idx["hr_polyp"]] -= hr_dx
        self.D[t, state_idx["healthy"]] += hr_dx
        self.I[t, state_idx["healthy"]] += hr_dx

        # --- Preclinical cancer → screen-detected (stage 1..4) ---
        for i in range(4):
            u_state, d_state = f"u_stage_{i+1}", f"d_stage_{i+1}"
            u_state_idx, d_state_idx = state_idx[u_state], state_idx[d_state]

            # Apply detection probability
            screen_detect = (
                alive_after_colo[u_state_idx] * colo_specs[f"sens_stage_{i+1}"]
            )

            # Log detected cancers
            self.screen_detected[t, i] += screen_detect

            # Update distribution D and incidence I
            self.D[t, u_state_idx] -= screen_detect
            self.D[t, d_state_idx] += screen_detect
            self.I[t, d_state_idx] += screen_detect

    def _apply_alternative_test(self, test: str, population: np.ndarray, t: int):
        """
        Apply a non-colonoscopy screening test (e.g., FIT) and return GLOBAL positives.

        Arguments:
            test (str):
                Name of test in inputs.SCREENING_TEST_SPECS (e.g., 'FIT').
            population (np.ndarray):
                GLOBAL-length vector representing the current *screening-eligible* population at cycle t.
                Typically this is D[t] masked to screening states.
            t (int):
                Cycle index.

        Returns:
            np.ndarray: GLOBAL-length vector of test positives (mass only in screening states).
        """
        s = c.health_states_stoi
        test_specs = inputs.SCREENING_TEST_SPECS[test]

        # Log number of people screened at time t for this test

        screening_pop = population[c.screening_states_idx]
        self.screening_log.at[t, test] = screening_pop.sum()
        positives = np.zeros(len(screening_pop))
        positives[s["healthy"]] = population[s["healthy"]] * test_specs["fpr"]
        positives[s["lr_polyp"]] = (
            population[s["lr_polyp"]] * test_specs["sens_lr_polyp"]
        )
        positives[s["hr_polyp"]] = (
            population[s["hr_polyp"]] * test_specs["sens_hr_polyp"]
        )
        for i in range(4):
            positives[s[f"u_stage_{i+1}"]] = (
                population[s[f"u_stage_{i+1}"]] * test_specs[f"sens_stage_{i+1}"]
            )

        # Return vector length n_states with those who tested positive
        return positives

    def _apply_screening(self, t: int) -> None:
        """
        Args: t (int): current cycle
        Run the screening workflow for cycle t:
        - Build a GLOBAL vector limited to screening states from D[t].
        - Apply the first-line test per protocol.
        - If not colonoscopy, follow up positives with colonoscopy.
        """

        # Get population eligible for screening
        screening_pop = np.zeros_like(self.D[t])
        screening_pop[c.screening_states_idx] = self.D[t, c.screening_states_idx]

        # Get screening protocol for the current cycle
        test = self.screening_protocol[t]

        # Apply non-colo test, then follow-up colo for positives
        if test and test != "colo":
            positives = self._apply_alternative_test(test, screening_pop, t)
            self._apply_colonoscopy(positives, t, followup=True)

        # First-line test is colo
        else:
            self._apply_colonoscopy(screening_pop, t, followup=False)

    def normalize_distribution_vectors(self) -> None:
        """
        Normalize the distribution vectors to ensure they sum to 1.
        This is done for both the distribution and incidence vectors.
        """
        # Incidence and prevalence denominator is out of living population
        dead_factor = np.divide(
            c.POPULATION_SIZE,
            (c.POPULATION_SIZE - self.D[len(c.alive_states)].sum(axis=0)),
        )
        self.P = np.zeros_like(self.D)
        self.R = self.I.copy()
        for state in range(c.n_states):
            if c.health_states_itos[state] in c.alive_states:
                self.P[:, state] = np.multiply(self.D[:, state], dead_factor)
                self.R[:, state] = np.multiply(self.R[:, state], dead_factor)

        # Transform into annual counts. For incidence, we sum; for prevalence, we average.
        self.Iy = self.I.reshape(c.NUM_YEARS, 12, c.n_states).sum(axis=1)
        self.Ry = self.I.reshape(c.NUM_YEARS, 12, c.n_states).sum(axis=1)
        self.Dy = self.D.reshape(c.NUM_YEARS, 12, c.n_states).mean(axis=1)
        self.Py = self.I.reshape(c.NUM_YEARS, 12, c.n_states).mean(axis=1)

    # ---------- main run ----------
    def run_markov(self) -> None:
        """
        Execute the cohort simulation across cycles:
        - Apply transitions (D, I)
        - Apply colonoscopy complications to symptom-detected cancers
        - Apply screening per protocol when due
        - Age updates and layer changes
        - Final normalization and annual aggregation
        """
        s = c.health_states_stoi

        # Start entirely healthy
        start_state = np.zeros(c.n_states)
        start_state[s["healthy"]] = c.POPULATION_SIZE
        self.D[0] = start_state

        self.curr_age = self.spec.model_start_age
        self.curr_age_layer = 0
        t = 1

        inflow_T = np.tril(self.T, k=-1)  # to capture only inflows

        # Initialize screening-cycle flag for t=1
        self._update_screening_cycle(t)

        while t <= self.spec.cycles:

            # Apply state transtion matrix to get new pop distribution and incidence
            # (n_states x n_states) * (n_states x 1) -> (n_states x 1)
            self.D[t] = self.T[self.curr_age_layer, :, :] @ self.D[t - 1]
            self.I[t] = inflow_T[self.curr_age_layer, :, :] @ self.D[t - 1]

            # Symptom-detected cancers this cycle (detected states inflow)
            newly_detected = self.I[t, c.detected_states_idx].copy()  # length 4
            self.symptom_detected[t] = newly_detected

            # Complications for those newly symptom-detected (retroactive colo)
            # Build a GLOBAL vector placing newly_detected mass into detected indices
            newly_detected_global = np.zeros_like(self.I[t])
            newly_detected_global[c.detected_states_idx] = newly_detected
            _ = self._apply_colonoscopy_complications(
                newly_detected_global, t, is_screening=False
            )

            # Apply screening if applicable
            if not self.spec.strategy.is_NH and self.is_screening_cycle:
                self._apply_screening(t)

            # Advance time
            t += 1

            # Age and age-layer updates where necessary
            if t % 12 == 0:
                self.curr_age += 1
                if self.curr_age >= c.END_AGE:
                    break
                if self.curr_age in c.AGE_LAYERS.keys():
                    self.curr_age_layer += 1

            # Update screen-cycle flag for next t
            self._update_screening_cycle(t)

        self.normalize_distribution_vectors()
