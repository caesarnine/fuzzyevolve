from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterable
from collections.abc import Sequence

from concurrent.futures import ThreadPoolExecutor, as_completed

from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Elite, MutationCandidate
from fuzzyevolve.core.ports import MutationOperator
from fuzzyevolve.core.pool import CrowdedPool, cosine_distance

log_mutation = logging.getLogger("mutation")


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    role: str
    min_jobs: int
    weight: float
    uncertainty_scale: float
    committee_size: int = 1
    partner_selection: str = "far_random"
    partner_farthest_k: int = 32


@dataclass(frozen=True, slots=True)
class MutationJob:
    operator: str
    focus: str | None = None
    partners: tuple[Elite, ...] = ()


class MutationPlanner:
    def __init__(
        self,
        *,
        specs: Iterable[OperatorSpec],
        jobs_per_iteration: int,
        rng: random.Random,
    ) -> None:
        self.specs = list(specs)
        if not self.specs:
            raise ValueError("At least one operator is required.")
        self.jobs_per_iteration = max(0, jobs_per_iteration)
        self.rng = rng

        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("Operator names must be unique.")

        for spec in self.specs:
            if spec.min_jobs < 0:
                raise ValueError("Operator min_jobs must be >= 0.")
            if spec.weight <= 0:
                raise ValueError("Operator weight must be > 0.")
            if spec.uncertainty_scale < 0:
                raise ValueError("Operator uncertainty_scale must be >= 0.")

    def plan(self, critique: Critique | None) -> list[MutationJob]:
        if self.jobs_per_iteration <= 0:
            return []

        jobs: list[MutationJob] = []
        remaining = self.jobs_per_iteration

        # 1) Allocate per-operator minimums.
        for spec in self.specs:
            if remaining <= 0:
                break
            count = min(spec.min_jobs, remaining)
            jobs.extend(MutationJob(operator=spec.name) for _ in range(count))
            remaining -= count

        # 2) Allocate remaining jobs by weights.
        if remaining > 0:
            names = [spec.name for spec in self.specs]
            weights = [spec.weight for spec in self.specs]
            chosen = self.rng.choices(names, weights=weights, k=remaining)
            jobs.extend(MutationJob(operator=name) for name in chosen)

        self.rng.shuffle(jobs)

        # 3) Assign per-job focus to reduce within-iteration mode collapse.
        routes = list(critique.routes) if critique else []
        issues = list(critique.issues) if critique else []
        route_idx = 0
        issue_idx = 0

        for idx, job in enumerate(jobs):
            role = self._role_for(job.operator)
            if role == "explore":
                focus = None
                if routes:
                    if route_idx < len(routes):
                        focus = routes[route_idx]
                        route_idx += 1
                    else:
                        focus = self.rng.choice(routes)
                jobs[idx] = MutationJob(operator=job.operator, focus=focus)
            elif role == "exploit":
                focus = None
                if issues:
                    if issue_idx < len(issues):
                        focus = issues[issue_idx]
                        issue_idx += 1
                    else:
                        focus = self.rng.choice(issues)
                jobs[idx] = MutationJob(operator=job.operator, focus=focus)

        return jobs

    def _role_for(self, operator_name: str) -> str:
        for spec in self.specs:
            if spec.name == operator_name:
                return spec.role
        return "unknown"


class OperatorMutator:
    """Runs a per-iteration mutation plan across multiple operators."""

    def __init__(
        self,
        *,
        pool: CrowdedPool,
        operators: dict[str, MutationOperator],
        specs: Iterable[OperatorSpec],
        jobs_per_iteration: int,
        rng: random.Random,
    ) -> None:
        self.pool = pool
        self.operators = dict(operators)
        self.specs = {spec.name: spec for spec in specs}
        self.planner = MutationPlanner(
            specs=self.specs.values(),
            jobs_per_iteration=jobs_per_iteration,
            rng=rng,
        )
        self.rng = rng

        missing = [name for name in self.specs if name not in self.operators]
        if missing:
            raise ValueError(f"Missing operators for specs: {sorted(missing)}")

    def propose(
        self,
        *,
        parent: Elite,
        critique: Critique | None,
        max_candidates: int,
        mutation_executor: ThreadPoolExecutor | None = None,
    ) -> list[MutationCandidate]:
        if max_candidates <= 0:
            return []

        jobs = self.planner.plan(critique)
        if not jobs:
            return []

        jobs = self._attach_partners(jobs, parent)

        def run_job(job: MutationJob) -> list[MutationCandidate]:
            spec = self.specs.get(job.operator)
            job_critique = None if (spec and spec.role == "crossover") else critique
            job_focus = None if (spec and spec.role == "crossover") else job.focus
            operator = self.operators[job.operator]
            partner_texts = tuple(p.text for p in job.partners)
            texts = operator.propose(
                parent=parent,
                partners=(job.partners or None),
                critique=job_critique,
                focus=job_focus,
            )
            candidates: list[MutationCandidate] = []
            spec = self.specs[job.operator]
            for text in texts:
                cleaned = text.strip()
                if not cleaned or cleaned == parent.text:
                    continue
                candidates.append(
                    MutationCandidate(
                        text=cleaned,
                        operator=spec.name,
                        uncertainty_scale=spec.uncertainty_scale,
                        focus=job_focus,
                        partner_texts=partner_texts,
                    )
                )
            return candidates

        results: list[MutationCandidate] = []
        if mutation_executor is None or len(jobs) <= 1:
            for job in jobs:
                try:
                    results.extend(run_job(job))
                except Exception:
                    log_mutation.exception("Operator '%s' failed.", job.operator)
        else:
            futures = [mutation_executor.submit(run_job, job) for job in jobs]
            for fut in as_completed(futures):
                try:
                    results.extend(fut.result())
                except Exception:
                    log_mutation.exception("Mutation job failed; skipping.")

        # Dedupe preserving order.
        seen: set[str] = set()
        unique: list[MutationCandidate] = []
        for cand in results:
            if cand.text in seen:
                continue
            seen.add(cand.text)
            unique.append(cand)

        if len(unique) > max_candidates:
            unique = self.rng.sample(unique, k=max_candidates)
        return unique

    def _attach_partners(
        self,
        jobs: Sequence[MutationJob],
        parent: Elite,
    ) -> list[MutationJob]:
        out: list[MutationJob] = []
        for job in jobs:
            spec = self.specs.get(job.operator)
            if spec is None or spec.role != "crossover":
                out.append(job)
                continue

            count = max(0, int(spec.committee_size) - 1)
            partners = self._sample_partners(
                parent,
                count=count,
                selection=str(spec.partner_selection),
                farthest_k=int(spec.partner_farthest_k),
            )
            out.append(
                MutationJob(operator=job.operator, focus=job.focus, partners=partners)
            )
        return out

    def _sample_partners(
        self,
        parent: Elite,
        *,
        count: int,
        selection: str,
        farthest_k: int,
    ) -> tuple[Elite, ...]:
        if count <= 0:
            return ()

        candidates = [
            elite for elite in self.pool.iter_elites() if elite.text != parent.text
        ]
        if not candidates:
            return ()

        k = min(count, len(candidates))
        if selection == "random":
            return tuple(self.rng.sample(candidates, k=k))

        scored = [
            (cosine_distance(parent.embedding, e.embedding), e) for e in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        if selection == "farthest":
            return tuple(e for _dist, e in scored[:k])

        if selection == "far_random":
            top = min(int(farthest_k), len(scored))
            subset = [e for _dist, e in scored[:top]]
            return tuple(self.rng.sample(subset, k=min(k, len(subset))))

        raise ValueError(f"Unknown partner selection '{selection}'.")
