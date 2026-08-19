# Experience-Driven Framework

The three components the paper adds on top of a stock coding agent, so that
experience accumulated inside one session survives context truncation and
carries across sessions.

| Component | Directory | Role |
|---|---|---|
| Experiment Journal | [`journal/`](journal/) | Append-only decision log the agent writes before and after every experiment |
| Skill Library | [`skills/`](skills/) | Validated, reusable procedures the agent reads when stuck and writes when it learns |
| Evaluator Agent | [`evaluator/`](evaluator/) | A separate agent that predicts, measures, and analyses every checkpoint |

All three are *agent-facing*: they are specified in natural language in the
prompts, and their contents are produced by the agent at run time. Nothing here
is enforced by code — which is precisely what makes the resulting artifacts
evidence about agent behaviour rather than about the harness.

The concrete instances produced during the runs analysed in the paper are
archived under [`../trajectories/`](../trajectories/); the skill library and
wiki checked in under `skills/` are the state at the end of the experiment
campaign.
