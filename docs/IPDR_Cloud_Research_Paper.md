# IPDR Cloud Research Paper

*Extracted text from original PDF: [IPDR_Cloud_Research_Paper.pdf](file:///Users/hayderr/Documents/PERSONAL FAILURES/ipdr cloud/docs/IPDR_Cloud_Research_Paper.pdf)*

---


## Page 1

ML-Based Proactive Disaster Recovery in
Single-Tier Cloud Systems Using Deep Q-Network
(IPDR-Cloud)
Haider Iqbal
Department of Cyber Security
Ghulam Ishaq Khan Institute of
Engineering Sciences and Technology (GIKI)
Topi, Pakistan
Shehroz Majeed
Department of Cyber Security
Ghulam Ishaq Khan Institute of
Engineering Sciences and Technology (GIKI)
Topi, Pakistan
Abstract—This paper presents IPDR-Cloud, an Intelligent
Proactive Disaster Recovery framework for single-tier cloud
systems. Existing cloud disaster recovery approaches are reactive
as they measure Recovery Time Objective (RTO) and Recovery
Point Objective (RPO) only after failures have occurred. We
address three critical gaps found across a systematic review of 16
papers: RTO/RPO are never predicted before failure, machine
learning models degrade under non-stationary workloads, and
connection of failure prediction to backup triggering in a single
end-to-end pipeline in existing system now a days. IPDR-Cloud
replaces static rule-based scheduling with a Deep Q-Network
(DQN) agent that learns an optimal adaptive backup policy
from an 8-feature state space derived from Google Cluster Trace
2019 data. Experimental results show our Deep Q-Network –
Reinforcement Learning (RL) achieves 96.5% SLA compliance
exceeding the 94% Paper 10 Tabular Reinforcement Learning
and reduces mean RPO to 0.5 minutes with zero SLA breaches
across all evaluated failure events. Outperforming of reactive
baselines of system on all four-target metrics while adapting to
workload shifts continuously that degrade fixed-policy alterna-
tives.
Index Terms—Disaster Recovery, Deep Q-Network, Reinforce-
ment Learning, Cloud Computing, RTO/RPO, SLA Compliance,
Google Cluster Trace 2019, Non-Stationary Workloads.
I. INTRODUCTION
Cloud computing infrastructure now underpins critical busi-
ness operations across every sector. When cloud services fail,
organisations face two hard deadlines: the Recovery Time
Objective (RTO), which defines how long a service can remain
unavailable, and the Recovery Point Objective (RPO), which
defines the maximum acceptable age of data that can be lost.
Together these form the backbone of Service Level Agree-
ments (SLAs), and breaching either one carries significant
financial and reputational consequences.
The fundamental problem is that every system in the current
literature reacts to failures after they have already happened.
The RTO clock only starts counting once the system has
already crashed. But by that point, the damage in terms of
downtime and the data loss is already accumulating. This is
analogous to a doctor who only begins treatment after the
patient has died, rather than taking action or remedy on early
symptoms or warning signs.
The research community has recognised pieces of this prob-
lem. Paper 9 predicts failures at the edge with 87% accuracy
and reduces RTO by 45%, but never triggers a backup in result
of prediction. Paper 10 optimises backup scheduling using
reinforcement learning and achieves 94% SLA compliance,
but the authors explicitly acknowledge that their tabular Q-
learning model degrades when workload distributions shift
a ubiquitous property of real production clusters. Paper 16
introduces adaptive RTO tiers through hard-coded severity
rules and achieves 89% compliance but cannot adapt to failure
patterns it was not programmed to handle a random value.
Our work unifies these three partial solutions into one
complete system. IPDR-Cloud uses a Deep Q-Network to
simultaneously predict failure severity and determine the ap-
propriate backup schedule, with prediction directly triggering
the backup. The agent is trained on patterns from the Google
Cluster Trace 2019, a 31-day record of real production work-
loads that displays precisely the non-stationary behaviour that
breaks prior approaches. The three main contributions are:
1) A proactive disaster recovery pipeline that connects ML-
based failure prediction to adaptive backup scheduling.
2) A DQN agent that approximates the optimal Q-function
over a continuous 8-feature state space, handling non-
stationary cloud workloads without periodic retraining.
3) Simultaneous optimisation of both RTO and RPO within
a single reward signal a combination absent in all 16
surveyed papers.
The remainder of this paper is structured as follows. Section
II reviews related literature. Section III describes the system
architecture. Section IV presents the DQN model and baseline
equations. Section V details the experimental setup. Section VI
presents and discusses results. Section VII concludes.
II. LITERATURE REVIEW
A systematic review of 16 papers published between 2020
and 2025 across IEEE Xplore, MDPI, and IJACSA was con-
ducted. Papers were categorised into three thematic clusters:
cloud and edge disaster recovery, IoT and edge data pipelines,


---


## Page 2

and critical infrastructure resilience. Three recurring gaps were
identified across all clusters.
A. Cloud Disaster Recovery
Abualkishik et al. surveyed 11 DR solutions and found that
most lack formal RTO/RPO benchmarks. Paper 5 reduced RTO
by approximately 40% using edge-tier buffering but did not
evaluate cascading failures or RPO guarantees. Paper 6 of-
floaded recovery workloads to edge nodes and reduced latency
by 35%, but ML-predictive failover was entirely absent. Paper
8 introduced severity-based resource scaling and improved
RTO by 28%, but operated without an edge tier and did not
test non-stationary conditions.
B. ML-Based Failure Prediction
The most directly relevant prior work is Paper 9, which
deployed ML anomaly detection at the network edge to predict
failures 30 to 60 seconds ahead, achieving 87% prediction
accuracy and a 45% RTO reduction versus a reactive baseline.
However, the system never triggers a backup — the prediction
and recovery pipelines are completely separate. Paper 10
takes the complementary approach: a reinforcement learning
agent that learns an optimal backup schedule, reaching 94%
SLA compliance. The critical limitation, stated by the authors
themselves, is that the tabular Q-function degrades when traffic
patterns shift, because the state space it was trained on no
longer matches the distribution encountered in deployment.
C. Rule-Based Adaptive Recovery
Paper 16 introduced adaptive RTO targets by classifying
failure severity into low, medium, and high tiers using hand-
coded CPU and error-rate thresholds, achieving 89% RTO
compliance. The approach is interpretable and computationally
cheap, but rules cannot generalise to failure signatures they
were not designed for, and RPO is never measured. Our
work is distinguished from all three baselines by the combi-
nation of learned policy, connected pipeline, and simultaneous
RTO/RPO optimisation.
III. METHODOLOGY
IPDR-Cloud is implemented as a four-component modular
Python system. Figure illustrates the complete data flow. Each
component maps to a discrete responsibility, and the interfaces
between them are narrow by design.
A. Workload Generator
The workload generator loads Google Cluster Trace 2019
data from two tables: instance usage.csv, which records CPU
and memory utilisation across thousands of production jobs
over 31 continuous days, and instance events.csv, which
records failure events. Usage data is aggregated into 10-second
bins using mean CPU, memory, and disk I/O across all active
instances. Network latency and error rate are derived from
resource pressure using empirical relationships observed in
the Borg cluster literature. The generator steps through this
aggregated stream one time step at a time and yields the raw
metric vector to C2.
B. DQN Prediction Engine
The prediction engine maintains a rolling buffer of the last
ten metric readings, from which it constructs the 8-dimensional
state vector fed to the DQN agent. The agent selects one of
three backup actions at each step. The network architecture
is a fully connected feed-forward network with four layers: 8
input neurons, two hidden layers of 64 and 32 neurons with
ReLU activations, and 3 output neurons corresponding to the
Q-values of each action. The agent uses experience replay with
a buffer of 10,000 transitions and a target network updated
every 10 episodes via hard copy of online weights.
C. Adaptive Backup Scheduler
The scheduler receives the DQN action and executes the
corresponding backup policy. Action 0 (NORMAL) sets a
backup interval of 30 simulation steps. Action 1 (W ARNING)
reduces this to 10 steps and triggers an immediate backup if
the last backup exceeds that threshold. Action 2 (CRITICAL)
triggers an immediate backup regardless of when the last one
occurred and sets a 2-step interval going forward.
D. Metrics Engine
The metrics engine records every failure event and backup
event with precise timestamps. When a failure occurs, it
computes RPO as the elapsed time since the most recent
backup, and RTO as a function of the data replay burden — a
linear combination of a fixed restore base time and the data-
age at the time of failure. SLA compliance is evaluated per
failure event, requiring both RTO and RPO to fall within their
respective targets.
IV. DQN MODEL AND BASELINE EQUATIONS
A. State Space
The state vector s R 8 is constructed at each time step t
from eight telemetry features:


---


## Page 3

TABLE I
TABLE I. STATE SPACE FEATURES
Feature Unit Description
cpu util % Mean CPU utilisation over 10-step
window
mem util % Mean memory utilisation over 10-
step window
disk io MB/s Mean disk I/O throughput
net latency ms Derived network latency from re-
source pressure
error rate % Derived request error rate
backup age min Time elapsed since last completed
backup
sla breach risk 0–1 Composite forward-looking breach
risk score
workload trend -1 to 1 CPU slope over window (positive
= accelerating)
B. Action Space
The DQN selects from three discrete backup actions A =
{0, 1, 2}at each step:
•Action 0 — NORMAL:back up every 30 steps, moni-
toring should be standard.
•Action 1 — W ARNING:back up every 10 steps if
pressure exceeded trigger should be immediated.
•Action 2 — CRITICAL:back up immediately after two
steps.
C. Baseline Equation
Our primary baseline is Paper 10, which uses a tabular Q-
learning agent with the Bellman update equation:
Q(s, a)←Q(s, a)+α
h
r+γ·max
a′
Q(s′, a′)−Q(s, a)
i
(1)
where is the learning rate, is the discount factor, r is
the immediate reward, and s’ is the successor state. The
fundamental limitation of this formulation is that Q(s,a) is
stored as a lookup table indexed by discrete state-action pairs.
When the workload distribution shifts — as it inevitably does
in real production clusters — the stored Q-values correspond
to a distribution the agent is no longer operating in. The
authors of Paper 10 explicitly identify this as a failure mode
of their system.
Our DQN replaces the tabular Q-function with a neural
network approximator parameterised by weights :
Q(s, a;θ)≈Q ∗(s, a)(2)
where s R 8 is the continuous feature vector and Q*(s,a) is
the true optimal Q-function. The neural network generalises
across the continuous state space rather than requiring discrete
enumeration, which is precisely what allows the agent to
handle novel workload patterns unseen during training. The
weights are updated using the DQN loss:
L(θ) =E

r+γ·max
a′
Q(s′, a′;θ −)−Q(s, a;θ)
2
(3)
where denotes the frozen target network weights, updated
every 10 episodes. This stabilises training by decoupling the
target from the rapidly changing online network.
D. Reward Function
The reward signal R(t) balances two competing objec-
tives: protecting SLA compliance and controlling unnecessary
backup overhead:
R(t) =



−10,if SLA breached
+1,else if SLA met at failure
−0.1,else if backup triggered
+0.01,otherwise (idle step)
The breach penalty of 10 dominates the backup cost of
0.1 by a factor of 100, which reflects the real-world cost
asymmetry between cloud downtime and backup storage. The
ratio prevents the degenerate policy of constant CRITICAL
backups while ensuring that any SLA breach is overwhelm-
ingly negative. The +1 reward at successful failure recovery
directly incentivises the proactive prediction behaviour that is
the core contribution of this work.
E. Comparison Metrics
Three quantitative targets derived from the literature are
used to evaluate our system:
RTO Reduction= RTOreactive −RTO ours
RTOreactive
>45%(4)
SLA Compliance= |{f:RTO f ≤T RTO ∧RPO f ≤T RPO}|
|F| >94%
(5)
RTO Compliancerule = |{f:RTO f ≤T RTO}|
|F| >89%(6)
where F is the set of all failure events, T RTO = 5 minutes,
and T RPO = 15 minutes. RTO and RPO for each failure event
f are defined as:
RPOf =t failure −t last backup (7)
RTOf =t recovered −t failure (8)
V. EXPERIMENTAL SETUP
A. Dataset
All experiments use the Google Cluster Workload Trace
2019 (Borg Cluster Trace), a publicly available dataset record-
ing 31 continuous days of production workload data from a
large-scale Google cluster. The trace captures genuine non-
stationary behaviour — traffic spikes from batch job submis-
sions, day-night usage cycles, and abrupt regime shifts — that
synthetic datasets cannot replicate. We use the instance usage
and instance events tables, extracting the first 7 days for
training and reserving the remaining 24 days for evaluation.
Task failure events of types EVICT, FAIL, KILL, and LOST
are treated as ground-truth failure events.


---


## Page 4

B. DQN Hyperparameters
TABLE II
TABLE II. DQN HYPERPARAMETERS
Hyperparameter Value
Network architecture 8→64→32→3 (ReLU)
Optimizer Adam (lr = 1×10³)
Replay buffer 10,000 transitions
Batch size 64
Discount factor 0.95
Initial epsilon 1.0
Epsilon decay 0.995 per episode
Minimum epsilon 0.01
Target net update Every 10 episodes (hard copy)
Training episodes 150
C. Baseline Protocols
1)Reactive DR:Back up schedule is of fixed 60 minutes
and only after failure has detected the recovery starts. It
represents the worst case intelligence.
2)Rule-Based DR:Paper 16 replica. CPU ¿ 80% triggers
W ARNING; error rate ¿ 8% or CPU ¿ 95% triggers
CRITICAL. Fixed backup interval multipliers per sever-
ity class.
3)DQN-Based DR (Ours):on 150 episodes of trace data
DQN is trained. From three action policy backup action
is selected.
VI. RESULTS AND DISCUSSION
A. Protocol Comparison
Table III summarises the performance of all three protocols
across the four primary metrics. Our DQN achieves 96.5%
SLA compliance exceeding the 94% Paper 10 target with a
mean RTO of 2.1 minutes and mean RPO of 0.5 minutes. Crit-
ically, the DQN run recorded zero SLA breaches, compared to
five for the reactive baseline and four for the rule-based sys-
tem. The rule-based protocol marginally outperforms reactive
on SLA compliance (92.8% vs 90.7%) because the CPU and
error-rate thresholds do capture some failure precursors, but
the static rules miss the multi-feature commerce patterns that
the DQN has learned to honor.
TABLE III
TABLE III. PROTOCOL COMPARISON RESULTS
Protocol SLA% RTO(min) RPO(min) Breaches
Reactive 90.7% 2.3 1.3 5
Rule-Based 92.8% 2.2 1.1 4
DQN-Ours 96.5% 2.1 0.5 0
Paper 9 Target — ¿45% reduction — —
Paper 10 Target ¿94% — — —
Paper 16 Target ¿89% — — —
Fig. 1. SLA compliance rates across three protocols. Dashed lines indicate
Paper 10(94%) and Paper 16(89%) standard targets. DQN-Ours achieves 96.5,
surpassing both marks.
Fig. 2. Mean RTO and RPO per protocol. DQN reduces mean RPO to 0.5 min
through visionary backup driving before failure appearance, a 62.7 reduction
versus the reactive base.
B. Learning Dynamics
Figure 3 presents the DQN accretive price over 150 training
occurrences. In the early stages (around occurrences 1–80),
prices show high change and frequent negative values due to
expansive disquisition under a high epsilon rate. After episode
100, the rolling average increases steadily, indicating that the
agent is learning associations between precursor patterns and
CRITICAL actions.


---


## Page 5

Fig. 3. Cumulative reward per episode over 150 training episodes (faint:
raw, bold: 15-episode rolling average). The agent transitions from random
exploration to a learned policy as epsilon decays.
Fig. 4. Average reward per episode with 50-episode smoothing. Reward
stabilises after episode 100, indicating policy convergence within the training
budget.
C. Non-Stationary Adaptation
The DQN approach compared to the tabular RL method
used in Paper 10. At simulation step 1000, a workload shift
changes the CPU and memory utilisation patterns. As a
result, the fixed-policy baseline from Paper 10 experiences a
noticeable decline in SLA compliance because it was trained
on the earlier workload distribution. In contrast, the DQN
model adapts by updating its Q-network using new experiences
and recovers to above 90% compliance within nearly 20 steps.
This outcome confirms the limitation previously identified by
the authors of Paper 10.
Fig. 5. Fig. 4. Sliding-window SLA compliance over simulation time for
DQN-Ours vs Paper 10 fixed RL policy. Workload shift injected at step
1000. DQN recovers within approximately 20 steps; fixed policy degrades
and recovers slowly.
D. Reward Weight Ablation
To estimate the price function, the DQN was trained using
three different breach-penalty and backup-cost settings. As
shown in Figure 7 and Table IV , the Conservative setup
(penalty 20, cost 0.05) achieved 99.7% SLA compliance but
resulted in a higher average RTO because the agent performed
backups too constantly. The Aggressive setup (penalty 5, cost
0.50) reduced backup frequency, but this caused a slight drop
in SLA compliance. The Balanced configuration (penalty 10,
cost 0.10) provided the best overall performance, achieving a
2.1-minute RTO with 100% SLA compliance, making it the
favored product setting.
TABLE IV
TABLE IV. REWARD WEIGHT ABLATION
Config Penalty Bkp Cost SLA% RTO (min)
Conservative 20 0.05 99.7% 2.9
Balanced 10 0.10 100.0% 2.1
Aggressive 5 0.50 100.0% 2.1


---


## Page 6

Fig. 6. Fig. 7. SLA compliance (%) and mean RTO (min) for three
reward configurations. Balanced configuration achieves best RTO at full SLA
compliance and is selected as the production setting.
E. Feature Correlation Analysis
The Pearson correlation between the eight state features
and SLA breach events. The strongest feature is backup age
(—r— = 0.905), indicating that longer backup delays increase
RPO risk during failures. The sla breach risk feature follows
with —r— = 0.612, confirming its usefulness as a predictive
metric. The other six features have weaker individual correla-
tions (0.18–0.39), suggesting their value mainly comes from
non-linear interactions that neural networks can learn better
than rule-based systems. Small variations may also appear due
to noise in the trace data.
Fig. 7. Fig. 8. Pearson —r— of each feature against SLA breach occurrence.
backup age (0.905) and sla breach risk (0.612) dominate; remaining features
contribute through non-linear interactions captured by the DQN.
VII. CONCLUSION
This paper introduced IPDR-Cloud, applying a Deep Q-
Network on proactive disaster recovery (DQN) to learn adap-
tive backup policies from real cloud workload traces. Three
major limitations are addressed and identified in the 16-
paper review: before failure occurs RPO/RTO be predicted,
adapting to changing workload distributions that affect fixed-
policy methods, and integrating failure prediction with backup
scheduling in a unified pipeline.
Experiments using the Google Cluster Trace 2019 dataset
show that IPDR-Cloud achieved 96.5% SLA compliance,
surpassing the 94% benchmark reported in Paper 10, while
reducing mean RPO to 0.5 minutes with zero SLA breaches.
The reward-function analysis further showed that the balanced
configuration (penalty 10, backup cost 0.10) provided the best
trade-off between reliability and operational cost. In addi-
tion, the non-stationary workload experiment demonstrated
the main advantage of the DQN approach, as it adapted to
workload shifts within nearly 20 simulation steps, whereas the
fixed-policy baseline showed longer performance degradation.
Future work will focus on extending IPDR-Cloud to a three-
layer edge–fog–cloud environment, where backup decisions
must consider different latency and storage limitations. A
multi-agent design, in which each layer operates its own DQN
agent while coordinating through shared rewards, is planned
as the next stage of the research.
REFERENCES
[1] IEEE Authors, ”Proactive Failure Detection and Recovery in Edge-
Cloud Systems,” IEEE Trans. Network Service Manage., DOI:
10.1109/TNSM.2024.10648015, 2024.
[2] IEEE Authors, ”Cloud Backup Optimization Using ML for RTO/RPO
Management,” IEEE Access, DOI: 10.1109/ACCESS.2020.9045528,
2020.
[3] IJDSIS Authors, ”Towards Adaptive Recovery Time Objectives in Hybrid
Cloud Systems,” Int. J. Distributed Systems and Information Security, V ol.
4, 2023.
[4] Google Research, ”Google Cluster Workload Traces 2019 (Borg Cluster
Trace),” github.com/google/cluster-data, 2019.
[5] A. Z. Abualkishik et al., ”Disaster Recovery in Cloud Computing Sys-
tems: An Overview,” IJACSA, V ol. 11, No. 9, 2020.
[6] IEEE Authors, ”Edge Computing for Resilient Cloud Backup Systems,”
IEEE Access, DOI: 10.1109/ACCESS.2024.10443484, 2024.
[7] IEEE Authors, ”Intelligent Fault-Tolerant Cloud Recovery with Edge
Offloading,” IEEE TNSM, DOI: 10.1109/TNSM.2024.10382350, 2024.
[8] IEEE Authors, ”Adaptive Cloud Resource Management for Disaster
Recovery,” IEEE Access, DOI: 10.1109/ACCESS.2025.10822992, 2025.


---
