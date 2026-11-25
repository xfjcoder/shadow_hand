# Methods tutorial: how the Shadow Hand baoding project works

This is a guide to every method used in this repository, from the first model check to a camera-driven
student policy. It explains each idea, says why it was needed here, and points to the code and the
README stage where it was used. The README has the run-by-run results; this file covers the methods
behind them.

Contents

1. [The setting: MuJoCo, the hand, the task](#1-the-setting)
2. [Checking the model before trusting it](#2-checking-the-model)
3. [Measuring what matters: contact faces and progress metrics](#3-measuring-what-matters)
4. [Search without learning: the cross-entropy method (CEM)](#4-search-without-learning-cem)
5. [Feasibility tests: is the task physically possible?](#5-feasibility-tests)
6. [Reinforcement learning with PPO](#6-reinforcement-learning-with-ppo)
7. [Reward design, and how rewards get gamed](#7-reward-design)
8. [Curricula: making a hard task learnable](#8-curricula)
9. [Where episodes start: reset distributions](#9-where-episodes-start)
10. [Reusing trained policies: warm starts, widening, residual RL, proxies](#10-reusing-trained-policies)
11. [Planning on the simulator: sampling MPC with backtracking](#11-planning-on-the-simulator)
12. [Learning from demonstrations: behaviour cloning and motion tracking](#12-learning-from-demonstrations)
13. [Timing many balls: timetables, early termination, controllability](#13-timing-many-balls)
14. [From true state to senses: teacher-student distillation (Stage 8)](#14-teacher-student-distillation)
15. [Domain randomisation (Stage 8, in progress)](#15-domain-randomisation)
16. [How results are judged](#16-how-results-are-judged)
17. [Glossary](#17-glossary)

---

## 1. The setting

**Simulator.** [MuJoCo](https://mujoco.org) 3.x: rigid bodies, joints, contacts with friction, and
actuators. Every result here is from `mj_step` under gravity, never from forward kinematics alone.

**The hand.** The Shadow Hand E3M5 from `mujoco_menagerie`, vendored unmodified (`assets/`). It has 24
joints and 20 actuators. Each finger's two end joints (J2, J1) share one actuator through a tendon, as
in a human finger. The torque limits are real ones (fingers ±1 N·m, thumb 1-3, wrist 5-10), so every
result has to work within them. The base is fixed and the palm faces up.

**The task.** Baoding: balls circulating in the palm, pushed by the fingers. The project worked up from
one ball (Stage 2) to two balls swapping (Stage 5), three balls (Stage 6), two hands with four balls
on a figure-8 (Stage 7), and finally sensing the balls with a camera instead of reading the simulator
(Stage 8).

**Control interface.** Actuators are position servos: the policy sets a target angle and MuJoCo's servo
produces the torque. Control runs every 20 ms (`CONTROL_DT`), with several physics substeps in between.

---

## 2. Checking the model

Before building anything, check that the model behaves as you think it does (`eval_stage1_verify.py`,
Stage 1):

- **Tracking:** sweep each actuator and measure the steady-state error. A large error can be a contact
  rather than a bug. Here the little finger's spread joint ran into the ring finger.
- **Sign conventions:** a "spread" joint may move the finger the opposite way to what its name
  suggests. Measure directions rather than assuming them.
- **Derived geometry:** `hand_env.palmar_axis` computes each segment's inner (pad) face as
  *flexion axis × segment long axis*, rather than hard-coding it. This catches per-segment differences;
  the thumb's middle segment differs from the fingers.
- **Compile-time constants:** MuJoCo computes some solver constants (`invweight0`) at compile time.
  Resizing balls at run time without `mj_setConst` silently changes the dynamics. A Stage 5 policy failed
  in the Stage 7 two-hand scene until the balls were compiled the same way (README Stage 7B, "a trap").

Lesson: if a replay of the same actions diverges from the first step, the two models differ. Find out
why before training anything.

---

## 3. Measuring what matters

A good metric is one that can't be satisfied by the wrong behaviour. Several metrics here were rewritten
after a policy or search found a loophole.

**Contact faces** (`hand_env.classify_contacts`, Stage 2). Each hand-ball contact is labelled with the
digit, the segment and the face: palmar (pad), dorsal (back), side, or tip. The face is found from the
contact point's offset from the segment's axis, measured against `palmar_axis`. A plain contact count was
satisfied by the thumb pressing with its back. Checked by touching every segment from known directions:
65/65 correct.

**Per-ball angle, worst ball** (`hand_env.BallAngles`, Stage 4). Measure each ball's own cumulative angle
around the centre. The vector between the two balls also turns when one ball merely wobbles.

**Progress = the minimum** (Stage 5). Swap progress is the least of each ball's arc and the rotation of
the line between them. Rocking, one ball circling alone, or jiggling at the centre all earn nothing.

**Worst over a window, not the final instant.** A search scored on the last frame found motions that end
on the peak of a rock (Stage 4). Score the worst value over the last cycle. Over several runs, score the
worst run, not the mean, because one lucky run inflates a mean.

**Real gaps, not proxy errors** (Stage 7). A 0.1-lap timing error looked small, but measured as actual
surface-to-surface gaps the balls were touching 23% of the time. Measure the thing you care about directly.

---

## 4. Search without learning: CEM

The **cross-entropy method** is a simple derivative-free optimiser:

1. Keep a Gaussian over the parameters, with a mean and a standard deviation per dimension.
2. Sample N candidates and score each one in simulation.
3. Refit the Gaussian to the best K candidates (the "elites").
4. Repeat until the scores converge.

Uses in this project:
- **Grasps** (`search_stage2_grasp.py`): 18 finger targets that hold a ball under gravity in six directions.
- **Bowls** (`search_stage3_bowl.py`): all 20 targets plus where the balls are placed.
- **Periodic motions** (`search_stage4_baoding.py`): the phases and amplitudes of a finger wave.
- **Open-loop escape tests** and the planner's inner optimiser (section 11).

**Perturbation-robust scoring.** Score each candidate as the average over several perturbed copies:
targets ±0.05 rad, ball placement ±3 mm, mass, friction. Without this, the first 4-pad grasp lost its pads
when its targets were rounded to 4 decimals. Robust scoring finds solutions that stay solutions.

**Restricting the search to a shape.** For the bowl, each search round was constrained toward the shape
the user described and reviewed as an image (`show_stage3_bowl.py`). An "even gaps" term was gamed by
moving the thumb inwards, so a person looking at the result is part of the method.

---

## 5. Feasibility tests

Before spending hours training, check that the task can be done at all.

- **Ideal push** (`hand_env.push_forces`, Stages 4-6). Hold the hand still and apply an ideal tangential
  force directly to each ball. If even that can't make the balls pass each other, no controller using that
  hand posture can. This showed that no fixed bowl allows a swap, so fingers must move out of the way
  (which became the reason for Stage 5's RL).
- **Holdability** (`eval_stage6_circle_hold.py`). For every phase of the target circle, CEM searches for a
  posture that holds the balls still. If some phase can't be held, tracking it is hopeless.
- **Open-loop escape** (Stage 5). From a stuck state, CEM over finger trajectories asks whether the stuck
  state is a physical wall or a learning problem. It found 0.04 rad possible with the old joint limits and
  2.31 rad with wider limits. Widening the limits then let RL get past it.
- **Controllability probe** (`probe_stage7_fig8.py`, Stage 7). From states of a policy's own rollouts,
  nudge each actuator and measure how each ball's position along the path responds. This gives a
  ball × actuator gain matrix. A push aimed with those gains separates two balls on the same hand 91% of
  the time (chance is 50%). So timing balls one by one is physically possible, and the failure to learn it
  was a learning problem. The probe also showed which actuators matter (the wrists and forearm rolls), and
  that led to the per-ball actions (section 13).
- **Open-loop replay** (`eval_stage6_demo_replay.py`). Replaying a demo's own controls from saved frames
  finds moments that are knife-edge (nothing could survive them), so they aren't blamed on the policy.

---

## 6. Reinforcement learning with PPO

**PPO** (Proximal Policy Optimization, Schulman et al. 2017; implementation: stable-baselines3) is the
learning algorithm for every policy from Stage 5 onward.

- **Policy and value networks:** MLPs. The policy outputs the mean of a Gaussian over actions, and a
  learnable log standard deviation sets the exploration noise. The value network estimates future reward,
  which lowers the variance of the policy gradient.
- **Loop:** run N environments in parallel (15-16 here, one process each, `SubprocVecEnv`), collect a batch
  of steps, compute advantages with GAE (generalised advantage estimation), then take several epochs of
  clipped policy-gradient steps. The clip keeps each update close to the previous policy.
- **Observation normalisation** (`VecNormalize`): running mean/std of each input. It is saved with the
  checkpoint and has to be loaded with it.
- **Deterministic evaluation:** at test time, use the mean action without noise.

Practical lessons from running it on a CPU:
- `torch.set_num_threads(1)` in each worker. Without it, 16 processes × 16 threads fight each other, and
  throughput fell from ~1200 to ~430 steps/s.
- Physics is the bottleneck, at about 1.5-2k steps/s on 16 cores.
- Exploration noise has to suit the task. With 40 actuators, a noise std of 0.37 dropped a ball within
  3 s, so Stage 7 started lower (log std −1.5).

---

## 7. Reward design

RL optimises exactly what you write, so most of the work was in the reward. Lessons, in the order they
were learned (README Stage 5 table):

1. **Rewards get gamed.** A per-step "matched angle" reward was satisfied by jiggling the balls at the
   centre. The fix was the minimum-based progress measure (section 3).
2. **A flat drop penalty makes "score, then drop" optimal.** Make the drop cost scale with the time left
   in the episode, or make it large compared with what can still be earned (−200 to −400 in Stage 7).
3. **Pay only for the goal; express style as limits and penalties.** Rewards for pad-facing, five pads on,
   relay order and ball-on-palm made the policy look right while barely moving the balls: style paid ~180
   per episode, a swap ~51. The rebuilt reward (`BaodingEnv2`) pays positively only for ball motion.
   Style became constraints: joint caps, a fingertip band, penalties for dorsal contact.
4. **Credit must be bounded by progress.** Uncapped "work" credit (force × speed) paid for pushing
   harder, not further. Capping the work credit per radian of progress fixed it (`BaodingEnv5`).
5. **Constraints can make the task impossible.** A 1.5 cm fingertip band ruled out the finger motion a
   swap needs, which the feasibility tests revealed (section 5). It was relaxed in steps to 3.5 cm, then
   6 cm.
6. **Shared credit** (`BaodingEnv9`): each digit earns at most 30% of the work cap, to spread the pushing
   across fingers. It helped modestly.
7. **Dense versus sparse shaping.** A tracking reward exp(−(err/0.05)²) gives no gradient a quarter lap
   out; (1 + cos 2π·err)/2 pays something everywhere (Stage 7 timetable).
8. **Make the reward measure the real failure.** The `--clear` term scales each ball's pay by its actual
   gap to the nearest ball (0 when touching, full at 1 cm). That cut touching from 23% to 6% (fig8_27).
9. **Watch for idle equilibria.** If being cut off late costs only the reward forgone, standing still and
   being cut off is a safe policy. Several runs (fig8_16, 19, 21) collapsed into it. Fixes: pay for
   movement during the spin-up, and stop tightening the tolerance past what the policy can hold.

---

## 8. Curricula

A **curriculum** starts with an easy version of the task and makes it harder as the policy succeeds.
Each one here is driven by a measured pass rate, not a fixed schedule:

- **Ball size** (Stage 5): 40 → 42 → 44 → 45 → 47 → 48 → 50 mm, advancing when 50% of the last 200
  episodes swap (`Curriculum` callback). Training directly at 50 mm never swapped. Going through the
  curriculum reached 10+ swaps per 20 s.
- **Speed** (Stage 6): the demo is played back at 1×…10×, one level per pass rate. It can also step
  *down* (`--down`, 2× → 1×) to move a fast policy to a slower speed.
- **Tolerance** (Stage 7, `SlotCurriculum`): the allowed timing error tightens from 0.4 to 0.15 laps
  once the median episode lasts long enough (10 s, later 22 s). With too short a threshold, it tightened
  on noise.
- **Number of balls** (Stage 7): 1 → 2 → 3 → 4, each run warm-started from the one before (section 10).

Side effect to watch for: a policy that follows the curriculum forgets the earlier levels (Stage 5 at
40 mm; Stage 6, one speed per policy).

---

## 9. Where episodes start

What the policy practises is largely decided by where episodes begin.

- **Randomised starts:** random start arrangements, ball mass, friction and target noise, with a filter
  that redraws starts where a ball is still rolling.
- **Demo-moment starts** (Stage 6): begin at a random moment of a demonstration, as in DeepMimic's
  "reference state initialisation".
- **Reverse curriculum from hard states** (`collect_stage6_jam_states.py`): start in or just before the
  jam. The policy learned to hold at the jam (drops 13/16 → 1/16) but not to pass it.
- **Failure-focused starts** (`--hard-frac`, Stage 6 step 8): each environment remembers where it lost a
  ball and restarts 1-3 s before that moment, or from the policy's own state ~2 s before. It practises
  recovering from its own mistakes, which took 60 s survival from 13/24 to 20/24.
- **Stall starts** (Stage 7): the same idea for stalls instead of drops.
- **State bank / mid-run starts** (`bank_stage7_fig8.py`, `--bank --p-bank`): save hundreds of
  good-circulation states from a working policy, and start half the episodes there. Once the tolerance
  is tight, episodes from a standing start end in 3-5 s, so without the bank almost all training would be
  the spin-up.

---

## 10. Reusing trained policies

- **Warm start / resume** (`--resume`): continue training from a checkpoint, often with a changed
  environment. Most runs here continue an earlier one.
- **Network widening** (`widen_stage7_fig8.py`): to go from n to n+1 balls, widen the first layer of
  both networks. The new inputs start at zero weight, and their normalisation is copied from ball 0. The
  widened policy initially behaves exactly like the old one and ignores the new ball. The same trick adds
  new inputs (timetable phase) or new outputs (per-ball tilts, also at zero weight) without changing
  behaviour at step 0.
- **Residual RL** (Stage 7): the policy's action is added to a scripted base motion (the finger wave,
  the rhythm). The script provides a sensible default, and RL learns the corrections. A finger-wave script
  alone kept one ball 4/10; RL on top of it lapped it 15 times in 30 s.
- **Proxy environments and mirroring** (`stage7_two_hands.py`): run a single-hand policy on each of two
  hands. Each hand's state is copied, mirrored for the left hand, into a proxy env that never steps
  physics, so the policy sees exactly its training observation. It worked with no retraining (7/8 kept).

---

## 11. Planning on the simulator

When RL stalled (three balls, Stage 6), the simulator itself was used as a model to plan with.

**Predictive sampling / CEM-MPC** (`stage6_teacher.py`, `stage6_mpc.py`, `stage7_teacher.py`). As in
MuJoCo MPC (Howell et al. 2022):
1. From the current true state, sample many finger trajectories (5 waypoints over the next move).
2. Simulate each one forward on a copy of the world and score it: ball targets reached, no loss, costs.
3. Refine with CEM, execute the best plan, then re-plan from the new state.

Additions that mattered:
- **Labelled "clicks":** each ball aims for its own next position a third of a turn on, instead of small
  15° steps. This was taken from a human tutorial ("going uphill, then they fall in place").
- **Backtracking:** when every plan loses a ball, back out one or two moves and search again. This was the
  breakthrough, from ~300° to 605°, then 825° in 71 s.
- **Cost terms** add style without hurting progress: a fingertip-contact cost (`--tip-weight`) gave pads
  only; a hard posture band (`--band`) gave smoother fingers in Stage 7.

A planner is not a controller. It needs a perfect copy of the world and about 3.5 h of compute for 71 s
of motion. Its output becomes a **demonstration** for learning (section 12).

---

## 12. Learning from demonstrations

**Behaviour cloning (BC):** supervised learning of the demo's actions from the demo's states. It was used
as a warm start, never on its own. BC alone dropped a ball within 1-9 s because of compounding error:
small mistakes lead to states the demo never visited. (DAgger, section 14, is the standard fix.)

**Motion imitation by RL** (`rl_env_track3.py`, `rl_env_stage7.py`; DeepMimic, Peng et al. 2018). The
demo supplies *states*, not actions:
- Each ball tracks its own position in the demo at that moment, with reward exp(−cm/1.5).
- The observation shows the target now and 0.5 s ahead.
- Episodes start at random demo moments, and the demo loops for long runs.
- PPO finds its own finger motions. From one demo moment, the policy survives where replaying the demo's
  own controls loses a ball.

Lessons:
- **Timing matters.** Targets moving at a constant speed stalled the curriculum, because the balls pass a
  jam with a hop. Following the demo's own timing worked.
- **Episode length should match the test.** Policies trained on 10 s episodes lost balls at 20-46 s in
  60 s tests. Training on 30 s, then 60 s, episodes fixed that.
- **One demo, one speed, one direction.** A policy trained on one demo scored 0/24 on a different one.
- **"Fixes" from the planner** (Stage 7): re-planning from where the policy gets stuck and cloning those
  fixes damaged the policy. Mixing them at a low learning rate was safe but slow. It's a DAgger-like idea
  that didn't pay off there.

---

## 13. Timing many balls

Several balls on one figure-8 have to keep their spacing, or they bunch into a clump and collide at the
crossing.

- **Timetable** (`--track-rate`): a reference clock runs around the 8, and each ball is paid for staying
  at its slot (offset) on it.
- **Natural-pace clock** (`--pace data/pace_fig8_3.npy`): balls move about 5× slower around the outer
  walls than through the crossing. A clock that is even in path length misreads that as falling behind and
  catching up every lap. A clock built from the balls' measured pace removed that false error.
- **Early termination** (`--slot-tol`, as in DeepMimic): a ball further than the tolerance from its slot
  ends the episode, with the tolerance tightening over training. Staying on time becomes staying alive.
- **Spin-up grace** (`--grace 3`): for the first 3 s the clock waits on the balls, which start at rest.
- **Waiting clock at test time** (`--lead 0.1`): the clock never leads the balls by more than 0.1 lap.
  This keeps the policy within the error range it was trained on. Watch out: timing errors wrap around, so
  a ball a full lap behind reads as on time. Also, a waiting clock can make a policy look slower than it
  is, so judge on the training clock (four-ball evaluation, `run4.py`).
- **Per-ball actions** (`--ball-act`, `Fig8Env._tilt`): one action per ball, which tilts the hand that ball
  is on so that gravity pushes it along the 8's direction there. This gives the policy a direct handle on
  each ball, based on the controllability probe. Its strength (`--tilt-k`) was tuned so a full push
  actually unsticks a stalled ball.
- **Slot offsets:** chosen by grid search over the 8's geometry, so that no two balls meet at the crossing
  (three at thirds; four as a "train", 0/.21/.42/.63).

---

## 14. Teacher-student distillation

Every policy up to Stage 7 read each ball's exact position and velocity from the simulator. A real hand
can't. Stage 8 moves the three-ball policy onto senses using a **teacher-student** scheme, also known as
learning with privileged information or "learning by cheating" (Chen et al. 2019). It is standard for
legged robots and dexterous hands (e.g. Lee et al. 2020; Chen et al. 2022). The teacher is
fig8_27 @104M, which sees the true state. The student sees only realistic inputs and learns by imitating
the teacher.

### 14.1 Sensors (`stage8_sense.py`)
- **Camera:** a 64×64 RGB overhead render with shadows and reflections off (0.7 ms per frame on the CPU
  with EGL), and the balls coloured red, green and blue. 64 px keeps rendering and the tracker cheap; each
  ball is about 6 px across.
- **Touch:** 48 pads, one per hand body, each reading the total normal force from ball contacts
  (`mj_contactForce`).
- **Realism:** a delay line per sensor (N steps of 20 ms), pixel noise, proportional touch noise, and a
  touch dead-band of 0.02 N.

### 14.2 How sensitive is the teacher? (`eval_stage8_noise.py`)
- Noisy positions are passed through an **alpha-beta filter**, a fixed-gain cousin of the Kalman filter.
  It predicts `x̂ = x + v·dt`, then corrects `x += α·r` and `v += β·r/dt`, where `r` is the measurement
  minus the prediction. With latency, the estimate is predicted forward by the delay.
- Result: the teacher survives 0.1 cm of noise but fails at 1 cm, unless it gets the true velocity. So
  velocity noise is what breaks it. Delay hurts even with no noise: 24/40 kept at 20 ms, 16/40 at 40 ms.
  The teacher can't be wired straight to a camera, which motivates the student.

### 14.3 The ball tracker (`collect_stage8.py`, `train_stage8_tracker.py`)
A CNN from image (and optionally touch) to 3D ball positions.
- **Data:** 120 teacher episodes, about 125k frames labelled with the true positions. The teacher acts
  with some randomness and, in some episodes, on noisy ball state, so the data covers off-nominal states.
- **CoordConv** (Liu et al. 2018): two extra input channels hold each pixel's x and y coordinates.
  Ordinary convolutions are translation-invariant and find it hard to output positions.
- **Heat map per ball + spatial soft-argmax** (Levine et al. 2016): a softmax over each heat map, then the
  expected pixel coordinate. It is differentiable and sub-pixel. That's how a 64-px image locates a ball to
  about 1 mm.
- **Head:** an MLP over the coordinates, the heat-map peaks (confidence), pooled features and touch,
  giving 3D positions in cm.
- **Training:** Smooth L1 loss, Adam with a cosine schedule, brightness and noise augmentation.
  Validation is on held-out *episodes*. Random held-out frames would be nearly copies of training frames.
- **Result:** median error 0.13 cm with touch (20 epochs), 0.31 cm camera-only (8 epochs).

### 14.4 Teacher on the tracker (`eval_stage8_tracker.py`)
The pipeline is camera → tracker → (filter) → teacher. The timing clock also runs on estimates, and no
true state reaches the policy. Results: 103/120 kept with no delay, 72/120 at 40 ms. The filter made
things worse at every delay. The tracker is accurate enough; latency is the remaining problem.

### 14.5 The student and DAgger (`train_stage8_student.py`)
- **Inputs:** the tracker's estimates from the last K = 4 frames (with a flag for whether each frame is
  new), log-compressed touch, joint angles and velocities, and the rhythm and clock phases. The frame
  history lets the network learn velocity and how far ahead to predict to cover the delay, which the
  teacher can't do.
- **Outputs:** 42 hand residuals and the 4 wrist/forearm tilts directly. (The teacher's per-ball tilts are
  converted to those 4 using the true ball positions, which the student doesn't have.)
- **Network:** an MLP with two hidden layers of 512 and stored input mean/std.

**Why not plain behaviour cloning?** A student trained only on the teacher's states hits unfamiliar states
as soon as it makes a small error, and the errors compound. Round 0 (pure cloning) kept 10/40 with almost
no turning.

**DAgger** (Dataset Aggregation, Ross, Gordon & Bagnell 2011):
1. Round 0: the teacher drives. Record the student's input and the teacher's action at every step.
2. Round k: at each step the teacher acts with probability β_k = 0.5^k, otherwise the student does.
3. Whoever acts, label every visited state with the teacher's deterministic action computed from the
   **true state**: what the teacher would do here.
4. Keep all data from all rounds, and retrain the student on the whole set (MSE, warm-started).
5. Test the student alone after each round.

The student therefore learns how to recover from its own mistakes, labelled by an expert that can see the
truth. DAgger's guarantee is that the training distribution converges to the states the student itself
visits.

**Choosing the round:** results jump between rounds (71-96/120), so the round is chosen on one seed set
(7000+) and reported on another (50000+, 120 episodes).

**Result at a 40 ms camera delay:** the camera-only student kept 104/120 at 2.64 laps per 30 s, against
the teacher's 72/120 on the same camera.

### 14.6 Ablations
Remove one input at a time to see what carries the information (`--cam-drop`, `--no-cam`, `--no-touch`):

| student (40 ms camera) | kept /120 | laps / 30 s |
|---|---|---|
| camera + touch | 91 | 2.32 |
| **camera only** | **104** | **2.64** |
| camera + touch, 30% of frames lost | 72 | 1.38 |
| touch only | 46 | 0.08 |

The camera carries the information. Touch adds nothing measurable here: the pads show *that* a ball is on
a segment, not *where* along the 8 it is.

---

## 15. Domain randomisation

**Problem:** everything above used one fixed, clean camera. Networks trained on a single look often fail
when the lighting, colours or camera mount change, which is a large part of the sim-to-real gap.

**Domain randomisation** (Tobin et al. 2017): randomise appearance in every training episode, so the
network has to rely on what stays the same (where the balls are). A real camera then looks like one more
variation. `Senses(dr=1.0)` redraws the following at every reset, visual only, with physics unchanged:

| what | range | models |
|---|---|---|
| camera position / height | ±6 mm / ±2 cm | mounting error |
| camera tilt / roll | up to 5° / ±6° | calibration error |
| light brightness / tint | 0.64-1.57× / ±12% per channel | room lighting |
| light positions and directions | moved | room lighting |
| hand and floor shades | ±25% (±8% per channel) | materials |
| ball colours | shade ±25%, hue drift | paint |
| pixel noise / defocus / smear | up to 8/255 / 3×3 blend / up to 35% of the previous frame | sensor, lens, exposure |
| gamma / white balance | ×0.82-1.22 / ±8% | camera processing |

**Measured so far:**
- The clean-trained camera-only student, unchanged, keeps 71/120 at half-strength randomisation and
  53/120 at full strength, with almost no turning.
- The cause is the tracker, whose error rises from 0.13-0.31 cm to 1.3-1.6 cm on randomised frames.
- 128k randomised frames have been collected (`collect_stage8.py --dr 1`).

**Next:** retrain the tracker on randomised frames, run DAgger again with `--dr 1` (it can warm-start with
`--init`), and test on both randomised and clean cameras.

**Not yet modelled:** lens distortion, occlusion by other objects, and identical-looking balls. With
identical balls, telling them apart would mean tracking over time.

---

## 16. How results are judged

- **Fresh seeds, many episodes.** 40 episodes are fine for a first look, but the score swings by about
  ±4 balls kept. Claims are made on 80-120 episodes. The best checkpoint is chosen on one seed set and
  confirmed on another.
- **Deterministic policy** (mean action) at test time, unless stated otherwise.
- **Test what you trained for.** A 60 s test exposes policies trained on 10 s episodes. A waiting clock
  can make a policy look slower than it is. A wrapped error can hide a ball a whole lap behind.
- **Look at it.** Every new behaviour gets a GIF (`media/`, compressed; `media_full/`, full size), and
  several failures were only spotted by watching: fingertips instead of pads, a little finger turned
  around, balls parked on one palm.
- **Report the failures.** The README keeps negative results (Stage 4, the Stage 7A packed square, the
  fix-cloning runs). They explain why the methods that worked are designed the way they are.

---

## 17. Glossary

| term | meaning |
|---|---|
| actuator / servo | a position-controlled motor; the policy sets target angles |
| alpha-beta filter | a fixed-gain position/velocity tracker (a simplified Kalman filter) |
| behaviour cloning (BC) | supervised learning of an expert's actions from its states |
| CEM | cross-entropy method: sample, keep the elites, refit a Gaussian, repeat |
| CoordConv | a CNN given pixel-coordinate channels as extra inputs |
| curriculum | training on easy versions of a task first, made harder as performance allows |
| DAgger | Dataset Aggregation: the student drives, the expert labels, all data is kept |
| DeepMimic | RL that tracks reference motion states; reference-state starts; early termination |
| domain randomisation | randomising the simulator's appearance/physics so the policy generalises |
| early termination | ending an episode when the agent drifts too far, so staying close is staying alive |
| GAE | generalised advantage estimation, the variance-reduced advantage used by PPO |
| privileged information | inputs the teacher gets in simulation but a real robot does not (the true state) |
| PPO | Proximal Policy Optimization, a clipped policy-gradient RL algorithm |
| predictive sampling / MPC | plan by simulating sampled action sequences, execute the best, re-plan |
| residual RL | the policy's action is added to a scripted base controller |
| soft-argmax | the expected coordinate under a softmax over a heat map; differentiable, sub-pixel |
| teacher-student | a privileged expert trains a sensor-only policy by imitation |
| timetable / slot | a reference clock round the path; each ball's assigned offset on it |
| VecNormalize | running mean/std normalisation of observations (saved with the checkpoint) |
| warm start / widening | continuing from a trained model; adding inputs/outputs at zero weight |

**References** (the methods' original sources):
Schulman et al. 2017, *Proximal Policy Optimization Algorithms* ·
Ross, Gordon & Bagnell 2011, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger) ·
Peng et al. 2018, *DeepMimic* ·
Tobin et al. 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* ·
Chen et al. 2019, *Learning by Cheating* ·
Liu et al. 2018, *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution* ·
Levine et al. 2016, *End-to-End Training of Deep Visuomotor Policies* (spatial soft-argmax) ·
Howell et al. 2022, *Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo* ·
Rubinstein 1999, *The Cross-Entropy Method* ·
Chiappa & Mathis et al. 2024, MyoChallenge baoding (SDS curriculum).
