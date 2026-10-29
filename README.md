# Shadow Hand Dexterous Manipulation

A fixed-base Shadow Hand E3M5 (24 joints, 20 actuators, 5 fingers) dexterous manipulation project
in MuJoCo -- started right after closing `allegro_hand`, whose main goal (baoding: two balls
circulating around each other in one hand) didn't work on a 4-finger hand.

## Scoping decisions

- **Why the Shadow Hand, and why now.** On the Allegro, every baoding attempt (scripted and RL)
  ran into the hardware: no little finger, a "spread" joint that actually rolls each finger
  about its own length instead of spreading it, and a thumb that can't oppose the fingers well.
  Real baoding leans on exactly those: the thumb and little finger push the balls around a palm
  cupped by the other fingers. The Shadow Hand has all three -- a fifth finger with an extra
  metacarpal joint for cupping (LFJ5), real sideways spread joints (J4), a 5-joint thumb -- and is
  already in `mujoco_menagerie`, so no model-building step.
- **Fixed base, palm up, no arm.** The vendored model lies palm-up out of the box (forearm along
  world +x, palm facing +z, a dropped object lands in the palm), the same working orientation the
  Allegro project used. The 2-joint wrist is part of the model and stays available.
- **Realistic torque limits are kept.** Unlike the Allegro model (no limits at all -- its toss and
  catch needed 1.5 N*m with nothing stopping it), this one caps finger actuators at +-1 N*m, the
  thumb at 1-3 N*m (3 at its base, 2 at its proximal, 1 at the rest), the wrist at 5-10 N*m. Results here are judged inside those limits.
- **Carried over from the Allegro project:** classify every contact by WHICH FACE of WHICH segment
  touches (a contact count was satisfied by the thumb pressing with its back); measure each ball's
  own angle, not the vector between them (one ball wobbling moves that vector too); show the
  starting grasp as an image and get it reviewed before any long training run; stress-test
  controllers under parameter perturbation, not just varied conditions.
- **Simulator: MuJoCo** (`~/robo/venv`, mujoco 3.15), same as every sibling project.

## Staged plan

1. **Vendor the model and verify it under real physics -- DONE.** Copied `mujoco_menagerie`'s
   `shadow_hand` (right hand only, 3.5MB, unmodified -- see `assets/NOTICE.md`).
   `eval_stage1_verify.py` checks, with `mj_step` under gravity rather than forward kinematics:

   - **Tracking**: each of the 20 actuators swept to 25% and 75% of its range (others at rest):
     median steady-state error 0.015 rad. The worst, 0.109 rad, is the little finger's spread
     joint (LFJ4) at 75% -- not a tracking fault: at that angle it swings the little finger INTO
     the ring finger (they touch), and with kp=1 the 0.1 N*m pushback shows up 1:1 as position
     error. Positive LFJ4 moves the little finger toward the ring finger, i.e. inward -- the same
     sign trap the Allegro's spread joint had, so check directions before assuming "spread".
   - **Palmar (inside) face of every segment**: derived from the model, not hardcoded --
     `hand_env.palmar_axis` = (flexion joint axis) x (segment's long axis). It's local -y for the
     fingers but local -x for the thumb's middle segment. All 15 flexion segments move along their
     derived axis when flexed.
   - **Coupled finger joints**: each finger's middle and distal joints (J2, J1) share ONE actuator
     through a fixed tendon, like a human finger. Commanding 1.5 rad splits 0.73 / 0.77.
   - **Gravity**: the scene's ellipsoid object, dropped on the open hand, comes to rest in the palm
     at z=0.038 (palm origin z=0.010), speed 0.0003 m/s.

   <p align="center">
     <img src="media/stage1_kinematic_sweep.gif" width="420">
   </p>
   <p align="center"><sub>Stage 1: each actuator swept to the top and bottom of its range and back
   (real-time playback), then the object dropped into the open palm.</sub></p>

2. **Contact-face tooling + a stable one-ball hold.** Port the Allegro's contact classifier
   (segment x face: palmar / dorsal / side / tip) to this model, using `palmar_axis`. Grasp one
   human-sized ball (40-50mm) palm-up and hold it under the torque limits, verified by which faces
   touch, not just whether the ball stays put.
3. **Two-ball cupped hold, reviewed as an image first.** Two touching balls resting in the palm,
   the fingers cupped around them with their pads on the balls, the little finger and thumb on the
   outsides -- found by a static search scored on contact face, shown to the user before anything
   is built on it.
4. **Scripted baoding.** The human technique: little finger and thumb push the balls around a
   circuit while the middle fingers form the cup. Judged on each ball's own cumulative angle
   around the palm center, and on whether the two balls actually swap places.
5. **RL baoding** from the reviewed grasp, reward keyed to same-direction per-ball progress and to
   contact face, only if the scripted stage shows what's physically possible and where it stalls.

## Repository layout

```
shadow_hand/
├── assets/
│   ├── right_hand.xml, scene_right.xml, keyframes.xml   # vendored, see NOTICE.md
│   ├── assets/                            # meshes (vendored)
│   ├── LICENSE.shadow_hand, CHANGELOG.shadow_hand.md, NOTICE.md
├── hand_env.py                            # joint/actuator/segment names, palmar_axis
├── eval_stage1_verify.py                  # Stage 1: tracking / palmar axis / coupling / gravity checks
├── demo_stage1_kinematic_sweep.py         # Stage 1: actuator sweep GIF
├── media/
│   └── stage1_kinematic_sweep.gif
└── README.md
```
