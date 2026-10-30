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

2. **Contact-face tooling + a stable one-ball hold -- DONE.** One 45mm / 80g ball
   (`assets/scene_one_ball.xml`).

   - **Classifier**: `hand_env.classify_contacts` labels every hand contact on a body by digit,
     segment and FACE -- palmar (pad), DORSAL (back), side, or tip (the very end of a distal
     segment) -- from the contact point's offset from the segment's own long axis, measured against
     `palmar_axis`. `eval_stage2_classifier.py` touches the ball to every flexion segment from known
     directions: 65/65 labelled correctly.
   - **Hold test**: the hand closes on the ball, then gravity is pointed in six directions -- palm
     up, palm DOWN, fingers up, fingers down, either side down (the same physics as turning a
     fixed-base hand over). Held = the ball moves under 1cm in the palm frame over 2s.
   - **Grasp search** (`search_stage2_grasp.py`, cross-entropy method over the 18 finger and thumb
     targets): holding turned out to be the easy part -- most random samples already held in all
     six directions, by caging the ball with fingertip ENDS and the sides of segments. The search
     only found real wraps once the score rewarded pad contacts specifically (and penalized
     fingertip ends), and only found ones that STAY wraps once those contact terms were averaged
     over 7 perturbed copies of each candidate: the first 4-pad grasp lost its pads to the
     4th-decimal rounding of its own targets.
   - **Result** (`hand_env.ONE_BALL_GRASP`, `eval_stage2_hold.py`): mf, rf and lf wrap the ball
     with the palmar faces of their proximal and distal segments. Held in all six directions
     (drift 0.5-1.9mm), peak effort 22% of the force limits; under 100 perturbed trials (every
     target +-0.05 rad, placement +-3mm, mass 40-150g, friction 0.5-1.0) held 100/100, three pads
     on the ball in 85, no dorsal contact in any.

   <p align="center">
     <img src="media/stage2_grasp_d_robust_no_thumb.png" width="720">
   </p>

   What it does NOT do: the thumb lies across the palm, off the ball, and the ball is wrapped up
   near the knuckles rather than sitting in the palm. Raising the thumb's weight in the score got
   the thumb on only as a 0.3N graze (`media/stage2_grasp_e_robust_light_thumb.png`); placing the
   ball mid-palm and holding it while the hand closes (it rolls off an open hand) gave grasps that
   held just as well but with a single pad. Left there deliberately: one 45mm ball in a palm built
   for two is a poor stand-in for baoding, where the two touching balls fill the palm's whole
   length and the thumb's job is pushing, not pinching. The earlier search rounds are kept as
   `media/stage2_grasp_{a,b,c}_*.png`.
3. **Two-ball bowl, reviewed as images -- DONE.** Two touching 45mm balls
   (`assets/scene_two_balls.xml`). The shape is the user's: *"all 5 finger point up and curl a
   little inside and make bowl shape with two balls sit in the palm"*, then *"tips of 5 fingers
   almost like in same height"*, then *"make 4 fingers separated more to make gap between fingers
   and thumb smaller on both side"*. Found by `search_stage3_bowl.py` (CEM over all 20 targets plus
   where the pair is placed; both balls are held in place while the hand closes, then released),
   each round restricted and scored toward that shape and reviewed as a render
   (`show_stage3_bowl.py`):

   | round | image | what it showed |
   |---|---|---|
   | cradle | `stage3_cradle_a.png` | thumb and little finger on the outsides, but mf/rf curled under the balls -- no bowl |
   | hand-set | `stage3_bowl_manual.png` | the finger shape, but a LEVEL palm is open at the wrist side: one ball rolled out. The wrist became part of the search; tilting the palm ~25 deg toward the fingers settles the balls into the cup |
   | a | `stage3_bowl_a.png` | fingers standing up, thumb sticking out sideways |
   | b | `stage3_bowl_b.png` | five tips at one height (8.2-9.0cm); tips bunched (5.7cm span) |
   | c | `stage3_bowl_c.png` | an "even gaps" term was GAMED: the thumb moved in across the palm, the finger span shrank |
   | d | `stage3_bowl_d.png` | fingers forced to their widest (9.4cm span): they lie down, the bowl opens, balls separate |
   | **e** | `stage3_bowl_e.png` | **chosen**: moderate fan (6.8cm span), tips 8.0-8.9cm, balls held + touching in all perturbed runs |

   Why the fingers can't fan much wider: a finger's spread joint (J4) swings it about the palm's
   normal, and a finger standing up lies ALONG that axis -- spreading just twists it (0.2 rad moves
   the tip <= 0.3cm). To move a tip sideways the finger must lean, and leaning also tips it forward,
   out of the bowl wall. `hand_env.BOWL_GRASP`: ~1.4 N total finger force, so the balls are cradled,
   not clamped -- free to roll, which is the point.

   <p align="center">
     <img src="media/stage3_bowl_e.png" width="720">
   </p>

4. **Scripted baoding -- tried, does not circulate (negative result).** All five digits take
   part -- the user: *"not only little finger and thumb, all fingers should involved as needed to
   make baoding trick"*. Judged on each ball's OWN cumulative angle around the pair's starting
   midpoint (`hand_env.BallAngles`, palm frame), the worse of the two balls, worst value over the
   last 2.5s of a 10s run -- a swap needs both past pi.

   - **Press wave** (`hand_env.BaodingWave`, `search_stage4_baoding.py`): every digit presses in
     and relaxes once per cycle, phases ordering the presses round the rim, plus a small wrist
     circle. The first search scored the angle at the final instant and found a wave that just
     rocks the jammed pair +-0.4 rad and ends on the top of a rock; with the worst-over-a-cycle
     score the best is 0.25 rad: rocking.
   - **Circular fingertip motion** (`hand_env.BaodingCircle`, `--controller circle`) -- the user,
     on the press-wave GIF: *"fingers maybe should make circular move to push balls cross each
     other"*. Each digit's two joints are driven a quarter-cycle apart so the tip traces a loop;
     plus the finger spread joints. Best after 20 CEM rounds: 0.10 rad -- still rocking.
   - **Why: the ideal-push test** (`hand_env.push_forces`): the hand holds the bowl still while a
     1 N tangential force on each ball drives them round the midpoint. In bowl e the gold ball
     stalls after ~1/4 turn either way -- against the ff/mf proximal segments (the knuckle row:
     upright fingers make a wall there) or the BACK of the thumb's proximal segment, which lies
     across the palm's thumb/wrist corner. Adding this test to the bowl search
     (`search_stage3_bowl.py --circulate`, worst of 4 perturbed pushes; the first two versions were
     gamed by angle gained before a drop and by one lucky run in a mean): from bowl e 0.27-0.57 rad;
     weighting circulation 10x (bowl shape mostly lost, fingers off the balls) ~1.4 rad; 40mm balls
     (`--ball-radius 0.02`) 1.31-1.47 rad. In every bowl the thumb's base joint THJ4 sits at its
     limit. Adding a centering spring to the push changes nothing.

   So no FIXED bowl lets the balls pass each other here, and no fixed periodic motion clears the
   way: baoding needs digits that move out of a ball's path as it arrives (the thumb lifting as the
   gold ball comes round) and push it from behind -- motion keyed to where the balls are. That is
   Stage 5.

   <p align="center">
     <img src="media/stage4_circle.gif" width="580">
   </p>
   <p align="center"><sub>Stage 4: the best circular fingertip motion (8s, balls rock in place),
   then the bowl held still under the ideal 1 N push each way -- the gold ball jams after ~1/4
   turn. The press wave's version: `media/stage4_wave_and_push.gif`.</sub></p>
5. **RL baoding** from the reviewed grasp, reward keyed to same-direction per-ball progress and to
   contact face, only if the scripted stage shows what's physically possible and where it stalls.

## Repository layout

```
shadow_hand/
├── assets/
│   ├── right_hand.xml, scene_right.xml, keyframes.xml   # vendored, see NOTICE.md
│   ├── assets/                            # meshes (vendored)
│   ├── LICENSE.shadow_hand, CHANGELOG.shadow_hand.md, NOTICE.md
│   ├── scene_one_ball.xml                 # Stage 2: hand + one 45mm ball
│   └── scene_two_balls.xml                # Stage 3: hand + two 45mm balls
├── hand_env.py                            # names, palmar_axis, contact-face classifier, grasp/hold tests
├── eval_stage1_verify.py                  # Stage 1: tracking / palmar axis / coupling / gravity checks
├── demo_stage1_kinematic_sweep.py         # Stage 1: actuator sweep GIF
├── eval_stage2_classifier.py              # Stage 2: classifier probe check
├── search_stage2_grasp.py                 # Stage 2: CEM grasp search
├── eval_stage2_hold.py                    # Stage 2: six-direction hold + perturbation test
├── search_stage3_bowl.py                  # Stage 3: CEM bowl search (two balls)
├── show_stage3_bowl.py                    # Stage 3: palm-aligned render of a bowl
├── search_stage4_baoding.py               # Stage 4: CEM over the press wave / circular motion
├── demo_stage4_baoding.py                 # Stage 4: controller + ideal-push GIF
├── media/
│   ├── stage1_kinematic_sweep.gif
│   ├── stage2_grasp_*.png                 # grasp renders, a-e in search order (d = final)
│   ├── stage3_*.png                       # bowl renders (stage3_bowl_e = final)
│   └── stage4_*.gif                       # scripted baoding attempts + ideal-push test
└── README.md
```
