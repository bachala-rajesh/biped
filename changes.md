### improvement of isaac lab training code


##### changes made
- in the simple_biped_config.py
  - solver_position_iteration_count increased from 4 to 8
  - joint effort limit decreased from 300 to 20 Nm
  - velocity limits decreased from 100 to 20 rad/s
  - damping increased from 2.5 to 4.0
  - increased solver velocity count to 8. intially it was 0
  - rigid props made to same as bipedal_locomotion_lab repo
  - 

##### rsl_rl_ppo_cfg.py
- the empricial normalization : false --> true
- changed the init_noise_std to 0.7 (default 1.0)


##### base_env_cfg.py
- explicitly defind the joint names in Action class
- reduced the velocity in y dirn
- ###### events
    - increased the dynamic friction range (0.7, 0.9) --> (0.4, 0.9)
    - increased the damping distribution params (2.0, 3.0) --> (3.0, 4.0)
    - decrease the push force from 500N --> 100N
    - changed the reset velocity ranges
        - decreased vel in y dirn
        - decreased the range of RPY
      - 





##### Todo in futire
- incease the range of velocity in y dirn
- remove height penality to walk at different height
- add more termination criteria i.e angle tilt more than specific degree where the robot is impossible to recover
- add randomize gravity as an event term
- increase penality for sudden increase in action_rate
- 
- 