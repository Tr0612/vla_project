# Behavior

Started with SigLip2 and BERT for vision and text encoding .

I trained basketball-v3, button-press-topdown-v2, button-press-topdown-v3, door-open-v2, door-open-v3, drawer-close-v2, drawer-close-v3, drawer-open-v2, peg-insert-side-v2, peg-insert-side-v3, pick-place-v2, pick-place-v3, push-v3, reach-v3, sweep-v3, window-open-v3. In this the robot didnt learn pulling motion hence fails pulling stick world.
It succeeds the seen tasks.The baseline small VLA learns coarse manipulation behaviors on some tasks, but struggles on precision-contact tasks such as peg insertion and pull-based tasks. This suggests limitations in action grounding and skill specialization.